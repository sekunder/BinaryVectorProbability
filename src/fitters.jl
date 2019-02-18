

"""
    first_order_model(X, I; kwargs...)

Returns a `BernoulliCodeDistribution` with `p_i` equal to the expected value of
bit `i` in `X[I,:]`.

"""
function first_order_model(X, I=1:size(X,1); kwargs...)
    mu = sum(X[I,:],2) / size(X,2)
    return BernoulliCodeDistribution(mu[:]; indices=I, autocomment="first_order_model", kwargs...)
end

"""
    second_order_model(X, I; kwargs...)

Returns an `IsingDistribution` which is fit to the pairwise correlations in `X[I,:]`.

Some useful keywords particular to this function:
 * `J0` is the initial value used for optimization

keyword argument `algorithm` sets the algorithm:
 * `algorithm = LBFGS()` default is to use the LBFGS algorithm in the `Optim` package.
 * `algorithm = :LD_LBFGS` uses LBFGS algorithm in `NLopt`
 * `algorithm = :naive` uses the function `gradient_optimizer` in `optimizers.jl`
 * `algorithm = :LD_MMA` uses the MMA algorithm as described in the `NLopt` package. Don't use this one it's hella slow.
"""
function second_order_model(X, I=1:size(X,1);
    verbose=0, algorithm=LBFGS(), kwargs...)
    # algorithm=:LD_LBFGS

    if algorithm == :naive
        return Naive_second_order_model(X, I; verbose=verbose, kwargs...)
    # elseif typeof(algorithm) == Symbol
    #     return _NLopt_second_order_model(X, I; verbose=verbose, algorithm=algorithm, kwargs...)
    else
        return Optim_second_order_model(X, I; verbose=verbose, algorithm=algorithm, kwargs...)
    end
end
function Naive_second_order_model(X, I=1:size(X,1); verbose=0, kwargs...)
    dkwargs = Dict(kwargs)
    Xselected = X[I,:]
    N_neurons,N_samples = size(Xselected)

    Jseed = rand(N_neurons,N_neurons); Jseed = (Jseed + Jseed') / (2 * N_neurons)
    Jseed = pop!(dkwargs, :J0, Jseed)
    mu = Xselected * Xselected' / N_samples
    L_X(J,g) = loglikelihood(Xselected, J, g; mu_X = mu)
    K_X(J,g) = MPF_objective(Xselected, J, g)
    fun = (N_neurons > ISING_METHOD_THRESHOLD || pop!(dkwargs, :force_MPF, false)) ? "MPF" : "loglikelihood"
    objective = (fun == "loglikelihood") ? :max : :min

    if verbose > 0
        println("second_order_model[gradient_optimizer]: setting $objective objective $fun")
    end
    # verbosity = verbose + pop!(dkwargs, :more_verbose, false)
    if fun == "loglikelihood"
        if verbose > 0
            println("\tApproximate cost: 2^$N_neurons + $N_samples per evalution")
        end
        (F_opt, J_opt, stop) = gradient_optimizer(L_X, Jseed[:]; objective=objective, verbose=verbose, dkwargs...)
    else
        if verbose > 0
            println("\tApproximate cost: $N_neurons * $N_samples per evaluation")
        end
        (F_opt, J_opt, stop) = gradient_optimizer(K_X, Jseed[:]; objective=objective, verbose=verbose, dkwargs...)
    end
    J_opt = reshape(J_opt, N_neurons, N_neurons)
    return IsingDistribution(J_opt; indices=I, autocomment="second_order_model[gradient_optimizer|$fun]", opt_val=F_opt, opt_ret=stop, dkwargs...)
    # return (F_opt, J_opt, stop, Jseed, mu, (fun == "loglikelihood" ? L_X : K_X))
end
function Optim_second_order_model(X, I=1:size(X,1); verbose=0, res=[], kwargs...)
    dkwargs = Dict(kwargs)
    _X = sortcols(X[I,:]) # sorting columns for branch prediction and memory access
    N_neurons,N_samples = size(_X)

    # J_prev_K = similar(J0)
    # J0_K = similar(J0)
    # J0_K[:] = J0[:]
    # J0_K[1:(N_neurons+1):end] = 0.0
    # buf_K = [_X, 2 * _X - 1, similar(_X, Float64), diag(J0), J0_K]
    # K_X(J) = K_MPF(J, J_prev_K, buf_K)
    # dK_X!(G, J) = dK_MPF!(G, J, J_prev_K, buf_K)

    alg = pop!(dkwargs, :algorithm, LBFGS())

    # Configure various options for Optim.jl
    # show_trace = pop!(dkwargs, :show_trace, verbose >= 2)
    options = Optim.Options(
        show_trace = pop!(dkwargs, :show_trace, verbose >= 2),
        x_tol = pop!(dkwargs, :x_tol, 0.0),
        f_tol = pop!(dkwargs, :f_tol, 0.0),
        allow_f_increases = pop!(dkwargs, :allow_f_increases, false),
        iterations = pop!(dkwargs, :iterations, 500),
        show_every = pop!(dkwargs, :show_every, 10),
        store_trace = pop!(dkwargs, :store_trace, false)
        )

    if verbose > 0
        println("second_order_model[Optim/$(summary(alg))]: preparing to run optimization")
        # println("  objective: $fun")
        print("  algorithm: $(summary(alg))")
        if typeof(alg) <: LBFGS
            println(" m = $(alg.m)")
        else
            println()
        end
        println("  N_neurons: $(size(_X,1))")
        println("  N_samples: $(size(_X,2))")
        #TODO potentially other information should be printed
    end

    # println("Using Minimum Probability Flow to estimate parameters...")
    # tic()
    # res_K = optimize(K_X, dK_X!, J0, alg, options)
    # t_K = toq()
    # J_K = Optim.minimizer(res_K)
    # println("Done! MPF $(Optim.converged(res_K) ? "did" : "did not") converge in $t_K s")
    # show(res_K)
    # println()

    J0 = -rand(N_neurons,N_neurons)
    J0 = (J0 + J0')/2
    J0 = pop!(dkwargs, :J0, J0)
    W0 = lt2v(J0)

    W_prev_L = similar(W0)
    mu_X = _X * _X' / N_samples
    buf_L = [_X, mu_X, IsingDistribution(J0), similar(W0)]
    L_X(W) = negloglikelihood(W, W_prev_L, buf_L)
    dL_X!(G, W) = dnegloglikelihood!(G, W, W_prev_L, buf_L)
    HL_X!(H, W) = Hnegloglikelihood!(H, W, W_prev_L, buf_L)

    if verbose > 0
        println("Maximizing likelihood...")
    end
    tic()
    res_L = optimize(L_X, dL_X!, HL_X!, W0, alg, options)
    t_L = toq()
    push!(res, res_L)
    println("Done! MLE $(Optim.converged(res_L) ? "did" : "did NOT") converge in $t_L s")
    show(res_L)
    println()
    # J_opt = reshape(Optim.minimizer(res_L), N_neurons, N_neurons)
    J_opt = v2lt(Optim.minimizer(res_L), N_neurons)
    P2 = IsingDistribution(J_opt;
        indices=I,
        autocomment="second_order_model[Optim/$(summary(alg))|MLE]",
        # MPF_converged=Optim.converged(res_K),
        MLE_converged=Optim.converged(res_L),
        # MPF_iterations=Optim.iterations(res_K),
        MLE_iterations=Optim.iterations(res_L),
        MLE_iteration_limit_reached=Optim.iteration_limit_reached(res_L),
        # J0_K=Optim.initial_state(res_K),
        # J0_L=Optim.initial_state(res_L),
        dkwargs...)
    # hide_metadata!(P2, :J0_L)
    # hide_metadata!(P2, :J0_K)

    return P2
end

"""
    data_model(X, I; kwargs...)

Returns a `DataDistribution` based on the specified data and indices.
"""
function data_model(X::Union{Matrix{Bool},BitMatrix}, I=1:size(X,1); kwargs...)
    return DataDistribution(X[I,:]; autocomment="data_model", indices=I, kwargs...)
end
