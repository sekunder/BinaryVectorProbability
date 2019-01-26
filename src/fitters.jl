

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
        return _Naive_second_order_model(X, I; verbose=verbose, kwargs...)
    elseif typeof(algorithm) == Symbol
        return _NLopt_second_order_model(X, I; verbose=verbose, algorithm=algorithm, kwargs...)
    else
        return _Optim_second_order_model(X, I; verbose=verbose, algorithm=algorithm, kwargs...)
    end
end
function _Naive_second_order_model(X, I=1:size(X,1); verbose=0, kwargs...)
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
function _NLopt_second_order_model(X::Union{Matrix{Bool},BitMatrix}, I=1:size(X,1); verbose=0, kwargs...)
    dkwargs = Dict(kwargs)
    Xselected = X[I,:]
    N_neurons,N_samples = size(Xselected)

    Jseed = rand(N_neurons,N_neurons); Jseed = (Jseed + Jseed') / (2 * N_neurons)
    Jseed = pop!(dkwargs, :J0, Jseed)
    # for some reaosn I'm getting "F_X not defined". So, I'm going to try
    # definind F_X as loglikelihood no matter what, then override F_X with MPF
    # if necessary. I'll set a flag for which function I used, and then set
    # max_objective or min_objective as necessary.
    # Update: turns out maybe Julia was setting F_X no matter whether it
    # actually entered the branch? Or something.
    mu = Xselected * Xselected' / N_samples
    L_X(J,g) = loglikelihood(Xselected, J, g; mu_X=mu)
    K_X(J,g) = MPF_objective(Xselected, J, g)
    fun = (N_neurons > ISING_METHOD_THRESHOLD || pop!(dkwargs, :force_MPF, false)) ? "MPF" : "loglikelihood"
    # fun = "loglikelihood"
    # if N_neurons > ISING_METHOD_THRESHOLD || pop!(dkwargs, :force_MPF, false)
    #     K_X(J,g) = MPF_objective(Xselected, J, g)
    #     fun = "MPF"
    # end

    # let's try only writing one method, since the only difference is the
    # function and max/min.
    alg = pop!(dkwargs,:algorithm,:LD_LBFGS)
    opt_Ising = Opt(alg, N_neurons^2)

    if fun == "loglikelihood"
        if verbose > 0
            println("second_order_model[NLopt/$alg]: setting max objective function $fun")
            println("\tApproximate cost: 2^$N_neurons + $N_samples per evalution")
        end
        max_objective!(opt_Ising, L_X)
    else
        if verbose > 0
            println("second_order_model[NLopt/$alg]: setting min objective function $fun")
            println("\tApproximate cost: $N_neurons * $N_samples per evaluation")
        end
        min_objective!(opt_Ising, K_X)
    end
    if haskey(dkwargs, :ftol_rel)
        ftol_rel!(opt_Ising, pop!(dkwargs,:ftol_rel))
    end
    if haskey(dkwargs, :ftol_abs)
        ftol_abs!(opt_Ising, pop!(dkwargs,:ftol_abs))
    end
    if haskey(dkwargs, :xtol_rel)
        xtol_rel!(opt_Ising, pop!(dkwargs,:xtol_rel))
    end
    if haskey(dkwargs, :xtol_abs)
        xtol_abs!(opt_Ising, pop!(dkwargs,:xtol_abs))
    end
    if haskey(dkwargs, :vector_storage)
        vector_storage!(opt_Ising, pop!(dkwargs,:vector_storage))
    end
    if haskey(dkwargs, :maxeval)
        maxeval!(opt_Ising, pop!(dkwargs,:maxeval))
    end
    if haskey(dkwargs, :maxtime)
        maxtime!(opt_Ising, pop!(dkwargs,:maxtime))
    end
    if verbose > 0
        println("second_order_model[NLopt/$alg]: running optimization")
        # println("\topt object: $(opt_Ising)") # turns out this just prints "Opt(:algorithm, N_vars)"
        println("\talgorithm: $(algorithm(opt_Ising))")
        println("\tftol (rel/abs): $(ftol_rel(opt_Ising)) / $(ftol_abs(opt_Ising))")
        println("\txtol (rel/|abs|): $(xtol_rel(opt_Ising)) / $(norm(xtol_abs(opt_Ising)))")
        println("\tvect. stor.: $(vector_storage(opt_Ising))")
        println("\tmax evals: $(maxeval(opt_Ising))")
    end
    (optVal, J_opt, optReturn) = optimize(opt_Ising, Jseed[:])
    # final_val = F_X(J_opt, [])
    J_opt = reshape(J_opt, N_neurons, N_neurons)
    return IsingDistribution(J_opt; indices=I, autocomment="second_order_model[NLopt/$alg|$fun]", opt_val=optVal, opt_ret=optReturn, dkwargs...)
end
function _Optim_second_order_model(X, I=1:size(X,1); verbose=0, kwargs...)
    dkwargs = Dict(kwargs)
    _X = sortcols(X[I,:]) # sorting columns for branch prediction and memory access
    N_neurons,N_samples = size(_X)

    J0 = rand(N_neurons,N_neurons)
    J0 = (J0 + J0')/2
    J0 = pop!(dkwargs, :J0, J0)

    J_prev_L = similar(J0)
    mu_X = _X * _X' / N_samples
    buf_L = [IsingDistribution(J0), _X, mu_X]
    L_X(J) = negloglikelihood(J, J_prev_L, buf_L)
    dL_X!(G, J) = dnegloglikelihood!(G, J, J_prev_L, buf_L)
    # L_X(J) = negloglikelihood(_X, J)
    # dL_X!(G, J) = dnegloglikelihood!(_X, G, J; mu_X=mu_X)

    J_prev_K = similar(J0)
    J0_K = similar(J0)
    J0_K[:] = J0[:]
    J0_K[1:(N_neurons+1):end] = 0.0
    buf_K = [_X, 2 * _X - 1, similar(_X, Float64), diag(J0), J0_K]
    K_X(J) = K_MPF(J, J_prev_K, buf_K)
    dK_X!(G, J) = dK_MPF!(G, J, J_prev_K, buf_K)
    # K_X(J) = K_MPF(_X, J)
    # dK_X!(G, J) = dK_MPF!(_X, G, J)

    # fun = pop!(dkwargs, :fun, "-loglikelihood")

    alg = pop!(dkwargs, :algorithm, LBFGS())

    # Configure various options for Optim.jl
    # show_trace = pop!(dkwargs, :show_trace, verbose >= 2)
    options = Optim.Options(
        show_trace = pop!(dkwargs, :show_trace, verbose >= 2),
        x_tol = pop!(dkwargs, :x_tol, 0.0),
        f_tol = pop!(dkwargs, :f_tol, 0.0),
        allow_f_increases = pop!(dkwargs, :allow_f_increases, false),
        iterations = pop!(dkwargs, :iterations, 500),
        show_every = pop!(dkwargs, :show_every, 10)
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

    # run the optimization using Optim.jl
    # res = fun == "MPF" ? optimize(K_X, dK_X!, J0, alg, options) : optimize(L_X, dL_X!, J0, alg, options)
    # J_opt = Optim.minimizer(res)
    # J_opt = reshape(J_opt, N_neurons, N_neurons)

    # TODO New plan: try using L then K, or K then L, or alternating. K is actually convex,
    # so it'll always converge, but there's no guarantee it'll be "correct". So K, L or L,
    # K, L should hopefully get us away from local minima and get us what we're after.

    println("Using Minimum Probability Flow to estimate parameters...")
    tic()
    res_K = optimize(K_X, dK_X!, J0, alg, options)
    t_K = toq()
    J_K = Optim.minimizer(res_K)
    println("Done! MPF $(Optim.converged(res_K) ? "did" : "did not") converge in $t_K s")
    show(res_K)
    println()

    println("Maximizing likelihood...")
    tic()
    res_L = optimize(L_X, dL_X!, J_K, alg, options)
    t_L = toq()
    println("Done! MLE $(Optim.converged(res_L) ? "did" : "did not") converge in $t_L s")
    show(res_L)
    println()
    J_opt = reshape(Optim.minimizer(res_L), N_neurons, N_neurons)
    P2 = IsingDistribution(J_opt;
        indices=I,
        autocomment="second_order_model[Optim/$(summary(alg))|MPF->MLE]",
        MPF_converged=Optim.converged(res_K),
        MLE_converged=Optim.converged(res_L),
        MPF_iterations=Optim.iterations(res_K),
        MLE_iterations=Optim.iterations(res_L),
        MLE_iterations_limit_reached=Optim.iteration_limit_reached(res_L),
        J0_K=Optim.initial_state(res_K),
        J0_L=Optim.initial_state(res_L),
        dkwargs...)
    hide_metadata!(P2, :J0_L)
    hide_metadata!(P2, :J0_K)

    return P2
end

"""
    data_model(X, I; kwargs...)

Returns a `DataDistribution` based on the specified data and indices.
"""
function data_model(X::Union{Matrix{Bool},BitMatrix}, I=1:size(X,1); kwargs...)
    return DataDistribution(X[I,:]; autocomment="data_model", indices=I, kwargs...)
end
