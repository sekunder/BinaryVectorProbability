

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

Returns an `IsingDistribution` which is fit to the pairwise correlations in `X`.

Some useful keywords particular to this function:
 * `J0` is the initial value used for optimization. Useful for debugging.

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
    _X = X[I,:]
    N_neurons,N_samples = size(_X)

    J0 = rand(N_neurons,N_neurons)
    J0 = (J0 + J0')/2
    J0 = pop!(dkwargs, :J0, J0)


    mu_X = _X * _X' / N_samples
    L_X(J) = negloglikelihood(_X, J)
    dL_X!(G, J) = dnegloglikelihood!(_X, G, J; mu_X=mu_X)

    K_X(J) = K_MPF(_X, J)
    dK_X!(G, J) = dK_MPF!(_X, G, J)
    fun = pop!(dkwargs, :fun, "-loglikelihood")

    alg = pop!(dkwargs, :algorithm, LBFGS())

    # Configure various options for Optim.jl
    # show_trace = pop!(dkwargs, :show_trace, verbose >= 2)
    options = Optim.Options(
        show_trace = pop!(dkwargs, :show_trace, verbose >= 2),
        f_tol = pop!(dkwargs, :f_tol, 0.0),
        allow_f_increases = pop!(dkwargs, :allow_f_increases, false),
        iterations = pop!(dkwargs, :iterations, 500)
        )

    if verbose > 0
        println("second_order_model[Optim/$(summary(alg))]: setting objective function $fun")
    end

    if verbose > 0
        println("second_order_model[Optim/$(summary(alg))]: running optimization")
        println("  algorithm: $(summary(alg))")
        #TODO potentially other information should be printed
    end

    # run the optimization using Optim.jl
    res = fun == "MPF" ? optimize(K_X, dK_X!, J0, alg, options) : optimize(L_X, dL_X!, J0, alg, options)
    J_opt = Optim.minimizer(res) #TODO I can actually optimize over matrices, without need to reshape
    J_opt = reshape(J_opt, N_neurons, N_neurons)
    # P2 = IsingDistribution(J_opt; indices=I, autocomment="second_order_model[Optim/$(summary(alg))|$fun]", opt_res=res, minimizer_converged=Optim.converged(res), dkwargs...)
    # FUN FACT JLD SEEMS TO BUG OUT WHEN YOU TRY TO SAVE THE RESULTS OBJECT (i.e. res). SO GLAD I WASTED ALL THAT FUCKING COMPUTATION TIME WHEEEEEE
    P2 = IsingDistribution(J_opt;
        indices=I,
        autocomment="second_order_model[Optim/$(summary(alg))|$fun]",
        minimizer_converged=Optim.converged(res),
        iterations=Optim.iterations(res),
        iterations_limit_reached=iteration_limit_reached(res),
        dkwargs...)
    # hide_metadata!(P2, :opt_res)
    return P2
end

"""
    data_model(X, I; kwargs...)

Returns a `DataDistribution` based on the specified data and indices.
"""
function data_model(X::Union{Matrix{Bool},BitMatrix}, I=1:size(X,1); kwargs...)
    return DataDistribution(X[I,:]; autocomment="data_model", indices=I, kwargs...)
end
