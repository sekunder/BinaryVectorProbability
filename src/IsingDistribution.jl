"""
    IsingDistribution

Represents the Ising distribution ``P(x) = 1/Z exp(1/2 x' J x - x' th)``
"""
mutable struct IsingDistribution <: AbstractBinaryVectorDistribution
    J::Matrix{Float64}
    theta::Vector{Float64}
    # I::Vector{Int} # moving to metadata
    metadata::Dict{Any,Any}
    cache::Dict{Any,Any}

    function IsingDistribution(J::Matrix{Float64}, theta::Vector{Float64}; kwargs...)
        if size(J,1) != size(J,2)
            error("Ising Distribution: J must be square")
        end
        if size(J,1) != length(theta)
            error("Ising Distribution: J and theta must have compatible size")
        end
        Jsym = (J + J') / 2
        Jsym = Jsym - Diagonal(Jsym)
        new(Jsym, theta, Dict(kwargs), Dict(:pdf=>spzeros(2^length(theta)), :energy=>spzeros(2^length(theta))))
    end
    function IsingDistribution(Jtilde::Matrix{Float64}; kwargs...)
        theta = diag(Jtilde)
        Jnodiag = Jtilde - Diagonal(Jtilde)
        IsingDistribution(Jnodiag, theta; kwargs...)
    end
end

function show(io::IO, P::IsingDistribution)
    println(io, "Ising Distribution")
    println(io, "N_neurons: $(n_bits(P))")
    # println(io, "Indices:   $(P.I)")
    show_metadata(io, P)
end

==(P1::IsingDistribution, P2::IsingDistribution) = (P1.J == P2.J) && (P1.theta == P2.theta)

################################################################################
#### Miscellaneous computations/internal functions
################################################################################
"""
    initialize_cache!(ID)

Checks if `ID.cache[:pdf]` and `ID.cache[:energy]` exist; if not, initializes those values to empty sparse vectors of length `2^n_bits(ID)`
"""
function initialize_cache!(ID)
    if !haskey(ID.cache, :pdf)
        ID.cache[:pdf] = spzeros(2^n_bits(ID))
    end
    if !haskey(ID.cache, :energy)
        ID.cache[:energy] = spzeros(2^n_bits(ID))
    end
end

"""
    _E_Ising(ID, x)

The Ising energy of state `x`, given by `x' θ - 1/2 x' J x`. Stores value in cache.
"""
function _E_Ising(ID::IsingDistribution, x::AbstractVector{Bool})
    idx = 1 + _binary_to_int(x)
    if ID.cache[:energy][idx] == 0.0
        ID.cache[:energy][idx] = dot(x, ID.theta - 0.5 * ID.J * x)
    end
    return ID.cache[:energy][idx]
end
"""
    _get_energies(ID)

Returns an array of the Ising energy of each state.
"""
_get_energies(ID::IsingDistribution) = [_E_Ising(ID, digits(Bool,x,2,n_bits(ID))) for x in 0:(2^n_bits(ID) - 1)]
function _get_Z(ID::IsingDistribution)
    if !haskey(ID.cache, :Z)
        if n_bits(ID) > ISING_METHOD_THRESHOLD
            warn("Computing Z for Ising PDF with $(n_bits(ID)) neurons. This might take a while, and numerical accuracy is not guaranteed. (Warning only displays once)", once=true, key=METH_THRESH_WARN_KEY)
        end
        # ID.cache[:Z] = sum_kbn([exp(-_E_Ising(ID, digits(Bool, k, 2, n_bits(ID)))) for k = 0:(2^n_bits(ID) - 1)])
        ID.cache[:Z] = sum_kbn(exp.(-_get_energies(ID)))
    end
    return ID.cache[:Z]
end

################################################################################
#### Implementation of methods for distributions
################################################################################
function entropy(P::IsingDistribution)
    if !haskey(P.metadata, :entropy)
        energies = _get_energies(P)
        # P.metadata[:entropy] = log(_get_Z(P)) + sum_kbn([exp(-_E_Ising(P,digits(Bool,2,x,n_bits(P)))) for x in 0:(2^n_bis(P) - 1)])
        P.metadata[:entropy] = log(_get_Z(P)) + sum_kbn(exp.(-energies) .* energies) / _get_Z(P)
    end
    return P.metadata[:entropy]
end
function entropy2(P::IsingDistribution)
    if !haskey(P.metadata, :entropy2)
        energies = _get_energies(P)
        P.metadata[:entropy2] = log2(_get_Z(P)) + log2(e) * sum_kbn(exp.(-energies) .* energies) / _get_Z(P)
    end
    return P.metadata[:entropy2]
end

n_bits(P::IsingDistribution) = length(P.theta)

function pdf(ID::IsingDistribution, x)
    if length(x) != n_bits(ID)
        error("IsingDistribution pdf: out of domain error")
    else
        idx = 1 + _binary_to_int(x)
        if ID.cache[:pdf][idx] == 0.0
            ID.cache[:pdf][idx] = exp(-_E_Ising(ID,x)) / _get_Z(ID)
        end
        return ID.cache[:pdf][idx]
    end
end


function random(ID::IsingDistribution, n_samples=1, force_gibbs=false)
    if n_bits(ID) <= ISING_METHOD_THRESHOLD && !force_gibbs
        return _random_exact(ID, n_samples)
    else
        return _random_gibbs(ID, n_samples)
    end
end
function _random_gibbs(ID::IsingDistribution, n_samples=1)
    burnin = _get_burnin(ID)
    ind_steps = _get_ind_steps(ID)
    N_steps = burnin + (n_samples - 1) * ind_steps

    d_i = rand(1:n_bits(ID), N_steps) # sequence of bits to update
    u_r = rand(N_steps) # sequence of uniform random samples from [0,1]
    X_out = falses(n_bits(ID), n_samples)
    x = rand(Bool, n_bits(ID)) # the current state

    sample = 1
    next_sample_step = burnin

    for step = 1:N_steps
        # compute the conditional probability of bit d_i[step] given the other bits
        E_act = dot(x, -0.5*ID.J[d_i[step],:]) + ID.theta[d_i[step]]
        P_act = 1 / (1 + exp(-E_act))
        x[d_i[step]] = P_act < u_r[step]
        if step == next_sample_step
            X_out[:,sample] = x
            sample += 1
            next_sample_step += ind_steps
        end
    end
    return X_out
end
_get_burnin(ID::IsingDistribution) = get!(ID.metadata, :burnin, 100*n_bits(ID))
_get_ind_steps(ID::IsingDistribution) = get!(ID.metadata, :ind_steps, 10*n_bits(ID))
