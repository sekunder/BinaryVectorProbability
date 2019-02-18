"""
    IsingDistribution

Represents the Ising distribution ``P(x) = 1/Z exp(x' J x)``, where ``J`` is a lower
triangular matrix.

"""
mutable struct IsingDistribution <: AbstractBinaryVectorDistribution
    # W # these are the n(n+1)/2 "true parameters."
    J::LowerTriangular # This is the lower triangular matrix used in calculations
    # I::Vector{Int} # moving to metadata
    metadata::Dict{Any,Any}
    cache::Dict{Any,Any}

    function IsingDistribution(J; kwargs...)
        if size(J,1) != size(J,2)
            error("IsingDistribution: J must be square")
        end
        Jsym = (J + J') / 2
        n = size(J,1)
        new(LowerTriangular(Jsym), Dict(kwargs), Dict(:pdf=>spzeros(2^n), :F=>spzeros(2^n)))
    end
    function IsingDistribution(W::Vector, kwargs...)
        IsingDistribution(v2lt(W), kwargs...)
    end
    # function IsingDistribution(J::Matrix, theta::Vector; kwargs...)
    #     if size(J,1) != size(J,2)
    #         error("Ising Distribution: J must be square")
    #     end
    #     if size(J,1) != length(theta)
    #         error("Ising Distribution: J and theta must have compatible size")
    #     end
    #     Jsym = (J + J') / 2
    #     Jsym = Jsym - Diagonal(Jsym)
    #     new(Jsym, theta, Dict(kwargs), Dict(:pdf=>spzeros(2^length(theta)), :energy=>spzeros(2^length(theta))))
    # end
    # function IsingDistribution(Jtilde::Matrix; kwargs...)
    #     theta = diag(Jtilde)
    #     Jnodiag = Jtilde - Diagonal(Jtilde)
    #     IsingDistribution(Jnodiag, theta; kwargs...)
    # end
    # function
end

n_bits(P::IsingDistribution) = size(P.J,1)

function show(io::IO, P::IsingDistribution)
    println(io, "Ising Distribution")
    println(io, "N_neurons: $(n_bits(P))")
    # println(io, "Indices:   $(P.I)")
    show_metadata(io, P)
end

# ==(P1::IsingDistribution, P2::IsingDistribution) = (P1.J == P2.J) && (P1.theta == P2.theta)
==(P::IsingDistribution, Q::IsingDistribution) = P.J == Q.J

################################################################################
#### Miscellaneous computations/internal functions
################################################################################
"""
    initialize_cache!(ID)

Checks if `ID.cache[:pdf]` and `ID.cache[:F]` exist; if not, initializes those values to empty sparse vectors of length `2^n_bits(ID)`
"""
function initialize_cache!(ID)
    if !haskey(ID.cache, :pdf)
        ID.cache[:pdf] = spzeros(2^n_bits(ID))
    end
    if !haskey(ID.cache, :F)
        ID.cache[:F] = spzeros(2^n_bits(ID))
    end
end

"""
    F_Ising(ID, x)

The "pseudo-energy" of state `x`, given by `x' J x`. Stores value in cache.
"""
function F_Ising(ID::IsingDistribution, x::AbstractVector{Bool})
    idx = 1 + _binary_to_int(x)
    if ID.cache[:F][idx] == 0.0
        ID.cache[:F][idx] = dot(x, ID.J * x)
    end
    return ID.cache[:F][idx]
end
# """
#     _get_energies(ID)
#
# Returns an array of the Ising energy of each state.
# """
# _get_energies(ID::IsingDistribution) = [_E_Ising(ID, digits(Bool,x,2,n_bits(ID))) for x in 0:(2^n_bits(ID) - 1)]
get_F(ID::IsingDistribution) = [F_Ising(ID, digits(Bool,x,2,n_bits(ID))) for x in 0:(2^n_bits(ID) - 1)]
function get_Z(ID::IsingDistribution)
    if !haskey(ID.cache, :Z)
        if n_bits(ID) > ISING_METHOD_THRESHOLD
            warn("Computing Z for Ising PDF with $(n_bits(ID)) neurons. This might take a while, and numerical accuracy is not guaranteed. (Warning only displays once)", once=true, key=METH_THRESH_WARN_KEY)
        end
        # ID.cache[:Z] = sum_kbn([exp(-_E_Ising(ID, digits(Bool, k, 2, n_bits(ID)))) for k = 0:(2^n_bits(ID) - 1)])
        ID.cache[:Z] = sum_kbn(exp.(get_F(ID)))
    end
    return ID.cache[:Z]
end

################################################################################
#### Implementation of methods for distributions
################################################################################
function entropy(P::IsingDistribution)
    if !haskey(P.metadata, :entropy)
        energies = get_F(P)
        # P.metadata[:entropy] = log(get_Z(P)) + sum_kbn([exp(-_E_Ising(P,digits(Bool,2,x,n_bits(P)))) for x in 0:(2^n_bis(P) - 1)])
        P.metadata[:entropy] = log(get_Z(P)) - sum_kbn(exp.(energies) .* energies) / get_Z(P)
    end
    return P.metadata[:entropy]
end
function entropy2(P::IsingDistribution)
    if !haskey(P.metadata, :entropy2)
        energies = get_F(P)
        P.metadata[:entropy2] = log2(get_Z(P)) - log2(e) * sum_kbn(exp.(energies) .* energies) / get_Z(P)
    end
    return P.metadata[:entropy2]
end

function pdf(ID::IsingDistribution, x)
    if length(x) != n_bits(ID)
        error("IsingDistribution pdf: out of domain error")
    else
        idx = 1 + _binary_to_int(x)
        if ID.cache[:pdf][idx] == 0.0
            ID.cache[:pdf][idx] = exp(F_Ising(ID,x)) / get_Z(ID)
        end
        return ID.cache[:pdf][idx]
    end
end


# function random(ID::IsingDistribution, n_samples=1, force_gibbs=false)
#     if n_bits(ID) <= ISING_METHOD_THRESHOLD && !force_gibbs
#         return _random_exact(ID, n_samples)
#     else
#         return _random_gibbs(ID, n_samples)
#     end
# end
function random(ID::IsingDistribution, n_samples=1)
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
        # E_act = dot(x, -0.5*ID.J[d_i[step],:]) + ID.theta[d_i[step]]
        E_act = F_Ising(ID, x)
        P_act = 1 / (1 + exp(E_act))
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

"""
    lt2v(V, n)

Return the lower triangular matrix `L` such that `L[:]`, skipping the entries above the
diagonal, is `V`.
"""
function v2lt(V, n=Int((-1 + sqrt(1 + 8*length(V)))/2))
    v = start(V)
    L = zeros(n,n)
    for i = 1:n
        for j = i:n
            L[j,i], v = next(V, v)
        end
    end
    return LowerTriangular(L)
end

lt2v(L) = vcat([[L[j,i] for j = i:size(L,1)] for i = 1:size(L,2)]...)
