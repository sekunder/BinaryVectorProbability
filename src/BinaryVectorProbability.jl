"""
BinaryVectorProbability

Includes types and methods for representing probability distributions on the set of binary vectors.
"""
module BinaryVectorProbability

using JLD
using Optim

import Base: show, ==, isequal, hash
include("metadata.jl")

################################################################################
#### Abstract type to use as placeholder/to get all docstrings in one place
################################################################################
"""
    AbstractBinaryVectorDistribution

Abstract type used as placeholder.
"""
abstract type AbstractBinaryVectorDistribution end

################################################################################
#### Common functions
################################################################################

_NI(m) = error("Not implemented: $m")

# override this definition for specific types of distributions. This should represent
# equality in the sense of P(x) = Q(x) for all x; other attributes shouldn't matter. on the
# other hand, isequal is a more stringent check used for hashing.
==(P::AbstractBinaryVectorDistribution, Q::AbstractBinaryVectorDistribution) = false

"""
    n_bits(P::AbstractBinaryVectorDistribution)

Size of the sample space
"""
n_bits(P::AbstractBinaryVectorDistribution) = _NI("n_bits($(typeof(P)))")

"""
    _binary_to_int(x::Union{BitVector, Vector{Bool}})

Returns the integer that `x` represents. Will use `x.chunks` for `BitVector`s
and the "dot product method" for `Vector{Bool}`s. Note that this will really
mess you up if `length(x) > 63`

"""
_binary_to_int(x::BitVector) = Int(x.chunks[1])
_binary_to_int(x::AbstractVector{Bool}) = dot([2^i for i = 0:(length(x) - 1)], x)

"""
    pdf(P)

Returns the function `x -> pdf(P, x)`
"""
pdf(P::AbstractBinaryVectorDistribution) = x -> pdf(P, x)

"""
    get_pdf(P)

Returns the pdf of `P` as a `Vector{Float64}`, where `pdf[i]` is the probability
of `digits(Bool, i-1, 2, n_bits(P))`

"""
function get_pdf(P::AbstractBinaryVectorDistribution)
    if get(P.metadata, :pdf_computed, false)
        return full(P.cache[:pdf])
    else
        P.metadata[:pdf_computed] = true
        return [pdf(P, digits(Bool,x,2,n_bits(P))) for x in 0:(2^n_bits(P) - 1)]
    end
end

"""
    get_cdf(P)

Returns the "cdf" of `P` as a `Vector{Float64}`, meaning `cumsum(get_pdf(P))`.
(I mean, it's a "cdf" in the sense that you can put a total order {0,1}^n, but
really it's just used for sampling from the distribution.)

"""
function get_cdf(P::AbstractBinaryVectorDistribution)
    if haskey(P.cache, :cdf)
        cdf = P.cache[:cdf]
    else
        cdf = cumsum_kbn(get_pdf(P))
        P.cache[:cdf] = cdf
    end
    return cdf
end

"""
    random(P, n_sample)

Naive sampling method to draw random vectors from a distribution. Relies on an
implementation of `get_cdf` for the distribution (so, don't use with distributions on more
than ~20 neurons). Numerical accuracy not guaranteed. Returns a `BitMatrix`.

"""
function random(P::AbstractBinaryVectorDistribution, n_samples=1)
    X = falses(n_bits(P), n_samples)
    r = rand(n_samples)
    cdf = get_cdf(P)
    for s = 1:n_samples
        k = searchsortedfirst(cdf, r[s])
        X[:,s] = digits(Bool, k - 1, 2, n_bits(P))
    end
    return X
end

"""
    expectation_matrix(P)

Returns a matrix with the expected values of bits and pairs of bits. Default
behavior is to call `get_pdf(P)` and then loop through those values.

"""
function expectation_matrix(P::AbstractBinaryVectorDistribution)
    em = zeros(n_bits(P), n_bits(P))
    p = get_pdf(P)
    for k in sortperm(p)
        x = digits(Bool, k-1, 2, n_bits(P))
        em += p[k] * (x * x')
    end
    return em
end

"""
    entropy(P)

Entropy, in nats (i.e. natural log is used) of the given distribution. Sets a
metadata value.

"""
function entropy(P::AbstractBinaryVectorDistribution)
    if !haskey(P.metadata, :entropy)
        #MAYBEDO figure out how the get! function works. I assume since it's a
        #function call, it'll evalute all arguments first. But, if that's not
        #the case this could be cleaned up, I suppose.
        P.metadata[:entropy] = -sum_kbn([p * log(p) for p in get_pdf(P) if 0.0 < p < 1.0])
    end
    return P.metadata[:entropy]
end

"""
    entropy2(P)

Entropy, in bits (i.e. log base 2 is used) of the given distribution. Sets a
metadata value.

"""
function entropy2(P::AbstractBinaryVectorDistribution)
    if !haskey(P.metadata, :entropy2)
        P.metadata[:entropy2] = -sum_kbn([p * log2(p) for p in get_pdf(P) if 0.0 < p < 1.0])
    end
    return P.metadata[:entropy2]
end




################################################################################
#### Constants
################################################################################
"""
    ISING_METHOD_THRESHOLD

If N_neurons is greater than this number, switch from exact sampling to Gibbs
sampling and use MPF instead of LogLikelihood to for the parameters.
"""
const ISING_METHOD_THRESHOLD = 20
const METH_THRESH_WARN_KEY = 1

################################################################################
#### Include definitions of concrete distributions, other functions
################################################################################
include("DataDistribution.jl")
include("BernoulliCodeDistribution.jl")
include("IsingDistribution.jl")

include("fileio.jl")
include("objective_functions.jl")
include("optimizers.jl")
include("fitters.jl")

export AbstractBinaryVectorDistribution,
       show, ==, isequal, hash,
       n_bits, random, expectation_matrix, entropy, entropy2, pdf, get_pdf, get_cdf,
       savedistribution, loaddistribution,
       DataDistribution, BernoulliCodeDistribution, IsingDistribution,
       first_order_model, second_order_model, data_model,
       loglikelihood, MPF_objective,
       negloglikelihood, dnegloglikelihood!, K_MPF, dK_MPF!

end
