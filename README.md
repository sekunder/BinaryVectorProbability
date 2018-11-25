# BinaryVectorProbability.jl

[![Build Status](https://travis-ci.org/sekunder/BinaryVectorProbability.jl.svg?branch=master)](https://travis-ci.org/sekunder/BinaryVectorProbability.jl)

[![Coverage Status](https://coveralls.io/repos/sekunder/BinaryVectorProbability.jl/badge.svg?branch=master&service=github)](https://coveralls.io/github/sekunder/BinaryVectorProbability.jl?branch=master)

[![codecov.io](http://codecov.io/github/sekunder/BinaryVectorProbability.jl/coverage.svg?branch=master)](http://codecov.io/github/sekunder/BinaryVectorProbability.jl?branch=master)

Types and functions for manipulating probability distributions on `{0,1}^n`, the set of binary vectors of length `n`. Also provides tools for fitting parameters.

## Installation and Usage

Currently only works with [Julia v0.6](https://julialang.org/downloads/oldreleases.html).

Installation via `Pkg.clone`:

`julia> Pkg.clone("https://github.com/sekunder/BinaryVectorProbability.jl.git")`

`julia> using BinaryVectorProbability`

## Types

The parent abstract type is `AbstractBinaryVectorDistribution`, all distributions are subtypes of this type.
 * `DataDistribution`: An arbitrary distribution. "Data" refers to the most common use for this type, which is to represent normalized counts of vectors occurring in a data sample.
 * `BernoulliCodeDistribution`: Probability distribution where each bit is an independent Bernoulli trial.
 * `IsingDistribution`: Probability distribution given by `P(x) = 1/Z exp(1/2 x' J x - x' h)`

To implement your own type `D`, at a minimum you need to define `pdf(D,x)` and `n_bits(D)`. See below for more details.

## Functions

This package provides the following operations on distributions:

 * `pdf(D<:AbstractBinaryVectorDistribution,x::Union{Vector{Bool},BitVector})`: The probability of vector `x`
 * `n_bits(D)`: `n`, where `D` is a distribution on `{0,1}^n`
 * `get_pdf(D)`: a vector of probabilities; `P[i]` is the probability of the vector `digits(Bool,i-1,2,n_bits(D))`
 * `get_cdf(D)`: the `cumsum` of `get_pdf(D)`, used for naive random sampling.
 * `expectation_matrix(D)`: An `n x n` matrix `M` where `M[i,j]` is the expected value of the product of the `i`th and `j`th bits. Diagonal entries are expected values of individual bits.
 * `entropy(D)`,`entropy2(D)`: The Shannon entropy of `D`, using natural log or log base 2, respectively.
 * `random(D, n_samples=1)`: Returns vectors drawn from this distribution.

## Fitting distributions

Given a binary matrix `X` representing sampled data, this package provides functions to find parameters for the best-fit Bernoulli model and best-fit Ising model (the maximum-entropy distributions matching, respectively, expected values of bits and expected values of pairs of bits).

 * `first_order_model(X::Union{Matrix{Bool},BitMatrix}, I=1:size(X,1))`: Treats columns of `X` as samples. Uses only rows indexed by `I`
 * `second_order_model(X,I)`: Fits the Ising model, using either loglikelihood or the minimum probability flow objective function.
 * `data_model`: computes a histogram of the data.

## Objective functions

This package uses the [`Optim`](https://github.com/JuliaNLSolvers/Optim.jl/) package to fit parameters for the Ising model, using either log likelihood or minimum probability flow.

## Defining New Distributions

To define a new type of distribution `NewDistribution`, define the type as follows:
```
mutable struct NewDistribution <: AbstractBinaryVectorDistribution
  ...
end
```

And define the following two functions:

`n_bits(D::NewDistribution)` the size of the sample space.

`pdf(D::NewDistribution,x)` the probability of observing `x` under this distribution

Overriding the other functions is, of course, also possible.
