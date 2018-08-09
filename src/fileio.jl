# Methods for saving distributions to disk

# more stringent check than ==
function isequal(P::AbstractBinaryVectorDistribution, Q::AbstractBinaryVectorDistribution)
    typeof(P) == typeof(Q) || return false
    for (fp,fq) in zip(fieldnames(typeof(P)),fieldnames(typeof(Q)))
        isequal(getfield(P, fp), getfield(Q, fq)) || return false
    end
    return true
end

function hash(P::AbstractBinaryVectorDistribution, h::UInt)
    _h = h
    for fn in fieldnames(P)
        _h = hash(getfield(P, fn), _h)
    end
    return _h
end



################################################################################
#### File IO
################################################################################
"""
    savedistribution(P::AbstractBinaryVectorDistribution; [filename], [dir])

Save the given distribution to disk using the `JLD` package. Returns the full path to the
saved file.

Default `filename` is `(typeof(P)) * "." * (hash(P)) * ".jld"`. Default `dir` is
`Pkg.dir(BinaryVectorProbability)/saved`.

If `dir` doesn't exist, will use `mkpath`. If file exists, contents will be overwritten.
"""
function savedistribution(P::AbstractBinaryVectorDistribution;
    filename="$(typeof(P)).$(hash(P))",
    dir=joinpath(Pkg.dir("BinaryVectorProbability"), "saved"))

    # _dir = abspath(dir)
    if !ispath(dir)
        mkpath(dir)
    end
    _fn = endswith(filename, ".jld") ? filename : filename * ".jld"
    # save(joinpath(_dir, _fn), "P", P)
    jldopen(joinpath(dir, _fn), "w") do file
        addrequire(file, BinaryVectorProbability)
        write(file, "P", P)
    end
    return joinpath(dir, _fn)
end

"""
    loaddistribution(filename; [dir])

Returns the distribution saved in `filename` as saved by `savedistribution`. Default `dir`
is `Pkg.dir(BinaryVectorProbability)/saved`
"""
function loaddistribution(filename; dir=joinpath(Pkg.dir("BinaryVectorProbability"), "saved"))
    _fn = endswith(filename, ".jld") ? filename : filename * ".jld"
    _fn = basename(_fn)
    return load(joinpath(dir, _fn), "P")
end
