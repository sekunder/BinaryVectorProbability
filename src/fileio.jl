
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
    savedistribution(P::AbstractBinaryVectorDistribution; [filename], [dir], [old])

Save the given distribution to disk using the `JLD` package. Returns the full path to the
saved file. Deletes file `old` if it exists in `dir`. By default `old` is `nothing` to avoid
unnecessarily deleting and rewriting files.

Default `filename` is `string(hash(P))`. Default `dir` is
`Pkg.dir(BinaryVectorProbability)/saved`.

If `dir` doesn't exist, will use `mkpath`. If file exists, contents will be overwritten.
"""
function savedistribution(P::AbstractBinaryVectorDistribution;
    filename=string(hash(P)),
    dir=joinpath(Pkg.dir("BinaryVectorProbability"), "saved"),
    old=nothing)

    # _dir = abspath(dir)
    if !ispath(dir)
        mkpath(dir)
    end
    if old != nothing
        _old = endswith(old, ".jld") ? old : old * ".jld"
        if isfile(joinpath(dir, _old))
            rm(joinpath(dir, _old))
        end
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
    loaddistribution(h::UInt; [dir])

Returns the distribution saved in `filename` (or identified by hash `h`) as saved by `savedistribution`. Default `dir`
is `Pkg.dir(BinaryVectorProbability)/saved`
"""
function loaddistribution(filename; dir=joinpath(Pkg.dir("BinaryVectorProbability"), "saved"))
    _fn = endswith(filename, ".jld") ? filename : filename * ".jld"
    _fn = basename(_fn)
    return load(joinpath(dir, _fn), "P")
end
loaddistribution(h::UInt; dir=joinpath(Pkg.dir("BinaryVectorProbability"),"saved")) = loaddistribution(string(h), dir=dir)
