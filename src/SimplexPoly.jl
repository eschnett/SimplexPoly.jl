module SimplexPoly

using ComputedFieldTypes
using DifferentialForms
using StaticArrays

################################################################################

export Term
struct Term{D,T}
    powers::SVector{D,Int}
    coeff::T
    function Term{D,T}(powers::SVector{D,Int}, coeff::T) where {D,T}
        D::Int
        @assert D >= 0
        @assert all(>=(0), powers)
        return new{D,T}(powers, coeff)
    end
end

function Base.map(f, x::Term{D,T}, ys::Term{D,T}...) where {D,T}
    @assert all(x.powers == y.powers for y in ys)
    return Term{D,T}(x.powers, map(f, x.coeff, (y.coeff for y in ys)...))
end

function Base.zero(::Type{Term{D,T}}) where {D,T}
    return Term{D,T}(zero(SVector{D,Int}), zero(T))
end
Base.zero(::Term{D,T}) where {D,T} = zero(Term{D,T})
Base.iszero(x::Term{D,T}) where {D,T} = iszero(x.coeff)

function Base.one(::Type{Term{D,T}}) where {D,T}
    return Term{D,T}(zero(SVector{D,Int}), one(T))
end
Base.one(::Term{D,T}) where {D,T} = one(Term{D,T})
Base.isone(x::Term{D,T}) where {D,T} = iszero(x.powers) && isone(x.coeff)

Base.:+(x::Term{D,T}) where {D,T} = map(+, x)
Base.:-(x::Term{D,T}) where {D,T} = map(-, x)

Base.:+(x::Term{D,T}, y::Term{D,T}) where {D,T} = map(+, x, y)
Base.:-(x::Term{D,T}, y::Term{D,T}) where {D,T} = map(-, x, y)
Base.:*(a, x::Term{D,T}) where {D,T} = map(c -> a * c, x)
Base.:*(x::Term{D,T}, a) where {D,T} = map(c -> c * a, x)
Base.:\(a, x::Term{D,T}) where {D,T} = map(c -> a \ c, x)
Base.:/(x::Term{D,T}, a) where {D,T} = map(c -> c / a, x)

function Base.:*(x::Term{D,T}, y::Term{D,T}) where {D,T}
    return Term{D,T}(x.powers + y.powers, x.coeff * y.coeff)
end
function Base.:^(x::Term{D,T}, n::Integer) where {D,T}
    @assert n >= 0
    r = one(x)
    while n > 0
        if n % 2 != 0
            r *= x
        end
        x *= x
        n ÷= 2
    end
    return r
end

export deriv
function deriv(term::Term{D,T}, dir::Int) where {D,T}
    D::Int
    @assert D >= 0
    @assert 1 <= dir <= D
    term.powers[dir] == 0 && return zero(Term{D,T})
    p = term.powers[dir]
    return Term{D,T}(Base.setindex(term.powers, p - 1, dir), p * term.coeff)
end

"""
    koszul

The Koszul operator `κ` (kappa)

See e.g.: Douglas Arnold, Richard Falk, Ragnar Winther, "Finite
element exterior calculus, homological techniques, and applications",
Acta Numerica 15, 1-155 (2006), DOI:10.1017/S0962492906210018.
"""
function koszul end

export koszul
function koszul(term::Term{D,T}, dir::Int) where {D,T}
    D::Int
    @assert D >= 0
    @assert 1 <= dir <= D
    p = term.powers[dir]
    return Term{D,T}(Base.setindex(term.powers, p + 1, dir), term.coeff)
end

################################################################################

export Poly
struct Poly{D,T} <: Number
    terms::Vector{Term{D,T}}
    function Poly{D,T}(terms::Vector{Term{D,T}}) where {D,T}
        D::Int
        @assert D >= 0
        terms = combine(terms)
        return new{D,T}(terms)
    end
end

function combine(terms::Vector{Term{D,T}}) where {D,T}
    terms = sort(terms; by=(t -> t.powers))
    isempty(terms) && return terms
    newterms = Term{D,T}[]
    i = 0
    term = terms[i += 1]
    while i < length(terms)
        term2 = terms[i += 1]
        if term2.powers == term.powers
            term += term2
        else
            !iszero(term) && push!(newterms, term)
            term = term2
        end
    end
    !iszero(term) && push!(newterms, term)
    return newterms
end

Base.:(==)(p::Poly{D,T}, q::Poly{D,T}) where {D,T} = p.terms == q.terms

Base.map(f, p::Poly{D,T}) where {D,T} = Poly{D,T}(map(t -> map(f, t), p.terms))

function Base.map(f, p::Poly{D,T}, q::Poly{D,T}) where {D,T}
    terms = Term{D,T}[]
    i = j = 1
    ni = length(p.terms)
    nj = length(q.terms)
    while i <= ni || j <= nj
        usei = usej = false
        if i <= ni && j <= nj
            c = cmp(p.terms[i].powers, q.terms[j].powers)
            usei = c <= 0
            usej = c >= 0
        else
            usei = i <= ni
            usej = !usei
        end
        pi = usei ? p.terms[i] : map(zero, q.terms[j])
        qj = usej ? q.terms[j] : map(zero, p.terms[i])
        push!(terms, map(f, pi, qj))
        i += usei
        j += usej
    end
    return Poly{D,T}(terms)
end

function Base.zero(::Type{Poly{D,T}}) where {D,T}
    D::Int
    @assert D >= 0
    return Poly{D,T}(Vector{Term{D,T}}())
end
Base.zero(::Poly{D,T}) where {D,T} = zero(Poly{D,T})
Base.iszero(x::Poly{D,T}) where {D,T} = isempty(x.terms)

function Base.one(::Type{Poly{D,T}}) where {D,T}
    return Poly{D,T}([one(Term{D,T})])
end
Base.one(::Poly{D,T}) where {D,T} = one(Poly{D,T})
Base.isone(x::Poly{D,T}) where {D,T} = length(x.terms) == 1 && isone(x.terms[1])

export unit
function DifferentialForms.unit(::Type{Poly{D,T}}, dir::Int,
                                coeff=one(T)) where {D,T}
    D::Int
    @assert D >= 0
    @assert 1 <= dir <= D
    term = Term{D,T}(SVector{D,T}(d == dir for d in 1:D), coeff)
    return Poly{D,T}([term])
end

Base.:+(p::Poly{D,T}) where {D,T} = map(+, p)
Base.:-(p::Poly{D,T}) where {D,T} = map(-, p)

Base.:+(p::Poly{D,T}, q::Poly{D,T}) where {D,T} = map(+, p, q)
Base.:-(p::Poly{D,T}, q::Poly{D,T}) where {D,T} = map(-, p, q)
Base.:*(a::Number, p::Poly{D,T}) where {D,T} = map(t -> a * t, p)
Base.:*(p::Poly{D,T}, a::Number) where {D,T} = map(t -> t * a, p)
Base.:/(a::Number, p::Poly{D,T}) where {D,T} = map(t -> a \ t, p)
Base.:\(p::Poly{D,T}, a::Number) where {D,T} = map(t -> t / a, p)

function Base.:*(p::Poly{D,T}, q::Poly{D,T}) where {D,T}
    return Poly{D,T}(Term{D,T}[t * u for t in p.terms for u in q.terms])
end
function Base.:^(x::Poly{D,T}, n::Integer) where {D,T}
    @assert n >= 0
    r = one(x)
    while n > 0
        if n % 2 != 0
            r *= x
        end
        x *= x
        n ÷= 2
    end
    return r
end

function deriv(poly::Poly{D,T}, dir::Int) where {D,T}
    return Poly{D,T}(map(t -> deriv(t, dir), poly.terms))
end

function koszul(poly::Poly{D,T}, dir::Int) where {D,T}
    return Poly{D,T}(map(t -> koszul(t, dir), poly.terms))
end

################################################################################

function deriv(f::Form{D,R,Poly{D,T}}) where {D,R,T}
    D::Int
    R::Int
    # deriv(::Form{D,D}) == 0
    @assert 0 <= R < D
    r = zero(Form{D,R + 1,Poly{D,T}})
    N = length(f)
    for n in 1:N
        bits = Forms.lin2bit(Val(D), Val(R), n)
        for dir in 1:D
            if !bits[dir]
                parity = false
                for d in (dir + 1):N
                    parity ⊻= bits[d]
                end
                s = bitsign(parity)
                rbits = Base.setindex(bits, true, dir)
                rn = Forms.bit2lin(Val(D), Val(R + 1), rbits)
                r = Base.setindex(r, r[rn] + s * deriv(f[n], dir), rn)
            end
        end
    end
    return r::Form{D,R + 1,Poly{D,T}}
end

function koszul(f::Form{D,R,Poly{D,T}}) where {D,R,T}
    D::Int
    R::Int
    # koszul(::Form{D,0}) == 0
    @assert 0 < R <= D
    r = zero(Form{D,R - 1,Poly{D,T}})
    N = length(f)
    for n in 1:N
        bits = Forms.lin2bit(Val(D), Val(R), n)
        for dir in 1:D
            if !bits[dir]
                parity = false
                for d in 1:(dir - 1)
                    parity ⊻= bits[d]
                end
                s = bitsign(parity)
                rbits = Base.setindex(bits, true, dir)
                rn = Forms.bit2lin(Val(D), Val(R - 1), rbits)
                r = Base.setindex(r, r[rn] + s * deriv(f[n], dir), rn)
            end
        end
    end
    return r::Form{D,R - 1,Poly{D,T}}
end

# @inline loop(f, ::Val{N}) where {N} = map(f, ntuple(n -> Val(n), N))

# function deriv(f::Form{D,0,Poly{D,T}}) where {D,T}
#     return Form{D,1}(ntuple(d -> deriv(f[], d), D))
# end

# function deriv(f::Form{2,1,Poly{2,T}}) where {T}
#     return Form{2,2}((deriv(f[1], 2) - deriv(f[2], 1),))
# end
# function deriv(f::Form{3,1,Poly{3,T}}) where {T}
#     r = zero(Form{3,2,Poly{3,T}})
#     r = Base.setindex(r, deriv(f[1], 2) - deriv(f[2], 1), 1, 2)
#     r = Base.setindex(r, deriv(f[3], 1) - deriv(f[1], 3), 1, 3)
#     r = Base.setindex(r, deriv(f[2], 3) - deriv(f[3], 2), 2, 3)
#     return r
# end

################################################################################

@computed struct Space{D,R,T}
    polys::Vector{fulltype(Form{D,R,Poly{D,T}})}
end

Base.zero(::Type{<:Space{D,R,T}}) where {D,R,T} = Space{D,R,T}([])

function basis(::Type{<:Space{D,R,T}}, p::Int) where {D,R,T}
    D::Int
    @assert 0 <= R <= D
    @assert p >= 0

    polys = Poly{D,T}[]
    for i0 in
        CartesianIndex(ntuple(d -> 0, D)):CartesianIndex(ntuple(d -> p, D))
        i = SVector{D,Int}(i.I)
        if sum(i) <= p
            push!(polys, Poly{D,T}([Term{D,T}(i, one(T))]))
        end
    end

    N = length(Form{D,R})
    forms = fulltype(Form{D,R,Poly{D,T}})[zero(Form{D,R,Poly{D,T}})]
    for n in 1:N
        newforms = fulltype(Form{D,R,Poly{D,T}})[]
        for f in forms
            for p in polys
                push!(newforms, setindex(f, p, n))
            end
        end
        forms = newforms
    end

    return Space{D,T}(forms)
end

################################################################################

function bernstein(::Type{<:Space{D,0,T}}, p::Int) where {D,T}
    D::Int
    R = 0
    @assert 0 <= R <= D
    @assert p >= 0
    polys = fulltype(Form{D,R,Poly{D,T}})[]
    for i0 in
        CartesianIndex(ntuple(d -> 0, D)):CartesianIndex(ntuple(d -> p, D))
        i = SVector{D,Int}(i.I)
        if sum(i) == p
            poly = Poly{D,T}([Term{D,T}(i, one(T))])
            push!(polys, Form{D,R}((poly,)))
        end
    end
    return Space{D,T}(polys)
end

function bernstein(::Type{<:Space{D,R,T}}, p::Int) where {D,R,T}
    D::Int
    R::Int
    @assert 0 <= R <= D
    @assert p >= 0
    polys = Poly{D,T}[]
    for i0 in
        CartesianIndex(ntuple(d -> 0, D)):CartesianIndex(ntuple(d -> p, D))
        i = SVector{D,Int}(i.I)
        if sum(i) == p
            push!(polys, Poly{D,T}([Term{D,T}(i, one(T))]))
        end
    end
    return Space{D,T}(polys)
end

################################################################################

end
