module SimplexPoly

using ComputedFieldTypes
using DifferentialForms
using LinearAlgebra
using SparseArrays
using StaticArrays

################################################################################

div_complex(x::Complex, y) = Complex(real(x) ÷ y, imag(x) ÷ y)
rationalize_complex(T, x::Complex; kws...) = Complex(rationalize(T, real(x); kws...), rationalize(T, imag(x); kws...))

################################################################################

abstract type PType end
# x^k
struct Pow <: PType end
# exp(ikx)
struct Exp <: PType end
export PType, Pow, Exp
Base.isequal(::Type{Pow}, ::Type{Pow}) = true
Base.isequal(::Type{Exp}, ::Type{Exp}) = true
Base.isequal(::Type{<:PType}, ::Type{<:PType}) = false
Base.isless(::Type{Pow}, ::Type{Exp}) = true
Base.isless(::Type{<:PType}, ::Type{<:PType}) = false

export Term
struct Term{P,D,T}
    powers::SVector{D,Int}
    coeff::T
    function Term{P,D,T}(powers::SVector{D,Int}, coeff::T) where {P<:PType,D,T}
        @assert P ≡ Pow || P ≡ Exp
        D::Int
        @assert D >= 0
        @assert all(>=(0), powers)
        return new{P,D,T}(powers, coeff)
    end
end
Term{P,D}(powers::SVector{D,Int}, coeff::T) where {P<:PType,D,T} = Term{P,D,T}(powers, coeff)
Term{P}(powers::SVector{D,Int}, coeff::T) where {P<:PType,D,T} = Term{P,D,T}(powers, coeff)
Base.convert(::Type{Term{P,D,T}}, term::Term{P,D}) where {P,D,T} = map(x -> convert(T, x), term)

export ptype
ptype(::Type{<:Term{P}}) where {P} = P
ptype(x::Term) = ptype(typeof(x))

function Base.show(io::IO, ::MIME"text/plain", x::Term{P,D,T}) where {P,D,T}
    skiptype = get(io, :typeinfo, Any) <: Term{P,D,T}
    if !skiptype
        print(io, "Term{$P}(")
    end
    print(io, x.coeff, " * ", x.powers)
    if !skiptype
        print(io, ")")
    end
    return nothing
end

Base.:(==)(x::Term{P,D}, y::Term{P,D}) where {P,D} = x.powers == y.powers && x.coeff == y.coeff

Base.isequal(x::Term, y::Term) = isequal((ptype(x), x.powers, x.coeff), (ptype(y), y.powers, y.coeff))
function Base.isless(x::Term{P,D,<:Real} where {P,D}, y::Term{P,D,<:Real} where {P,D})
    return isless((ptype(x), x.powers, x.coeff), (ptype(y), y.powers, y.coeff))
end
function Base.isless(x::Term{P,D,<:Complex} where {P,D}, y::Term{P,D,<:Complex} where {P,D})
    return isless((ptype(x), x.powers, real(x.coeff), imag(x.coeff)), (ptype(y), y.powers, real(y.coeff), imag(y.coeff)))
end
Base.isless(x::Term, y::Term) = error()
export compare
function compare(x::Term, y::Term)
    # Note reversed signs; we want (1,0) < (0,1)
    x.powers > y.powers && return -1
    x.powers < y.powers && return +1
    return 0
end

function Base.map(f, x::Term{P,D}, ys::Term{P,D}...) where {P,D}
    @assert all(x.powers == y.powers for y in ys)
    return Term{P,D}(x.powers, map(f, x.coeff, (y.coeff for y in ys)...))
end

Base.zero(::Type{Term{P,D,T}}) where {P,D,T} = Term{P,D,T}(zero(SVector{D,Int}), zero(T))
Base.zero(::Term{P,D,T}) where {P,D,T} = zero(Term{P,D,T})
Base.iszero(x::Term) = iszero(x.coeff)

Base.one(::Type{Term{P,D,T}}) where {P,D,T} = Term{P,D,T}(zero(SVector{D,Int}), one(T))
Base.one(::Term{P,D,T}) where {P,D,T} = one(Term{P,D,T})
Base.isone(x::Term) = iszero(x.powers) && isone(x.coeff)

Base.:+(x::Term) = map(+, x)
Base.:-(x::Term) = map(-, x)
Base.conj(x::Term) = map(conj, x)

Base.:+(x::Term{P,D}, y::Term{P,D}) where {P,D} = map(+, x, y)
Base.:-(x::Term{P,D}, y::Term{P,D}) where {P,D} = map(-, x, y)
Base.:*(a, x::Term) = map(c -> a * c, x)
Base.:*(x::Term, a) = map(c -> c * a, x)
Base.:\(a, x::Term) = map(c -> a \ c, x)
Base.:/(x::Term, a) = map(c -> c / a, x)
Base.div(x::Term, a) = map(c -> div(c, a), x)
Base.mod(x::Term, a) = map(c -> mod(c, a), x)

Base.:*(x::Term{P,D}, y::Term{P,D}) where {P,D} = Term{P,D}(x.powers + y.powers, x.coeff * y.coeff)
function Base.:^(x::Term, n::Integer)
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
function deriv(term::Term{Pow,D}, dir::Int) where {D}
    D::Int
    @assert D >= 0
    @assert 1 <= dir <= D
    p = term.powers[dir]
    p == 0 && return zero(term)
    # D[x^p] = p x^(p-1)
    return Term{Pow,D}(setindex(term.powers, p - 1, dir), p * term.coeff)
end
function deriv(term::Term{Exp,D}, dir::Int) where {D}
    D::Int
    @assert D >= 0
    @assert 1 <= dir <= D
    p = term.powers[dir]
    p == 0 && return zero(term)
    # D[exp(i p x)] = i p exp(i p x)
    return Term{Exp,D}(setindex(term.powers, p, dir), im * p * term.coeff)
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
function koszul(term::Term{Pow,D}, dir::Int) where {D}
    D::Int
    @assert D >= 0
    @assert 1 <= dir <= D
    p = term.powers[dir]
    return Term{Pow,D}(setindex(term.powers, p + 1, dir), term.coeff)
end
# function koszul(term::Term{Exp,D}, dir::Int) where {D}
#     D::Int
#     @assert D >= 0
#     @assert 1 <= dir <= D
#     p = term.powers[dir]
#     return Term{Exp,D}(setindex(term.powers, p, dir), term.coeff)
# end

export integral
function integral(term::Term{Pow,D,T}) where {D,T}
    if T <: Real
        R = T <: Union{Integer,Rational} ? Rational{BigInt} : T <: AbstractFloat ? T : Nothing
    elseif T <: Complex
        R = T <: Union{Complex{<:Integer},Complex{<:Rational}} ? Complex{Rational{BigInt}} :
            T <: Complex{<:AbstractFloat} ? T : Nothing
    else
        R = Nothing
    end
    @assert R ≢ Nothing
    D == 0 && return R(term.coeff)
    # See <https://math.stackexchange.com/questions/207073/definite-integral-over-a-simplex>
    # We set ν₀ = 1 since the respective term is absent
    return (term.coeff * (prod(factorial(big(p)) for p in term.powers) // factorial(big(sum(term.powers) + length(term.powers)))))::R
end

################################################################################

export Poly
struct Poly{P,D,T} <: Number
    terms::Vector{Term{P,D,T}}
    function Poly{P,D,T}(terms::Vector{Term{P,D,T}}) where {P,D,T}
        D::Int
        @assert D >= 0
        terms = combine(terms)
        return new{P,D,T}(terms)
    end
end
Poly{P,D}(terms::Vector{Term{P,D,T}}) where {P,D,T} = Poly{P,D,T}(terms)
Poly{P}(terms::Vector{Term{P,D,T}}) where {P,D,T} = Poly{P,D,T}(terms)
Poly(terms::Vector{Term{P,D,T}}) where {P,D,T} = Poly{P,D,T}(terms)
Base.convert(::Type{Poly{P,D,T}}, poly::Poly{P,D}) where {P,D,T} = map(x -> convert(T, x), poly)

function combine(terms::Vector{Term{P,D,T}}) where {P,D,T}
    terms = sort(terms; by=(t -> t.powers), rev=true)
    isempty(terms) && return terms
    newterms = Term{P,D,T}[]
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

function Base.show(io::IO, mime::MIME"text/plain", p::Poly{P,D,T}) where {P,D,T}
    skiptype = get(io, :typeinfo, Any) <: Poly{P,D,T}
    if !skiptype
        print(io, "Poly")
    end
    # print(io, "[")
    print(io, "(")
    for (i, term) in enumerate(p.terms)
        # i > 1 && print(io, ", ")
        i > 1 && print(io, " + ")
        show(IOContext(io, :compact => true, :typeinfo => Term{P,D,T}), mime, term)
    end
    # print(io, "]")
    print(io, ")")
    return nothing
end

Base.:(==)(p::Poly{P,D}, q::Poly{P,D}) where {P,D} = p.terms == q.terms

Base.isequal(p::Poly, q::Poly) = isequal(p.terms, q.terms)
Base.isless(p::Poly, q::Poly) = isless(p.terms, q.terms)
Base.hash(p::Poly, h::UInt) = hash(p.terms, hash(0x0b9503a7, h))
export compare
function compare(p::Poly, q::Poly)
    for i in 1:min(length(p.terms), length(q.terms))
        c = compare(p.terms[i], q.terms[i])
        c != 0 && return c
    end
    # Polynomials with more terms come first since a non-zero term
    # comes before the zero term
    length(p.terms) > length(q.terms) && return -1
    length(p.terms) < length(q.terms) && return +1
    return 0
end

Base.map(f, p::Poly) = Poly(map(t -> map(f, t), p.terms))

function Base.map(f, p::Poly{P,D,T}, q::Poly{P,D,U}) where {P,D,T,U}
    R = typeof(f(zero(T), zero(U)))
    terms = Term{P,D,R}[]
    i = j = 1
    ni = length(p.terms)
    nj = length(q.terms)
    while i <= ni || j <= nj
        usei = usej = false
        if i <= ni && j <= nj
            c = compare(p.terms[i], q.terms[j])
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
    return Poly{P,D,R}(terms)
end

Base.reduce(op, p::Poly; init=Base._InitialValue()) = reduce(op, p.terms; init=init)

Base.mapreduce(f, op, p::Poly; init=Base._InitialValue()) = reduce(op, f(p.terms); init=init)

function Base.mapreduce(f, op, p::Poly{P,D}, q::Poly{P,D}; init) where {P,D}
    res = init
    i = j = 1
    ni = length(p.terms)
    nj = length(q.terms)
    while i <= ni || j <= nj
        usei = usej = false
        if i <= ni && j <= nj
            c = compare(p.terms[i], q.terms[j])
            usei = c <= 0
            usej = c >= 0
        else
            usei = i <= ni
            usej = !usei
        end
        pi = usei ? p.terms[i] : map(zero, q.terms[j])
        qj = usej ? q.terms[j] : map(zero, p.terms[i])
        res = op(res, f(pi.coeff, qj.coeff))::typeof(init)
        i += usei
        j += usej
    end
    return res
end

function Base.zero(::Type{Poly{P,D,T}}) where {P,D,T}
    D::Int
    @assert D >= 0
    return Poly{P,D,T}(Vector{Term{P,D,T}}())
end
Base.zero(::Poly{P,D,T}) where {P,D,T} = zero(Poly{P,D,T})
Base.iszero(x::Poly{P,D,T}) where {P,D,T} = isempty(x.terms)

Base.one(::Type{Poly{P,D,T}}) where {P,D,T} = Poly{P,D,T}([one(Term{P,D,T})])
Base.one(::Poly{P,D,T}) where {P,D,T} = one(Poly{P,D,T})
Base.isone(x::Poly{P,D,T}) where {P,D,T} = length(x.terms) == 1 && isone(x.terms[1])

export unit
function Forms.unit(::Type{Poly{P,D,T}}, dir::Int, coeff=one(T)) where {P,D,T}
    D::Int
    @assert D >= 0
    @assert 1 <= dir <= D
    term = Term{P,D,T}(SVector{D,Int}(d == dir for d in 1:D), coeff)
    return Poly{P,D,T}([term])
end

Base.:+(p::Poly) = map(+, p)
Base.:-(p::Poly) = map(-, p)
Base.conj(p::Poly) = map(conj, p)

Base.:+(p::Poly{P,D}, q::Poly{P,D}) where {P,D} = map(+, p, q)
Base.:-(p::Poly{P,D}, q::Poly{P,D}) where {P,D} = map(-, p, q)
Base.:*(a::Number, p::Poly) = map(t -> a * t, p)
Base.:*(p::Poly, a::Number) = map(t -> t * a, p)
Base.:\(a::Number, p::Poly) = map(t -> a \ t, p)
Base.:/(p::Poly, a::Number) = map(t -> t / a, p)
Base.div(p::Poly, a::Number) = map(t -> div(t, a), p)
Base.mod(p::Poly, a::Number) = map(t -> mod(t, a), p)

Base.:*(p::Poly{P,D}, q::Poly{P,D}) where {P,D} = Poly{P,D}([t * u for t in p.terms for u in q.terms])
function Base.:^(x::Poly, n::Integer)
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

deriv(poly::Poly{P}, dir::Int) where {P} = Poly{P}(map(t -> deriv(t, dir), poly.terms))

koszul(poly::Poly{P}, dir::Int) where {P} = Poly{P}(map(t -> koszul(t, dir), poly.terms))

function integral(poly::Poly{P,D,T}) where {P,D,T}
    if T <: Real
        R = T <: Union{Integer,Rational} ? Rational{BigInt} : T <: AbstractFloat ? T : Nothing
    elseif T <: Complex
        R = T <: Union{Complex{<:Integer},Complex{<:Rational}} ? Complex{Rational{BigInt}} :
            T <: Complex{<:AbstractFloat} ? T : Nothing
    else
        R = Nothing
    end
    @assert R ≢ Nothing
    isempty(poly.terms) && return zero(R)
    return sum(integral(term) for term in poly.terms)::R
end

# LinearAlgebra.dot(p::Poly{P,D}, q::Poly{P,D}) where {P,D} = integral(conj(p) * q)

function LinearAlgebra.dot(p::Poly{P,D,T1}, q::Poly{P,D,T2}) where {P,D,T1,T2}
    init = zero(T1) ⋅ zero(T2)
    return mapreduce(⋅, +, p, q; init=init)::typeof(init)
end

################################################################################

# Note: PolySpace is unused

export PolySpace
struct PolySpace{P,D,T}
    polys::Vector{Poly{P,D,T}}
    function PolySpace{P,D,T}(polys::Vector{Poly{P,D,T}}) where {P,D,T}
        D::Int
        @assert D >= 0
        polys = combine(polys)
        return new{P,D,T}(polys)
    end
end

function combine(polys::Vector{Poly{P,D,T}}) where {P,D,T}
    # TODO: This is wrong -- need to combine polynomials that are
    # multiples of each other
    polys = sort(polys; rev=true)
    unique!(polys)
    filter!(p -> !iszero(p), polys)
    return polys
end

PolySpace{P,D,T}() where {P,D,T} = PolySpace{P,D,T}(Poly{P,D,T}[])

function Base.show(io::IO, mime::MIME"text/plain", ps::PolySpace{P,D,T}) where {P,D,T}
    skiptype = get(io, :typeinfo, Any) <: PolySpace{P,D,T}
    if !skiptype
        print(io, "PolySpace")
    end
    print(io, "[")
    for (i, poly) in enumerate(ps.polys)
        i > 1 && print(io, ", ")
        show(IOContext(io, :compact => true, :typeinfo => Poly{P,D,T}), mime, poly)
    end
    print(io, "]")
    return nothing
end

Base.:(==)(ps::PolySpace{P,D}, qs::PolySpace{P,D}) where {P,D} = ps.polys == qs.polys

Base.isequal(ps::PolySpace, qs::PolySpace) = isequal(ps.polys, qs.polys)
Base.isless(ps::PolySpace, qs::PolySpace) = isless(ps.polys, qs.polys)
Base.hash(ps::PolySpace, h::UInt) = hash(ps.polys, hash(0xa3e4cfbf, h))

Base.map(f, ps::PolySpace{P,D,T}) where {P,D,T} = PolySpace{P,D,T}(map(f, ps.polys))

Base.isempty(ps::PolySpace) = isempty(ps.polys)
Base.length(ps::PolySpace) = length(ps.polys)
Base.issubset(ps::PolySpace{P,D,T}, qs::PolySpace{P,D,T}) where {P,D,T} = issubset(Set(ps.polys), Set(qs.polys))
Base.union(ps::PolySpace{P,D,T}, qs::PolySpace{P,D,T}) where {P,D,T} = PolySpace{P,D,T}([ps.polys; qs.polys])
function Base.setdiff(ps::PolySpace{P,D,T}, qs::PolySpace{P,D,T}) where {P,D,T}
    return PolySpace{P,D,T}(collect(setdiff(Set(ps.polys), Set(qs.polys))))
end

Base.zero(::Type{PolySpace{P,D,T}}) where {P,D,T} = PolySpace{P,D,T}()
Base.iszero(ps::PolySpace) = isempty(ps)
# Base.:+(ps::PolySpace) = ps
# Base.:+(ps::PolySpace{P,D,T}, qs::PolySpace{P,D,T}) where {P,D,T} = union(ps, qs)
# Base.:*(a, ps::PolySpace) = map(p -> a * p, ps)
# Base.:*(ps::PolySpace, a) = map(p -> p * a, ps)
# Base.:\(a, ps::PolySpace) = map(p -> a \ p, ps)
# Base.:/(ps::PolySpace, a) = map(p -> p / a, ps)

deriv(ps::PolySpace, dir::Int) = map(p -> deriv(p, dir), ps)
koszul(ps::PolySpace, dir::Int) = map(p -> koszul(p, dir), ps)

################################################################################

function deriv(f::Form{D,R,T}) where {D,R,P,T}
    D::Int
    R::Int
    @assert 0 <= R < D
    r = zero(Form{D,R + 1,T})
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
                rbits = setindex(bits, true, dir)
                rn = Forms.bit2lin(Val(D), Val(R + 1), rbits)
                r = setindex(r, r[rn] + s * deriv(f[n], dir), rn)
            end
        end
    end
    return r::Form{D,R + 1,T}
end

function koszul(f::Form{D,R,T}) where {D,R,P,T}
    D::Int
    R::Int
    @assert 0 < R <= D
    r = zero(Form{D,R - 1,T})
    N = length(f)
    for n in 1:N
        bits = Forms.lin2bit(Val(D), Val(R), n)
        for dir in 1:D
            if bits[dir]
                parity = false
                for d in 1:(dir - 1)
                    parity ⊻= bits[d]
                end
                s = bitsign(parity)
                rbits = setindex(bits, false, dir)
                rn = Forms.bit2lin(Val(D), Val(R - 1), rbits)
                r = setindex(r, r[rn] + s * koszul(f[n], dir), rn)
            end
        end
    end
    return r::Form{D,R - 1,T}
end

function integral(f::Form{D,R,Poly{P,D,T}}) where {D,R,P,T}
    D::Int
    R::Int
    @assert 0 <= R <= D
    if T <: Real
        U = T <: Union{Integer,Rational} ? Rational{BigInt} : T <: AbstractFloat ? T : Nothing
    elseif T <: Complex
        U = T <: Union{Complex{<:Integer},Complex{<:Rational}} ? Complex{Rational{BigInt}} :
            T <: Complex{<:AbstractFloat} ? T : Nothing
    else
        U = Nothing
    end
    @assert U ≢ Nothing
    return map(integral, f)::Form{D,R,U}
end

# function LinearAlgebra.dot(f::Form{D,R,Poly{P,D,T}},
#                            g::Form{D,R,Poly{P,D,T}}) where {D,R,P,T}
#     U = T <: Union{Integer,Rational} ? Rational{BigInt} :
#         T <: AbstractFloat ? T : Nothing
#     r = zero(U)
#     for (fi, gi) in zip(f.elts, g.elts)
#         r += (fi ⋅ gi)::U
#     end
#     return r
# end

# The dot product of forms uses the wedge product and hodge dual.
# These are not defined for polynomials. We thus roll our own dot
# product which falls back onto the dot product for polynomials.
LinearAlgebra.dot(f::Form{D,R,<:Poly}, g::Form{D,R,<:Poly}) where {D,R} = error("not implemented")
function LinearAlgebra.dot(f::Form{D,R,Poly{P,D,T1}}, g::Form{D,R,Poly{P,D,T2}}) where {D,R,P,T1,T2}
    T = typeof(zero(T1) ⋅ zero(T2))
    r = zero(T)
    for (fp, gp) in zip(f.elts, g.elts)
        r += (fp ⋅ gp)::T
    end
    return r::T
end

################################################################################

export deriv1
function deriv1(f::TensorForm{D,R1,R2,T}) where {D,R1,R2,P,T}
    D::Int
    R1::Int
    R2::Int
    @assert 0 <= R1 < D
    @assert 0 <= R2 <= D
    r = zero(Form{D,R1 + 1,fulltype(Form{D,R2,T})})
    N = length(f.form)
    for n in 1:N
        bits = Forms.lin2bit(Val(D), Val(R1), n)
        for dir in 1:D
            if !bits[dir]
                parity = false
                for d in 1:(dir - 1)
                    parity ⊻= bits[d]
                end
                s = bitsign(parity)
                rbits = setindex(bits, true, dir)
                rn = Forms.bit2lin(Val(D), Val(R1 + 1), rbits)
                r = setindex(r, r[rn] + s * Form{D,R2,T}(deriv.(f.form[n], dir)), rn)
            end
        end
    end
    return TensorForm(r)::TensorForm{D,R1 + 1,R2,T}
end

export koszul1
function koszul1(f::TensorForm{D,R1,R2,T}) where {D,R1,R2,P,T}
    D::Int
    R1::Int
    R2::Int
    @assert 0 < R1 <= D
    @assert 0 <= R2 <= D
    r = zero(Form{D,R1 - 1,fulltype(Form{D,R2,T})})
    N = length(f.form)
    for n in 1:N
        bits = Forms.lin2bit(Val(D), Val(R1), n)
        for dir in 1:D
            if bits[dir]
                parity = false
                for d in 1:(dir - 1)
                    parity ⊻= bits[d]
                end
                s = bitsign(parity)
                rbits = setindex(bits, false, dir)
                rn = Forms.bit2lin(Val(D), Val(R1 - 1), rbits)
                r = setindex(r, r[rn] + s * Form{D,R2,T}(koszul.(f.form[n], dir)), rn)
            end
        end
    end
    return TensorForm(r)::TensorForm{D,R1 - 1,R2,T}
end

export deriv2
function deriv2(f::TensorForm{D,R1,R2,T}) where {D,R1,R2,P,T}
    D::Int
    R1::Int
    R2::Int
    @assert 0 ≤ R1 ≤ D
    @assert 0 ≤ R2 < D
    return TensorForm(Form{D,R1,fulltype(Form{D,R2 + 1,T})}(deriv.(f.form)))::TensorForm{D,R1,R2 + 1,T}
end

export koszul2
function koszul2(f::TensorForm{D,R1,R2,T}) where {D,R1,R2,P,T}
    D::Int
    R1::Int
    R2::Int
    @assert 0 ≤ R1 ≤ D
    @assert 0 < R2 ≤ D
    return TensorForm(Form{D,R1,fulltype(Form{D,R2 - 1,T})}(koszul.(f.form)))::TensorForm{D,R1,R2 - 1,T}
end

# LinearAlgebra.dot(f::TensorForm{D,R1,R2,<:Poly}, g::TensorForm{D,R1,R2,<:Poly}) where {D,R1,R2} = f.form ⋅ g.form
LinearAlgebra.dot(f::TensorForm{D,R1,R2,<:Poly}, g::TensorForm{D,R1,R2,<:Poly}) where {D,R1,R2} = error("not implemented")
function LinearAlgebra.dot(f::TensorForm{D,R1,R2,Poly{P,D,T1}}, g::TensorForm{D,R1,R2,Poly{P,D,T2}}) where {D,R1,R2,P,T1,T2}
    T = typeof(zero(T1) ⋅ zero(T2))
    r = zero(T)
    for (fp1, gp1) in zip(f.form.elts, g.form.elts)
        for (fp2, gp2) in zip(fp1.elts, gp1.elts)
            r += (fp2 ⋅ gp2)::T
        end
    end
    return r::T
end

################################################################################

function powers2ind(powers::SVector{D,Int}, maxp::Int) where {D}
    ind = 0
    for d in D:-1:1
        @assert 0 <= powers[d] <= maxp
        ind = (maxp + 1) * ind + powers[d]
    end
    return ind + 1
end

function poly2vec(poly::Poly{P,D,T}, maxp::Int) where {P,D,T}
    nrows = (maxp + 1)^D
    I = Int[]
    V = T[]
    for term in poly.terms
        push!(I, powers2ind(term.powers, maxp))
        push!(V, term.coeff)
    end
    return sparsevec(I, V, nrows)
end

export form2vec
function form2vec(form::Form{D,R,Poly{P,D,T}}, maxp::Int) where {D,R,P,T}
    stride = (maxp + 1)^D
    nrows = length(form) * stride
    I = Int[]
    V = T[]
    for (n, poly) in enumerate(form)
        offset = (n - 1) * stride
        vec = poly2vec(poly, maxp)
        Is, Vs = findnz(vec)
        append!(I, Is .+ offset)
        append!(V, Vs)
    end
    return sparsevec(I, V, nrows)
end

export forms2mat
function forms2mat(forms::Vector{<:Form{D,R,Poly{P,D,T}}}, maxp::Int) where {D,R,P,T}
    nrows = length(Form{D,R}) * (maxp + 1)^D
    ncols = length(forms)
    I = Int[]
    J = Int[]
    V = T[]
    for (j, form) in enumerate(forms)
        vec = form2vec(form, maxp)
        Is, Vs = findnz(vec)
        append!(I, Is)
        append!(J, fill(j, length(Is)))
        append!(V, Vs)
    end
    return sparse(I, J, V, nrows, ncols)
end

export maxpower
maximum0(xs) = isempty(xs) ? 0 : maximum(xs)
maxpower(term::Term) = maximum0(term.powers)
maxpower(poly::Poly) = maximum0(maxpower(term) for term in poly.terms)
maxpower(form::Form) = maximum0(maxpower(elt) for elt in form)
maxpower(forms::Vector{<:Form}) = maximum0(maxpower(form) for form in forms)

################################################################################

export Basis
@computed struct Basis{P,D,R,T}
    forms::Vector{fulltype(Form{D,R,Poly{P,D,T}})}
    function Basis{P,D,R,T}(forms::Vector{<:Form{D,R,Poly{P,D,T}}}) where {P,D,R,T}
        D::Int
        R::Int
        @assert 0 ≤ R ≤ D
        forms = combine(forms)
        return new{P,D,R,T}(forms)
    end
end

export compare
function compare(x::Form{D,R,<:Poly{P,D}}, y::Form{D,R,<:Poly{P,D}}) where {D,R,P}
    px = maxpower(x)
    py = maxpower(y)
    c = px - py
    c != 0 && return c

    for (fx, fy) in zip(x.elts, y.elts)
        c = compare(fx, fy)
        c != 0 && return c
    end

    return 0
end

function combine(forms::Vector{<:Form{D,R,Poly{P,D,T}}}) where {D,R,P,T}
    forms = sort(forms; rev=true)
    unique!(forms)
    filter!(form -> !iszero(form), forms)
    map!(normalize, forms, forms)
    sort!(forms; lt=(x, y) -> compare(x, y) < 0)
    unique!(forms)
    # Gram-Schmidt creates polynomials with many and large coefficients
    forms = gram_schmidt(forms)

    # basisforms = fulltype(Form{D,R,Poly{P,D,T}})[]
    # for form in forms
    #     if !is_in_span(form, basisforms)
    #         push!(basisforms, form)
    #     end
    # end
    # forms = basisforms

    return forms
end

function normalize(form::Form{D,R,Poly{P,D,T}}) where {D,R,P,T<:Integer}
    coeffs = T[]
    for poly in form
        poly::Poly{P,D,T}
        for term in poly.terms
            term::Term{P,D,T}
            push!(coeffs, term.coeff)
        end
    end
    isempty(coeffs) && return form
    q = sign(coeffs[1]) * gcd(coeffs)
    r = map(x -> x ÷ q, form)
    return r
end

function normalize(form::Form{D,R,Poly{P,D,T}}) where {D,R,P,T<:Complex{<:Integer}}
    RT = typeof(real(zero(T)))
    coeffs = RT[]
    for poly in form
        poly::Poly{P,D,T}
        for term in poly.terms
            term::Term{P,D,T}
            real(term.coeff) ≠ 0 && push!(coeffs, real(term.coeff))
            imag(term.coeff) ≠ 0 && push!(coeffs, imag(term.coeff))
        end
    end
    isempty(coeffs) && return form
    q = sign(coeffs[1]) * gcd(coeffs)
    r = map(poly -> map(coeff -> div_complex(coeff, q), poly), form)
    return r
end

function normalize(form::Form{D,R,Poly{P,D,T}}) where {D,R,P,T<:Rational}
    I = typeof(numerator(one(T)))
    coeffs = T[]
    for poly in form
        poly::Poly{P,D,T}
        for term in poly.terms
            term::Term{P,D,T}
            push!(coeffs, term.coeff)
        end
    end
    isempty(coeffs) && return zero(Form{D,R,Poly{P,D,I}})
    q = T(lcm(denominator.(coeffs)))
    r = Form{D,R,Poly{P,D,I}}(q * form)
    return
end

function gram_schmidt(forms::Vector{<:Form{D,R,T}}) where {D,R,T}
    # @show :gram_schmidt forms
    ortho_forms = fulltype(Form{D,R,T})[]
    for form in forms
        # @show :loop form
        for oform in ortho_forms
            ovlp = oform ⋅ form
            scal = oform ⋅ oform
            # @show ovlp scal
            form = normalize(scal * form - ovlp * oform)
            # @show form
        end
        if !iszero(form)
            for oform in ortho_forms
                if !(iszero(oform ⋅ form))
                    @show :assert oform form oform ⋅ form forms ortho_forms
                end
                @assert iszero(oform ⋅ form)
            end
            push!(ortho_forms, form)
        end
    end
    return ortho_forms
end

# function Basis{P,D,R,T}() where {D,R,T}
#     return Basis{P,D,R,T}([])
# end

function Base.show(io::IO, mime::MIME"text/plain", b::Basis{P,D,R,T}) where {P,D,R,T}
    skiptype = get(io, :typeinfo, Any) <: Basis{P,D,R,T}
    if !skiptype
        print(io, "Basis{$P,$D,$R,$T}")
    end
    print(io, "[")
    for (i, form) in enumerate(b.forms)
        i > 1 && print(io, ", ")
        show(IOContext(io, :compact => true, :typeinfo => Form{D,R,Poly{P,D,T}}), mime, form)
    end
    print(io, "]")
    return nothing
end

function Base.:(==)(b1::Basis{P,D,R}, b2::Basis{P,D,R}) where {P,D,R}
    # return b1.forms == b2.forms
    # Fast paths
    b1.forms == b2.forms && return true
    isempty(b1.forms) != isempty(b2.forms) && return false
    # Use subset relations
    return issubset(b1, b2) && issubset(b2, b1)
end

Base.isequal(b1::Basis, b2::Basis) = isequal(b1.forms, b2.forms)
Base.isless(b1::Basis, b2::Basis) = isless(b1.forms, b2.forms)
Base.hash(b::Basis, h::UInt) = hash(b.forms, hash(0xe8439c4b, h))

Base.zero(::Type{<:Basis{P,D,R,T}}) where {P,D,R,T} = Basis{P,D,R,T}(fulltype(Form{D,R,Poly{P,D,T}})[])
Base.zero(::Basis{P,D,R,T}) where {P,D,R,T} = zero(Basis{P,D,R,T})
Base.iszero(basis::Basis) = isempty(basis.forms)

Forms.unit(form::Form{D,R,Poly{P,D,T}}) where {D,R,P,T} = Basis{P,D,R,T}([form])

export tensorsum, ⊕
Forms.tensorsum(basis1::Basis{P,D,R,T}, basis2::Basis{P,D,R,T}) where {P,D,R,T} = Basis{P,D,R,T}([basis1.forms; basis2.forms])

function tensordiff end
const ⊖ = tensordiff
export tensordiff, ⊖
function tensordiff(basis1::Basis{P,D,R,T}, basis2::Basis{P,D,R,T}) where {P,D,R,T}
    forms2 = Set(basis2.forms)
    forms = fulltype(Form{D,R,Poly{P,D,T}})[]
    for form in basis1.forms
        if form ∉ forms2
            push!(forms, form)
        end
    end
    return Basis{P,D,R,T}(forms)
end

Base.isempty(basis::Basis) = isempty(basis.forms)
Base.length(basis::Basis) = length(basis.forms)

function is_in_span(form::Form{D,R,Poly{P,D,T}}, forms::Vector{<:Form{D,R,Poly{P,D,T}}}) where {P,D,R,T}
    # Fast paths
    iszero(form) && return true
    isempty(forms) && return false
    any(form == f for f in forms) && return true
    # Convert representation to sparse vectors/matrices
    maxp = max(maxpower(form), maxpower(forms))
    fvec = form2vec(form, maxp)
    bmat = forms2mat(forms, maxp)
    # Sparse solver doesn't handle sparse vectors
    fvec = collect(fvec)
    # Is there a linear combination of the vectors in bmat that yields fvec?
    #     fⁱ = bⁱⱼ aʲ
    if T <: Integer
        # Use floating point numbers for integer polynomials
        avec = float.(bmat) \ float.(fvec)
        avec = rationalize.(BigInt, avec; tol=1e3 * eps())
    elseif T <: Complex{<:Integer}
        # Use floating point numbers for integer polynomials
        avec = float.(bmat) \ float.(fvec)
        avec = rationalize_complex.(BigInt, avec; tol=1e3 * eps())
    else
        @assert false
    end
    rvec = fvec - bmat * avec
    iszero(rvec) && return true
    if D ≥ 6 && maximum(abs.(rvec)) ≤ 1e5 * binomial(D, R) * eps()
        @info "`is_in_span` returns an approximate result"
        return true
    end
    return false
end

Base.in(form::Form{D,R,Poly{P,D,T}}, basis::Basis{P,D,R,T}) where {P,D,R,T} = is_in_span(form, basis.forms)

function Base.issubset(basis1::Basis{P,D,R}, basis2::Basis{P,D,R}) where {P,D,R}
    isempty(basis1.forms) && return true
    isempty(basis2.forms) && return false
    return all(f in basis2 for f in basis1.forms)
end

# Unused
function make_basis1(::Type{<:Basis{P,D,R,T}}, p::Int, cond) where {P,D,R,T}
    D::Int
    R::Int
    @assert 0 <= R <= D
    @assert p >= -1             # -1 is the empty basis
    polys = Poly{P,D,T}[]
    for i0 in CartesianIndex(ntuple(d -> 0, D)):CartesianIndex(ntuple(d -> p, D))
        i = SVector{D,Int}(i0.I)
        if cond(sum(i))
            push!(polys, Poly{P,D,T}([Term{P,D,T}(i, one(T))]))
        end
    end
    N = length(Form{D,R})
    forms = fulltype(Form{D,R,Poly{P,D,T}})[zero(Form{D,R,Poly{P,D,T}})]
    for n in 1:N
        newforms = fulltype(Form{D,R,Poly{P,D,T}})[]
        for f in forms
            for p in polys
                push!(newforms, setindex(f, p, n))
            end
        end
        forms = newforms
    end

    return Basis{P,D,R,T}(forms)
end

# Unused
function make_basis2(::Type{<:Basis{P,D,R,T}}, p::Int, cond) where {P,D,R,T}
    D::Int
    R::Int
    @assert 0 <= R <= D
    @assert p >= -1             # -1 is the empty basis
    N = length(Form{D,R})
    forms = fulltype(Form{D,R,Poly{P,D,T}})[zero(Form{D,R,Poly{P,D,T}})]
    for i0 in CartesianIndex(ntuple(d -> 0, D)):CartesianIndex(ntuple(d -> p, D))
        i = SVector{D,Int}(i0.I)
        if cond(sum(i))
            poly = Poly{P,D,T}([Term{P,D,T}(i, one(T))])
            form = Form{D,R,Poly{P,D,T}}(ntuple(n -> poly, N))
            push!(forms, form)
        end
    end
    return Basis{PnD,R,T}(forms)
end

function make_basis(::Type{<:Basis{P,D,R,T}}, p::Int, cond) where {P,D,R,T}
    D::Int
    R::Int
    @assert 0 <= R <= D
    @assert p >= -1             # -1 is the empty basis
    polys = Poly{P,D,T}[]
    for i0 in CartesianIndex(ntuple(d -> 0, D)):CartesianIndex(ntuple(d -> p, D))
        i = SVector{D,Int}(i0.I)
        if cond(sum(i))
            push!(polys, Poly{P,D,T}([Term{P,D,T}(i, one(T))]))
        end
    end
    N = length(Form{D,R})
    forms = fulltype(Form{D,R,Poly{P,D,T}})[]
    for n in 1:N
        for poly in polys
            form = Form{D,R,Poly{P,D,T}}(ntuple(m -> m == n ? poly : zero(Poly{P,D,T}), N))
            push!(forms, form)
        end
    end
    return Basis{P,D,R,T}(forms)
end

export full_basis
full_basis(::Type{<:Basis{P,D,R,T}}, p::Int) where {P,D,R,T} = make_basis(Basis{P,D,R,T}, p, <=(p))

export homogeneous_basis
homogeneous_basis(::Type{<:Basis{P,D,R,T}}, p::Int) where {P,D,R,T} = make_basis(Basis{P,D,R,T}, p, ==(p))

function deriv(basis::Basis{P,D,R,T}) where {P,D,R,T}
    D::Int
    R::Int
    @assert 0 <= R < D
    dforms = fulltype(Form{D,R + 1,Poly{P,D,T}})[]
    for form in basis.forms
        dform = deriv(form)
        !iszero(dform) && push!(dforms, dform)
    end
    return Basis{P,D,R + 1,T}(dforms)
end

function koszul(basis::Basis{P,D,R,T}) where {P,D,R,T}
    D::Int
    R::Int
    @assert 0 < R <= D
    κforms = fulltype(Form{D,R - 1,Poly{P,D,T}})[]
    for form in basis.forms
        κform = koszul(form)
        !iszero(κform) && push!(κforms, κform)
    end
    return Basis{P,D,R - 1,T}(κforms)
end

################################################################################

# function bernstein(::Type{<:Basis{P,D,0,T}}, p::Int) where {D,T}
#     D::Int
#     R = 0
#     @assert 0 <= R <= D
#     @assert p >= 0
#     polys = fulltype(Form{D,R,Poly{P,D,T}})[]
#     for i0 in
#         CartesianIndex(ntuple(d -> 0, D)):CartesianIndex(ntuple(d -> p, D))
#         i = SVector{D,Int}(i.I)
#         if sum(i) == p
#             poly = Poly{P,D,T}([Term{P,D,T}(i, one(T))])
#             push!(polys, Form{D,R}((poly,)))
#         end
#     end
#     return Basis{P,D,T}(polys)
# end
# 
# function bernstein(::Type{<:Basis{P,D,R,T}}, p::Int) where {D,R,T}
#     D::Int
#     R::Int
#     @assert 0 <= R <= D
#     @assert p >= 0
#     polys = Poly{P,D,T}[]
#     for i0 in
#         CartesianIndex(ntuple(d -> 0, D)):CartesianIndex(ntuple(d -> p, D))
#         i = SVector{D,Int}(i.I)
#         if sum(i) == p
#             push!(polys, Poly{P,D,T}([Term{P,D,T}(i, one(T))]))
#         end
#     end
#     return Basis{P,D,T}(polys)
# end

################################################################################

export polynomial_complex
"""
polynomial_complex

Pᵣ Λᵏ

The spaces with constant r form a complex. NOTE: This differs from Arnold's definition!
"""
function polynomial_complex(::Type{P}, ::Val{D}, ::Type{T}, p::Int) where {P,D,T}
    D::Int
    @assert 0 <= D
    @assert p >= -1

    cc = Dict{Int,Basis{P,D,R,T} where R}()

    for R in 0:D
        cc[R] = full_basis(Basis{P,D,R,T}, max(-1, p - R))::Basis{P,D,R,T}
    end

    return cc
end

export trimmed_polynomial_complex
"""
trimmed_polynomial_complex

Pᵣ⁻ Λᵏ

The spaces with constant r form a complex.
"""
function trimmed_polynomial_complex(::Type{P}, ::Val{D}, ::Type{T}, p::Int) where {P,D,T}
    D::Int
    @assert 0 <= D
    @assert p >= -1

    cc = Dict{Int,Basis{P,D,R,T} where R}()

    for R in 0:D
        if R == 0
            b = full_basis(Basis{P,D,R,T}, p)
        else
            b = full_basis(Basis{P,D,R,T}, max(-1, p - 1))
            if R + 1 <= D
                b = b ⊕ koszul(homogeneous_basis(Basis{P,D,R + 1,T}, p - 1))
            end
        end
        cc[R] = b::Basis{P,D,R,T}
    end

    return cc
end

export extended_trimmed_polynomial_complex
"""
extended_trimmed_polynomial_complex

This is experimental; note that this is not actually a complex.
"""
function extended_trimmed_polynomial_complex(::Type{P}, ::Val{D}, ::Type{T}, p::Int) where {P,D,T}
    D::Int
    @assert 0 <= D
    @assert p >= -1

    cc = Dict{Int,Basis{P,D,R,T} where R}()

    for R in 0:D
        if R == 0
            b = full_basis(Basis{P,D,R,T}, p)
        else
            b = full_basis(Basis{P,D,R,T}, max(-1, p - 1))
            if R + 1 <= D
                b = b ⊕ koszul(homogeneous_basis(Basis{P,D,R + 1,T}, p - 1))
            end
            if R - 1 >= 0
                b = b ⊕ deriv(homogeneous_basis(Basis{P,D,R - 1,T}, p + 1))
            end
        end
        cc[R] = b::Basis{P,D,R,T}
    end

    return cc
end

################################################################################

export whitney
"""
whitney

Calculate Whitney forms
"""
function whitney(::Type{P}, ::Val{D1}, ::Type{T}, inds::SVector{N,Int}) where {P,D1,T,N}
    D1::Int
    N::Int
    @assert 1 <= D1
    @assert 1 <= N <= D1
    R = N - 1
    @assert all(1 <= inds[n] <= D1 for n in 1:N)
    @assert all(inds[n] < inds[n + 1] for n in 1:(N - 1))
    ϕ = zero(Form{D1,R,Poly{P,D1,T}})
    q = factorial(N - 1)
    for n in 1:N
        inds′ = deleteat(inds, n)
        f = unit(Form{D1,R,T}, inds′)
        p = q * bitsign(n - 1) * unit(Poly{P,D1,T}, inds[n])
        ϕ += p * f
    end
    return ϕ::Form{D1,R,Poly{P,D1,T}}
end
whitney(::Type{P}, ::Val{D1}, ::Type{T}, inds::NTuple{N,Int}) where {P,D1,T,N} = whitney(P, Val(D1), T, SVector{N,Int}(inds))
whitney(p::Type, d::Val, inds::SVector) = whitney(p, d, Int, inds)
whitney(p::Type, d::Val, inds::NTuple) = whitney(p, d, Int, inds)
function whitney(::Type{Basis{P,D1,R,T}}) where {P,D1,R,T}
    D1::Int
    R::Int
    @assert 1 <= D1
    @assert 0 <= R <= D1 - 1
    nelts = binomial(D1, R + 1)
    forms = [whitney(P, Val(D1), T, Forms.lin2lst(Val(D1), Val(R + 1), n)) for n in 1:nelts]
    return Basis{P,D1,R,T}(forms)
end

export whitney_support
"""
whitney_support

Determine on what part of a simplex (vertex, edge, face, ...) a
particular Whitney forms should live.
"""
whitney_support(::Type{P}, ::Val{D1}, inds::SVector{N,Int}) where {P,D1,N} = inds

################################################################################

export barycentric2cartesian
function barycentric2cartesian(λterm::Term{P,D1,T}) where {P,D1,T}
    D1::Int
    @assert 1 <= D1
    D = D1 - 1
    # This assumes a standard simplex:
    #     λᵢ = xᵢ
    #     λₙ = 1 - Σᵢ λᵢ
    λn = one(Poly{P,D,T})
    if D > 0
        λn -= sum(unit(Poly{P,D,T}, d) for d in 1:D)
    end
    xpoly = Poly{P,D,T}([Term{P,D,T}(deleteat(λterm.powers, D1), λterm.coeff)])
    xpoly *= λn^λterm.powers[D1]
    return xpoly::Poly{P,D,T}
end
function barycentric2cartesian(λpoly::Poly{P,D1,T}) where {P,D1,T}
    D1::Int
    @assert 1 <= D1
    D = D1 - 1
    isempty(λpoly.terms) && return zero(Poly{P,D,T})
    xpoly = sum(barycentric2cartesian(λterm) for λterm in λpoly.terms)
    return xpoly::Poly{P,D,T}
end
function barycentric2cartesian(λform::Form{D1,R,Poly{P,D1,T}}) where {D1,R,P,T}
    D1::Int
    R::Int
    @assert 1 <= D1
    D = D1 - 1
    @assert 0 <= R <= D
    # This assumes a standard simplex:
    #     λᵢ = xᵢ
    #     λₙ = 1 - Σᵢ λᵢ
    #     dλᵢ = dxᵢ
    #     dλₙ = -Σᵢ dλᵢ
    if D == 0
        λpoly = λform[]
        xpoly = barycentric2cartesian(λpoly)
        xform = Form{D,R,Poly{P,D,T}}((xpoly,))
    else
        dλn = -sum(unit(Form{D,1,Poly{P,D,T}}, d) for d in 1:D)
        xform = zero(Form{D,R,Poly{P,D,T}})
        for (λlin, λpoly) in enumerate(λform)
            λbits = Forms.lin2bit(Val(D1), Val(R), λlin)
            xbits = deleteat(λbits, D1)
            if λbits[D1]
                xlin = Forms.bit2lin(Val(D), Val(R - 1), xbits)
                xunit = unit(Form{D,R - 1,Poly{P,D,T}}, xlin) ∧ dλn
            else
                xlin = Forms.bit2lin(Val(D), Val(R), xbits)
                xunit = unit(Form{D,R,Poly{P,D,T}}, xlin)
            end
            xpoly = barycentric2cartesian(λpoly)
            xform += xpoly * xunit
        end
    end
    return xform::Form{D,R,Poly{P,D,T}}
end
function barycentric2cartesian(λbasis::Basis{P,D1,R,T}) where {P,D1,R,T}
    D1::Int
    R::Int
    @assert 1 <= D1
    D = D1 - 1
    @assert 0 <= R <= D
    return Basis{P,D,R,T}(barycentric2cartesian.(λbasis.forms))
end

end
