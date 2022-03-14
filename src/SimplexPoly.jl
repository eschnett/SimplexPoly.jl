module SimplexPoly

using ComputedFieldTypes
using DifferentialForms
using LinearAlgebra
using SparseArrays
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
# Term{D,T}(powers::SVector{D,Int}, coeff::T) where {D,T} = Term{D,T,Pow{powers, coeff}
Term{D}(powers::SVector{D,Int}, coeff::T) where {D,T} = Term{D,T}(powers, coeff)
Term(powers::SVector{D,Int}, coeff::T) where {D,T} = Term{D,T}(powers, coeff)
function Base.convert(::Type{Term{D,T}}, term::Term{D}) where {D,T}
    return map(x -> convert(T, x), term)
end

function Base.show(io::IO, ::MIME"text/plain", x::Term{D,T}) where {D,T}
    skiptype = get(io, :typeinfo, Any) <: Term{D,T}
    if !skiptype
        print(io, "Term(")
    end
    print(io, x.coeff, " * ", x.powers)
    if !skiptype
        print(io, ")")
    end
    return nothing
end

function Base.:(==)(x::Term{D}, y::Term{D}) where {D}
    return x.powers == y.powers && x.coeff == y.coeff
end

function Base.isequal(x::Term, y::Term)
    return isequal((x.powers, x.coeff), (y.powers, y.coeff))
end
Base.isless(x::Term, y::Term) = isless((x.powers, x.coeff), (y.powers, y.coeff))
Base.hash(x::Term, h::UInt) = hash((x.powers, x.coeff), hash(0x11325cd9, h))
export compare
function compare(x::Term, y::Term)
    # Note reversed signs; we want (1,0) < (0,1)
    x.powers > y.powers && return -1
    x.powers < y.powers && return +1
    return 0
end

function Base.map(f, x::Term{D}, ys::Term{D}...) where {D}
    @assert all(x.powers == y.powers for y in ys)
    return Term{D}(x.powers, map(f, x.coeff, (y.coeff for y in ys)...))
end

function Base.zero(::Type{Term{D,T}}) where {D,T}
    return Term{D,T}(zero(SVector{D,Int}), zero(T))
end
Base.zero(::Term{D,T}) where {D,T} = zero(Term{D,T})
Base.iszero(x::Term) = iszero(x.coeff)

function Base.one(::Type{Term{D,T}}) where {D,T}
    return Term{D,T}(zero(SVector{D,Int}), one(T))
end
Base.one(::Term{D,T}) where {D,T} = one(Term{D,T})
Base.isone(x::Term) = iszero(x.powers) && isone(x.coeff)

Base.:+(x::Term) = map(+, x)
Base.:-(x::Term) = map(-, x)
Base.conj(x::Term) = map(conj, x)

Base.:+(x::Term{D}, y::Term{D}) where {D} = map(+, x, y)
Base.:-(x::Term{D}, y::Term{D}) where {D} = map(-, x, y)
Base.:*(a, x::Term) = map(c -> a * c, x)
Base.:*(x::Term, a) = map(c -> c * a, x)
Base.:\(a, x::Term) = map(c -> a \ c, x)
Base.:/(x::Term, a) = map(c -> c / a, x)
Base.div(x::Term, a) = map(c -> div(c, a), x)
Base.mod(x::Term, a) = map(c -> mod(c, a), x)

function Base.:*(x::Term{D}, y::Term{D}) where {D}
    return Term{D}(x.powers + y.powers, x.coeff * y.coeff)
end
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
function deriv(term::Term{D}, dir::Int) where {D}
    D::Int
    @assert D >= 0
    @assert 1 <= dir <= D
    p = term.powers[dir]
    p == 0 && return zero(term)
    return Term{D}(Base.setindex(term.powers, p - 1, dir), p * term.coeff)
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
function koszul(term::Term{D}, dir::Int) where {D}
    D::Int
    @assert D >= 0
    @assert 1 <= dir <= D
    p = term.powers[dir]
    return Term{D}(Base.setindex(term.powers, p + 1, dir), term.coeff)
end

export integral
function integral(term::Term{D,T}) where {D,T}
    R = T <: Union{Integer,Rational} ? Rational{BigInt} : T <: AbstractFloat ? T : Nothing
    D == 0 && return R(term.coeff)
    # See <https://math.stackexchange.com/questions/207073/definite-integral-over-a-simplex>
    # We set ν₀ = 1 since the respective term is absent
    return (term.coeff * (prod(factorial(big(p)) for p in term.powers) // factorial(big(sum(term.powers) + length(term.powers)))))::R
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
Poly{D}(terms::Vector{Term{D,T}}) where {D,T} = Poly{D,T}(terms)
Poly(terms::Vector{Term{D,T}}) where {D,T} = Poly{D,T}(terms)
function Base.convert(::Type{Poly{D,T}}, poly::Poly{D}) where {D,T}
    return map(x -> convert(T, x), poly)
end

function combine(terms::Vector{Term{D,T}}) where {D,T}
    terms = sort(terms; by=(t -> t.powers), rev=true)
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

function Base.show(io::IO, mime::MIME"text/plain", p::Poly{D,T}) where {D,T}
    skiptype = get(io, :typeinfo, Any) <: Poly{D,T}
    if !skiptype
        print(io, "Poly")
    end
    # print(io, "[")
    print(io, "(")
    for (i, term) in enumerate(p.terms)
        # i > 1 && print(io, ", ")
        i > 1 && print(io, " + ")
        show(IOContext(io, :compact => true, :typeinfo => Term{D,T}), mime, term)
    end
    # print(io, "]")
    print(io, ")")
    return nothing
end

Base.:(==)(p::Poly{D}, q::Poly{D}) where {D} = p.terms == q.terms

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

function Base.map(f, p::Poly{D,T}, q::Poly{D,U}) where {D,T,U}
    R = typeof(f(zero(T), zero(U)))
    terms = Term{D,R}[]
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
    return Poly{D,R}(terms)
end

function Base.reduce(op, p::Poly; init=Base._InitialValue())
    return reduce(op, p.terms; init=init)
end

function Base.mapreduce(f, op, p::Poly; init=Base._InitialValue())
    return reduce(op, f(p.terms); init=init)
end

function Base.mapreduce(f, op, p::Poly{D}, q::Poly{D}; init) where {D}
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
function Forms.unit(::Type{Poly{D,T}}, dir::Int, coeff=one(T)) where {D,T}
    D::Int
    @assert D >= 0
    @assert 1 <= dir <= D
    term = Term{D,T}(SVector{D,Int}(d == dir for d in 1:D), coeff)
    return Poly{D,T}([term])
end

Base.:+(p::Poly) = map(+, p)
Base.:-(p::Poly) = map(-, p)
Base.conj(p::Poly) = map(conj, p)

Base.:+(p::Poly{D}, q::Poly{D}) where {D} = map(+, p, q)
Base.:-(p::Poly{D}, q::Poly{D}) where {D} = map(-, p, q)
Base.:*(a::Number, p::Poly) = map(t -> a * t, p)
Base.:*(p::Poly, a::Number) = map(t -> t * a, p)
Base.:\(a::Number, p::Poly) = map(t -> a \ t, p)
Base.:/(p::Poly, a::Number) = map(t -> t / a, p)
Base.div(p::Poly, a::Number) = map(t -> div(t, a), p)
Base.mod(p::Poly, a::Number) = map(t -> mod(t, a), p)

function Base.:*(p::Poly{D}, q::Poly{D}) where {D}
    return Poly{D}([t * u for t in p.terms for u in q.terms])
end
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

function deriv(poly::Poly, dir::Int)
    return Poly(map(t -> deriv(t, dir), poly.terms))
end

function koszul(poly::Poly, dir::Int)
    return Poly(map(t -> koszul(t, dir), poly.terms))
end

function integral(poly::Poly{D,T}) where {D,T}
    R = T <: Union{Integer,Rational} ? Rational{BigInt} : T <: AbstractFloat ? T : Nothing
    isempty(poly.terms) && return zero(R)
    return sum(integral(term) for term in poly.terms)::R
end

# function LinearAlgebra.dot(p::Poly{D,T}, q::Poly{D,T}) where {D,T}
#     return integral(conj(p) * q)
# end

function LinearAlgebra.dot(p::Poly{D,T}, q::Poly{D,U}) where {D,T,U}
    init = zero(T) ⋅ zero(U)
    return mapreduce(⋅, +, p, q; init=init)::typeof(init)
end

################################################################################

# Note: PolySpace is unused

export PolySpace
struct PolySpace{D,T}
    polys::Vector{Poly{D,T}}
    function PolySpace{D,T}(polys::Vector{Poly{D,T}}) where {D,T}
        D::Int
        @assert D >= 0
        polys = combine(polys)
        return new{D,T}(polys)
    end
end

function combine(polys::Vector{Poly{D,T}}) where {D,T}
    # TODO: This is wrong -- need to combine polynomials that are
    # multiples of each other
    polys = sort(polys; rev=true)
    unique!(polys)
    filter!(p -> !iszero(p), polys)
    return polys
end

PolySpace{D,T}() where {D,T} = PolySpace{D,T}(Poly{D,T}[])

function Base.show(io::IO, mime::MIME"text/plain", ps::PolySpace{D,T}) where {D,T}
    skiptype = get(io, :typeinfo, Any) <: PolySpace{D,T}
    if !skiptype
        print(io, "PolySpace")
    end
    print(io, "[")
    for (i, poly) in enumerate(ps.polys)
        i > 1 && print(io, ", ")
        show(IOContext(io, :compact => true, :typeinfo => Poly{D,T}), mime, poly)
    end
    print(io, "]")
    return nothing
end

function Base.:(==)(ps::PolySpace{D}, qs::PolySpace{D}) where {D}
    return ps.polys == qs.polys
end

Base.isequal(ps::PolySpace, qs::PolySpace) = isequal(ps.polys, qs.polys)
Base.isless(ps::PolySpace, qs::PolySpace) = isless(ps.polys, qs.polys)
Base.hash(ps::PolySpace, h::UInt) = hash(ps.polys, hash(0xa3e4cfbf, h))

Base.map(f, ps::PolySpace{D,T}) where {D,T} = PolySpace{D,T}(map(f, ps.polys))

Base.isempty(ps::PolySpace) = isempty(ps.polys)
Base.length(ps::PolySpace) = length(ps.polys)
function Base.issubset(ps::PolySpace{D,T}, qs::PolySpace{D,T}) where {D,T}
    return issubset(Set(ps.polys), Set(qs.polys))
end
function Base.union(ps::PolySpace{D,T}, qs::PolySpace{D,T}) where {D,T}
    return PolySpace{D,T}([ps.polys; qs.polys])
end
function Base.setdiff(ps::PolySpace{D,T}, qs::PolySpace{D,T}) where {D,T}
    return PolySpace{D,T}(collect(setdiff(Set(ps.polys), Set(qs.polys))))
end

Base.zero(::Type{PolySpace{D,T}}) where {D,T} = PolySpace{D,T}()
Base.iszero(ps::PolySpace) = isempty(ps)
# Base.:+(ps::PolySpace) = ps
# Base.:+(ps::PolySpace{D,T}, qs::PolySpace{D,T}) where {D,T} = union(ps, qs)
# Base.:*(a, ps::PolySpace) = map(p -> a * p, ps)
# Base.:*(ps::PolySpace, a) = map(p -> p * a, ps)
# Base.:\(a, ps::PolySpace) = map(p -> a \ p, ps)
# Base.:/(ps::PolySpace, a) = map(p -> p / a, ps)

deriv(ps::PolySpace, dir::Int) = map(p -> deriv(p, dir), ps)
koszul(ps::PolySpace, dir::Int) = map(p -> koszul(p, dir), ps)

################################################################################

function deriv(f::Form{D,R,Poly{D,T}}) where {D,R,T}
    D::Int
    R::Int
    @assert 0 <= R < D
    r = zero(Form{D,R + 1,Poly{D,T}})
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
    @assert 0 < R <= D
    r = zero(Form{D,R - 1,Poly{D,T}})
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
                rbits = Base.setindex(bits, false, dir)
                rn = Forms.bit2lin(Val(D), Val(R - 1), rbits)
                r = Base.setindex(r, r[rn] + s * koszul(f[n], dir), rn)
            end
        end
    end
    return r::Form{D,R - 1,Poly{D,T}}
end

function integral(f::Form{D,R,Poly{D,T}}) where {D,R,T}
    D::Int
    R::Int
    @assert 0 <= R <= D
    U = T <: Union{Integer,Rational} ? Rational{BigInt} : T <: AbstractFloat ? T : Nothing
    return map(integral, f)::Form{D,R,U}
end

# function LinearAlgebra.dot(f::Form{D,R,Poly{D,T}},
#                            g::Form{D,R,Poly{D,T}}) where {D,R,T}
#     U = T <: Union{Integer,Rational} ? Rational{BigInt} :
#         T <: AbstractFloat ? T : Nothing
#     r = zero(U)
#     for (fi, gi) in zip(f.elts, g.elts)
#         r += (fi ⋅ gi)::U
#     end
#     return r
# end

function LinearAlgebra.dot(f::Form{D,R,Poly{D,T1}}, g::Form{D,R,Poly{D,T2}}) where {D,R,T1,T2}
    T = typeof(zero(T1) * zero(T2))
    r = zero(T)
    for (fp, gp) in zip(f.elts, g.elts)
        r += (fp ⋅ gp)::T
    end
    return r
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

function poly2vec(poly::Poly{D,T}, maxp::Int) where {D,T}
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
function form2vec(form::Form{D,R,Poly{D,T}}, maxp::Int) where {D,R,T}
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
function forms2mat(forms::Vector{<:Form{D,R,Poly{D,T}}}, maxp::Int) where {D,R,T}
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
@computed struct Basis{D,R,T}
    forms::Vector{fulltype(Form{D,R,Poly{D,T}})}
    function Basis{D,R,T}(forms::Vector{<:Form{D,R,Poly{D,T}}}) where {D,R,T}
        D::Int
        R::Int
        @assert 0 ≤ R ≤ D
        forms = combine(forms)
        return new{D,R,T}(forms)
    end
end

export compare
function compare(x::Form{D,R,<:Poly{D}}, y::Form{D,R,<:Poly{D}}) where {D,R}
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

function combine(forms::Vector{<:Form{D,R,Poly{D,T}}}) where {D,R,T}
    forms = sort(forms; rev=true)
    unique!(forms)
    filter!(form -> !iszero(form), forms)
    map!(normalize, forms, forms)
    sort!(forms; lt=(x, y) -> compare(x, y) < 0)
    unique!(forms)
    # Gram-Schmidt creates polynomials with many and large coefficients
    forms = gram_schmidt(forms)

    # basisforms = fulltype(Form{D,R,Poly{D,T}})[]
    # for form in forms
    #     if !is_in_span(form, basisforms)
    #         push!(basisforms, form)
    #     end
    # end
    # forms = basisforms

    return forms
end

function normalize(form::Form{D,R,Poly{D,T}}) where {D,R,T<:Integer}
    coeffs = T[]
    for poly in form
        poly::Poly{D,T}
        for term in poly.terms
            term::Term{D,T}
            push!(coeffs, term.coeff)
        end
    end
    isempty(coeffs) && return form
    q = sign(coeffs[1]) * gcd(coeffs)
    return map(x -> x ÷ q, form)
end

function normalize(form::Form{D,R,Poly{D,T}}) where {D,R,T<:Rational}
    coeffs = T[]
    for poly in form
        poly::Poly{D,T}
        for term in poly.terms
            term::Term{D,T}
            push!(coeffs, term.coeff)
        end
    end
    isempty(coeffs) && return zero(Form{D,R,Poly{D,Int}})
    q = T(lcm(denominator.(coeffs)))
    return normalize(Form{D,R,Poly{D,Int}}(q * form))
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
                    @show :assert oform form oform ⋅ form
                end
                @assert iszero(oform ⋅ form)
            end
            push!(ortho_forms, form)
        end
    end
    return ortho_forms
end

# function Basis{D,R,T}() where {D,R,T}
#     return Basis{D,R,T}([])
# end

function Base.show(io::IO, mime::MIME"text/plain", b::Basis{D,R,T}) where {D,R,T}
    skiptype = get(io, :typeinfo, Any) <: Basis{D,R,T}
    if !skiptype
        print(io, "Basis{$D,$R,$T}")
    end
    print(io, "[")
    for (i, form) in enumerate(b.forms)
        i > 1 && print(io, ", ")
        show(IOContext(io, :compact => true, :typeinfo => Form{D,R,Poly{D,T}}), mime, form)
    end
    print(io, "]")
    return nothing
end

function Base.:(==)(b1::Basis{D,R}, b2::Basis{D,R}) where {D,R}
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

function Base.zero(::Type{<:Basis{D,R,T}}) where {D,R,T}
    return Basis{D,R,T}(fulltype(Form{D,R,Poly{D,T}})[])
end
Base.zero(::Basis{D,R,T}) where {D,R,T} = zero(Basis{D,R,T})
Base.iszero(basis::Basis) = isempty(basis.forms)

function Forms.unit(form::Form{D,R,Poly{D,T}}) where {D,R,T}
    return Basis{D,R,T}([form])
end

export tensorsum, ⊕
function Forms.tensorsum(basis1::Basis{D,R,T}, basis2::Basis{D,R,T}) where {D,R,T}
    return Basis{D,R,T}([basis1.forms; basis2.forms])
end

function tensordiff end
const ⊖ = tensordiff
export tensordiff, ⊖
function tensordiff(basis1::Basis{D,R,T}, basis2::Basis{D,R,T}) where {D,R,T}
    forms2 = Set(basis2.forms)
    forms = fulltype(Form{D,R,Poly{D,T}})[]
    for form in basis1.forms
        if form ∉ forms2
            push!(forms, form)
        end
    end
    return Basis{D,R,T}(forms)
end

Base.isempty(basis::Basis) = isempty(basis.forms)
Base.length(basis::Basis) = length(basis.forms)

function is_in_span(form::Form{D,R,Poly{D,T}}, forms::Vector{<:Form{D,R,Poly{D,T}}}) where {D,R,T}
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

function Base.in(form::Form{D,R,Poly{D,T}}, basis::Basis{D,R,T}) where {D,R,T}
    return is_in_span(form, basis.forms)
end

function Base.issubset(basis1::Basis{D,R}, basis2::Basis{D,R}) where {D,R}
    isempty(basis1.forms) && return true
    isempty(basis2.forms) && return false
    return all(f in basis2 for f in basis1.forms)
end

# Unused
function make_basis1(::Type{<:Basis{D,R,T}}, p::Int, cond) where {D,R,T}
    D::Int
    R::Int
    @assert 0 <= R <= D
    @assert p >= -1             # -1 is the empty basis
    polys = Poly{D,T}[]
    for i0 in CartesianIndex(ntuple(d -> 0, D)):CartesianIndex(ntuple(d -> p, D))
        i = SVector{D,Int}(i0.I)
        if cond(sum(i))
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

    return Basis{D,R,T}(forms)
end

# Unused
function make_basis2(::Type{<:Basis{D,R,T}}, p::Int, cond) where {D,R,T}
    D::Int
    R::Int
    @assert 0 <= R <= D
    @assert p >= -1             # -1 is the empty basis
    N = length(Form{D,R})
    forms = fulltype(Form{D,R,Poly{D,T}})[zero(Form{D,R,Poly{D,T}})]
    for i0 in CartesianIndex(ntuple(d -> 0, D)):CartesianIndex(ntuple(d -> p, D))
        i = SVector{D,Int}(i0.I)
        if cond(sum(i))
            poly = Poly{D,T}([Term{D,T}(i, one(T))])
            form = Form{D,R,Poly{D,T}}(ntuple(n -> poly, N))
            push!(forms, form)
        end
    end
    return Basis{D,R,T}(forms)
end

function make_basis(::Type{<:Basis{D,R,T}}, p::Int, cond) where {D,R,T}
    D::Int
    R::Int
    @assert 0 <= R <= D
    @assert p >= -1             # -1 is the empty basis
    polys = Poly{D,T}[]
    for i0 in CartesianIndex(ntuple(d -> 0, D)):CartesianIndex(ntuple(d -> p, D))
        i = SVector{D,Int}(i0.I)
        if cond(sum(i))
            push!(polys, Poly{D,T}([Term{D,T}(i, one(T))]))
        end
    end
    N = length(Form{D,R})
    forms = fulltype(Form{D,R,Poly{D,T}})[]
    for n in 1:N
        for poly in polys
            form = Form{D,R,Poly{D,T}}(ntuple(m -> m == n ? poly : zero(Poly{D,T}), N))
            push!(forms, form)
        end
    end
    return Basis{D,R,T}(forms)
end

export full_basis
function full_basis(::Type{<:Basis{D,R,T}}, p::Int) where {D,R,T}
    return make_basis(Basis{D,R,T}, p, <=(p))
end

export homogeneous_basis
function homogeneous_basis(::Type{<:Basis{D,R,T}}, p::Int) where {D,R,T}
    return make_basis(Basis{D,R,T}, p, ==(p))
end

function deriv(basis::Basis{D,R,T}) where {D,R,T}
    D::Int
    R::Int
    @assert 0 <= R < D
    dforms = fulltype(Form{D,R + 1,Poly{D,T}})[]
    for form in basis.forms
        dform = deriv(form)
        !iszero(dform) && push!(dforms, dform)
    end
    return Basis{D,R + 1,T}(dforms)
end

function koszul(basis::Basis{D,R,T}) where {D,R,T}
    D::Int
    R::Int
    @assert 0 < R <= D
    κforms = fulltype(Form{D,R - 1,Poly{D,T}})[]
    for form in basis.forms
        κform = koszul(form)
        !iszero(κform) && push!(κforms, κform)
    end
    return Basis{D,R - 1,T}(κforms)
end

################################################################################

# function bernstein(::Type{<:Basis{D,0,T}}, p::Int) where {D,T}
#     D::Int
#     R = 0
#     @assert 0 <= R <= D
#     @assert p >= 0
#     polys = fulltype(Form{D,R,Poly{D,T}})[]
#     for i0 in
#         CartesianIndex(ntuple(d -> 0, D)):CartesianIndex(ntuple(d -> p, D))
#         i = SVector{D,Int}(i.I)
#         if sum(i) == p
#             poly = Poly{D,T}([Term{D,T}(i, one(T))])
#             push!(polys, Form{D,R}((poly,)))
#         end
#     end
#     return Basis{D,T}(polys)
# end
# 
# function bernstein(::Type{<:Basis{D,R,T}}, p::Int) where {D,R,T}
#     D::Int
#     R::Int
#     @assert 0 <= R <= D
#     @assert p >= 0
#     polys = Poly{D,T}[]
#     for i0 in
#         CartesianIndex(ntuple(d -> 0, D)):CartesianIndex(ntuple(d -> p, D))
#         i = SVector{D,Int}(i.I)
#         if sum(i) == p
#             push!(polys, Poly{D,T}([Term{D,T}(i, one(T))]))
#         end
#     end
#     return Basis{D,T}(polys)
# end

################################################################################

export polynomial_complex
"""
polynomial_complex

Pᵣ Λᵏ

The spaces with constant r form a complex. NOTE: This differs from Arnold's definition!
"""
function polynomial_complex(::Val{D}, ::Type{T}, p::Int) where {D,T}
    D::Int
    @assert 0 <= D
    @assert p >= -1

    cc = Dict{Int,Basis{D,R,T} where R}()

    for R in 0:D
        cc[R] = full_basis(Basis{D,R,T}, max(-1, p - R))::Basis{D,R,T}
    end

    return cc
end

export trimmed_polynomial_complex
"""
trimmed_polynomial_complex

Pᵣ⁻ Λᵏ

The spaces with constant r form a complex.
"""
function trimmed_polynomial_complex(::Val{D}, ::Type{T}, p::Int) where {D,T}
    D::Int
    @assert 0 <= D
    @assert p >= -1

    cc = Dict{Int,Basis{D,R,T} where R}()

    for R in 0:D
        if R == 0
            b = full_basis(Basis{D,R,T}, p)
        else
            b = full_basis(Basis{D,R,T}, max(-1, p - 1))
            if R + 1 <= D
                b = b ⊕ koszul(homogeneous_basis(Basis{D,R + 1,T}, p - 1))
            end
            # if R - 1 >= 0
            #     b = b ⊕ deriv(homogenous_basis(Basis{D,R - 1,T}, p + 1))
            # end
        end
        cc[R] = b::Basis{D,R,T}
    end

    return cc
end

export extended_trimmed_polynomial_complex
"""
extended_trimmed_polynomial_complex

This is experimental; note that this is not actually a complex.
"""
function extended_trimmed_polynomial_complex(::Val{D}, ::Type{T}, p::Int) where {D,T}
    D::Int
    @assert 0 <= D
    @assert p >= -1

    cc = Dict{Int,Basis{D,R,T} where R}()

    for R in 0:D
        if R == 0
            b = full_basis(Basis{D,R,T}, p)
        else
            b = full_basis(Basis{D,R,T}, max(-1, p - 1))
            if R + 1 <= D
                b = b ⊕ koszul(homogeneous_basis(Basis{D,R + 1,T}, p - 1))
            end
            if R - 1 >= 0
                b = b ⊕ deriv(homogeneous_basis(Basis{D,R - 1,T}, p + 1))
            end
        end
        cc[R] = b::Basis{D,R,T}
    end

    return cc
end

################################################################################

export whitney
"""
whitney

Calculate Whitney forms
"""
function whitney(::Val{D1}, ::Type{T}, inds::SVector{N,Int}) where {D1,T,N}
    D1::Int
    N::Int
    @assert 1 <= D1
    @assert 1 <= N <= D1
    R = N - 1
    @assert all(1 <= inds[n] <= D1 for n in 1:N)
    @assert all(inds[n] < inds[n + 1] for n in 1:(N - 1))
    ϕ = zero(Form{D1,R,Poly{D1,T}})
    q = factorial(N - 1)
    for n in 1:N
        inds′ = deleteat(inds, n)
        f = unit(Form{D1,R,T}, inds′)
        p = q * bitsign(n - 1) * unit(Poly{D1,T}, inds[n])
        ϕ += p * f
    end
    return ϕ::Form{D1,R,Poly{D1,T}}
end
function whitney(::Val{D1}, ::Type{T}, inds::NTuple{N,Int}) where {D1,T,N}
    return whitney(Val(D1), T, SVector{N,Int}(inds))
end
function whitney(::Type{Basis{D1,R,T}}) where {D1,R,T}
    D1::Int
    R::Int
    @assert 1 <= D1
    @assert 0 <= R <= D1 - 1
    nelts = binomial(D1, R + 1)
    forms = [whitney(Val(D1), T, Forms.lin2lst(Val(D1), Val(R + 1), n)) for n in 1:nelts]
    return Basis{D1,R,T}(forms)
end
whitney(d::Val, inds::SVector) = whitney(d, Int, inds)
whitney(d::Val, inds::NTuple) = whitney(d, Int, inds)

export whitney_support
"""
whitney_support

Determine on what part of a simplex (vertex, edge, face, ...) a
particular Whitney forms should live.
"""
function whitney_support(::Val{D1}, inds::SVector{N,Int}) where {D1,N}
    return inds
end

################################################################################

export barycentric2cartesian
function barycentric2cartesian(λterm::Term{D1,T}) where {D1,T}
    D1::Int
    @assert 1 <= D1
    D = D1 - 1
    # This assumes a standard simplex:
    #     λᵢ = xᵢ
    #     λₙ = 1 - Σᵢ λᵢ
    λn = one(Poly{D,T})
    if D > 0
        λn -= sum(unit(Poly{D,T}, d) for d in 1:D)
    end
    xpoly = Poly{D,T}([Term{D,T}(deleteat(λterm.powers, D1), λterm.coeff)])
    xpoly *= λn^λterm.powers[D1]
    return xpoly::Poly{D,T}
end
function barycentric2cartesian(λpoly::Poly{D1,T}) where {D1,T}
    D1::Int
    @assert 1 <= D1
    D = D1 - 1
    isempty(λpoly.terms) && return zero(Poly{D,T})
    xpoly = sum(barycentric2cartesian(λterm) for λterm in λpoly.terms)
    return xpoly::Poly{D,T}
end
function barycentric2cartesian(λform::Form{D1,R,Poly{D1,T}}) where {D1,R,T}
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
        xform = Form{D,R,Poly{D,T}}((xpoly,))
    else
        dλn = -sum(unit(Form{D,1,Poly{D,T}}, d) for d in 1:D)
        xform = zero(Form{D,R,Poly{D,T}})
        for (λlin, λpoly) in enumerate(λform)
            λbits = Forms.lin2bit(Val(D1), Val(R), λlin)
            xbits = deleteat(λbits, D1)
            if λbits[D1]
                xlin = Forms.bit2lin(Val(D), Val(R - 1), xbits)
                xunit = unit(Form{D,R - 1,Poly{D,T}}, xlin) ∧ dλn
            else
                xlin = Forms.bit2lin(Val(D), Val(R), xbits)
                xunit = unit(Form{D,R,Poly{D,T}}, xlin)
            end
            xpoly = barycentric2cartesian(λpoly)
            xform += xpoly * xunit
        end
    end
    return xform::Form{D,R,Poly{D,T}}
end
function barycentric2cartesian(λbasis::Basis{D1,R,T}) where {D1,R,T}
    D1::Int
    R::Int
    @assert 1 <= D1
    D = D1 - 1
    @assert 0 <= R <= D
    return Basis{D,R,T}(barycentric2cartesian.(λbasis.forms))
end

end
