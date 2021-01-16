using ComputedFieldTypes
using DifferentialForms
using Random
using SimplexPoly
using StaticArrays
using Test

# Ignore a statement
macro DISABLED(expr)
    return quote end
end

const Dmax = 6

# Set reproducible random number seed
Random.seed!(0)

################################################################################

# Random terms
const maxrandpower = 5
function Base.rand(rng::AbstractRNG,
                   ::Random.SamplerType{Term{D,T}}) where {D,T}
    return Term{D,T}(SVector{D,Int}(rand(rng, 0:maxrandpower, D)),
                     T(rand(rng, -10:10)))
end

@testset "Terms as vector space D=$D" for D in 0:Dmax
    T = Int

    for iter in 1:100
        x = rand(Term{D,T})
        n = map(c -> zero(T), x)
        y = map(c -> T(rand(-10:10)), x)
        z = map(c -> T(rand(-10:10)), x)
        a = T(rand(-10:10))
        b = T(rand(-10:10))

        @test n == n

        f(x) = 2 * x
        g(x) = x + 1
        h(x) = x^2
        @test map(identity, x) == x
        @test map(f ∘ identity, x) == map(f, x)
        @test map(identity ∘ f, x) == map(f, x)
        @test map(h ∘ g, map(f, x)) == map(h, map(g ∘ f, x))

        @test n + x == x
        @test x + n == x
        @test x + y == y + x
        @test x + (y + z) == (x + y) + z
        @test +x == x

        @test x + (-x) == n
        @test -(-x) == x
        @test -n == n
        @test x - y == x + (-y)

        @test zero(T) * x == n
        @test one(T) * x == x
        @test (-one(T)) * x == -x
        @test a * x == x * a
        @test (a * b) * x == a * (b * x)
        @test (a + b) * x == a * x + b * x
        @test a * (x + y) == a * x + a * y

        # Cannot test this with integers
        # if a != zero(T)
        #     @test (a * x) / a == x
        #     @test a \ (a * x) == x
        # end
    end
end

@testset "Terms as ring D=$D" for D in 0:Dmax
    T = Int

    for iter in 1:100
        n = zero(Term{D,T})
        e = one(Term{D,T})
        x = rand(Term{D,T})
        x′ = map(c -> T(rand(-10:10)), x)
        y = rand(Term{D,T})
        z = rand(Term{D,T})
        a = T(rand(-10:10))
        b = T(rand(-10:10))
        i = rand(0:10)
        j = rand(0:10)

        @test n == n
        @test e == e
        @test e != n
        @test iszero(n)
        @test isone(e)

        # This is not true for terms, only for polynomials
        # @test n * x == n
        # @test x * n == n
        @test iszero(n * x)
        @test iszero(x * n)
        @test e * x == x
        @test x * e == x
        @test x * y == y * x
        @test x * (y * z) == (x * y) * z
        @test (a * x) * y == a * (x * y)
        @test x * (y * a) == (x * y) * a
        @test (x + x′) * y == x * y + x′ * y

        @test x^0 == e
        @test x^1 == x
        @test x^(i + j) == x^i * x^j
    end
end

@testset "Derivatives of terms D=$D" for D in 1:Dmax
    T = Int

    for iter in 1:100
        x = rand(Term{D,T})
        n = map(c -> zero(T), x)
        y = map(c -> T(rand(-10:10)), x)
        a = T(rand(-10:10))

        dir = rand(1:D)

        dx = deriv(x, dir)
        if x.powers[dir] == 0
            @test dx.coeff == 0
        else
            @test dx.powers[dir] == x.powers[dir] - 1
        end

        @test deriv(x + y, dir) == deriv(x, dir) + deriv(y, dir)
        @test deriv(a * x, dir) == a * deriv(x, dir)

        # @test deriv(x * y, dir) == deriv(x, dir) * y + x * deriv(y, dir)

        κx = koszul(x, dir)
        @test κx.powers[dir] == x.powers[dir] + 1

        @test koszul(x + y, dir) == koszul(x, dir) + koszul(y, dir)
        @test koszul(a * x, dir) == a * koszul(x, dir)
    end
end

################################################################################

# Random polynomials
function Base.rand(rng::AbstractRNG,
                   ::Random.SamplerType{Poly{D,T}}) where {D,T}
    n = rand(0:5)
    return Poly{D,T}(rand(rng, Term{D,T}, n))
end

@testset "Polynomials D=$D" for D in 0:Dmax
    T = Int

    for iter in 1:100
        n = zero(Poly{D,T})
        e = one(Poly{D,T})
        x = rand(Poly{D,T})
        y = rand(Poly{D,T})
        z = rand(Poly{D,T})
        a = T(rand(-10:10))
        b = T(rand(-10:10))
        i = rand(0:5)
        j = rand(0:5)

        # Ensure terms are nonzero
        for i in 1:length(x.terms)
            @test !iszero(x.terms[i])
        end

        # Ensure terms are sorted and unique
        for i in 2:length(x.terms)
            @test x.terms[i - 1].powers > x.terms[i].powers
            @test compare(x.terms[i - 1], x.terms[i]) < 0
        end

        @test n == n
        @test e == e
        @test e != n

        f(x) = 2 * x
        g(x) = x + 1
        h(x) = x^2
        @test map(identity, x) == x
        @test map(f ∘ identity, x) == map(f, x)
        @test map(identity ∘ f, x) == map(f, x)
        @test map(h ∘ g, map(f, x)) == map(h, map(g ∘ f, x))

        @test n + x == x
        @test x + n == x
        @test x + y == y + x
        @test x + (y + z) == (x + y) + z
        @test +x == x

        @test x + (-x) == n
        @test -(-x) == x
        @test -n == n
        @test x - y == x + (-y)

        @test zero(T) * x == n
        @test one(T) * x == x
        @test (-one(T)) * x == -x
        @test a * x == x * a
        @test (a * b) * x == a * (b * x)
        @test (a + b) * x == a * x + b * x
        @test a * (x + y) == a * x + a * y

        # Cannot test this with integers
        # if a != zero(T)
        #     @test (a * x) / a == x
        #     @test a \ (a * x) == x
        # end

        @test n * x == n
        @test x * n == n
        @test e * x == x
        @test x * e == x
        @test x * y == y * x
        @test x * (y * z) == (x * y) * z
        @test (a * x) * y == a * (x * y)
        @test x * (y * a) == (x * y) * a
        @test (x + y) * z == x * z + y * z

        @test x^0 == e
        @test x^1 == x
        @test x^(i + j) == x^i * x^j

        if D > 0
            dir = rand(1:D)

            dx = deriv(x, dir)
            @test deriv(n, dir) == n
            @test deriv(e, dir) == n
            @test deriv(unit(Poly{D,T}, dir), dir) == e
            @test deriv(x + y, dir) == deriv(x, dir) + deriv(y, dir)
            @test deriv(a * x, dir) == a * deriv(x, dir)

            @test deriv(x * y, dir) == deriv(x, dir) * y + x * deriv(y, dir)

            κx = koszul(x, dir)
            @test koszul(n, dir) == n
            @test koszul(e, dir) == unit(Poly{D,T}, dir)
            @test koszul(x + y, dir) == koszul(x, dir) + koszul(y, dir)
            @test koszul(a * x, dir) == a * koszul(x, dir)

            # This is not true
            # @test koszul(x * y, dir) == koszul(x, dir) * y + x * koszul(y, dir)
        end
    end
end

################################################################################

# Random polynomial spaces
function Base.rand(rng::AbstractRNG,
                   ::Random.SamplerType{PolySpace{D,T}}) where {D,T}
    n = rand(0:5)
    ps = rand(rng, Poly{D,T}, n)
    ps = filter(p -> !iszero(p), ps)
    return PolySpace{D,T}(ps)
end

@testset "Polynomial spaces D=$D" for D in 0:Dmax
    T = Int

    for iter in 1:100
        ps = rand(PolySpace{D,T})
        qs = rand(PolySpace{D,T})
        rs = rand(PolySpace{D,T})
        ns = PolySpace{D,T}()
        a = rand(-10:10)
        b = rand(-10:10)

        #TODO
        # # Ensure polynomials are nonzero
        # for i in 1:length(ps.polys)
        #     @test !iszero(ps.polys[i])
        # end

        #TODO
        # # Ensure polynomials are sorted and unique
        # for i in 2:length(ps.polys)
        #     @show i ps ps.polys[i - 1] ps.polys[i]
        #     @test compare(ps.polys[i - 1], ps.polys[i]) < 0
        # end

        @test ps == ps
        @test qs == qs
        @test (issubset(ps, qs) && issubset(qs, ps)) == (ps == qs)

        @test isempty(ns)
        @test isempty(ps) == (length(ps) == 0)

        @test union(ps, ns) == ps
        @test union(ps, qs) == union(qs, ps)
        @test union(union(ps, qs), rs) == union(ps, union(qs, rs))

        @test setdiff(ps, ns) == ps
        @test setdiff(setdiff(ps, qs), rs) == setdiff(ps, union(qs, rs))

        @test iszero(ns)
        # @test ps + ns == ps
        # @test ps + qs == qs + ps
        # @test (ps + qs) + rs == ps + (qs + rs)

        for dir in 1:D
            @test deriv(union(ps, qs), dir) ==
                  union(deriv(ps, dir), deriv(qs, dir))
            @test koszul(union(ps, qs), dir) ==
                  union(koszul(ps, dir), koszul(qs, dir))
        end
    end
end

################################################################################
################################################################################
################################################################################

@testset "Polynomial forms as vector space D=$D R=$R" for D in 0:Dmax, R in 0:D
    T = Int

    for iter in 1:100
        n = zero(Form{D,R,Poly{D,T}})
        x = rand(Form{D,R,Poly{D,T}})
        x′ = rand(Form{D,R,Poly{D,T}})
        y = rand(Form{D,R,Poly{D,T}})
        z = rand(Form{D,R,Poly{D,T}})
        a = T(rand(-10:10))
        b = T(rand(-10:10))
        pn = zero(Poly{D,T})
        pe = one(Poly{D,T})
        p = rand(Poly{D,T})
        q = rand(Poly{D,T})

        @test n == n

        @test n + x == x
        @test x + n == x
        @test x + y == y + x
        @test x + (y + z) == (x + y) + z
        @test +x == x

        @test x + (-x) == n
        @test -(-x) == x
        @test -n == n
        @test x - y == x + (-y)

        @test zero(T) * x == n
        @test one(T) * x == x
        @test (-one(T)) * x == -x
        @test a * x == x * a
        @test (a * b) * x == a * (b * x)
        @test (a + b) * x == a * x + b * x
        @test a * (x + y) == a * x + a * y

        @test pn * x == n
        @test pe * x == x
        @test (-pe) * x == -x
        @test p * x == x * p
        @test (p * q) * x == p * (q * x)
        @test (p + q) * x == p * x + q * x
        @test p * (x + y) == p * x + p * y
    end
end

@testset "Polynomial forms as ring D=$D" for D in 0:Dmax
    T = Int

    for iter in 1:100
        Rn = rand(0:D)
        Rx = rand(0:D)
        Ry = rand(0:D)
        Rz = rand(0:D)
        n = zero(Form{D,Rn,Poly{D,T}})
        e = one(Form{D,0,Poly{D,T}})
        x = rand(Form{D,Rx,Poly{D,T}})
        x′ = rand(Form{D,Rx,Poly{D,T}})
        y = rand(Form{D,Ry,Poly{D,T}})
        z = rand(Form{D,Rz,Poly{D,T}})
        a = T(rand(-10:10))
        p = rand(Poly{D,T})

        @test e == e
        @test iszero(n)
        @test isone(e)
        @test !iszero(e)
        @test !isone(n)

        if Rn + Rx <= D
            @test iszero(n ∧ x)
            @test iszero(x ∧ n)
        end
        @test e ∧ x == x
        @test x ∧ e == x
        if Rx + Ry <= D
            @test x ∧ y == bitsign(Rx * Ry) * (y ∧ x)
        end
        if Rx + Ry + Rz <= D
            @test x ∧ (y ∧ z) == (x ∧ y) ∧ z
        end
        if Rx + Ry <= D
            @test (a * x) ∧ y == a * (x ∧ y)
            @test x ∧ (y * a) == (x ∧ y) * a
            @test (p * x) ∧ y == p * (x ∧ y)
            @test x ∧ (y * p) == (x ∧ y) * p
            @test (x + x′) ∧ y == x ∧ y + x′ ∧ y
        end
    end
end

@testset "Derivatives of polynomial forms D=$D" for D in 0:Dmax
    T = Int

    for iter in 1:100
        Rx = rand(0:D)
        Ry = rand(0:D)
        x = rand(Form{D,Rx,Poly{D,T}})
        x2 = rand(Form{D,Rx,Poly{D,T}})
        y = rand(Form{D,Ry,Poly{D,T}})
        a = T(rand(-10:10))

        if Rx <= D - 1
            @test deriv(x + x2) == deriv(x) + deriv(x2)
            @test deriv(a * x) == a * deriv(x)
        end
        if Rx + Ry <= D - 1
            @test deriv(x ∧ y) == deriv(x) ∧ y + bitsign(Rx) * x ∧ deriv(y)
        end
        if Rx <= D - 2
            @test iszero(deriv(deriv(x)))
        end

        if 1 <= Rx
            @test koszul(x + x2) == koszul(x) + koszul(x2)
            @test koszul(a * x) == a * koszul(x)
        end
        if 1 <= min(Rx, Ry) && Rx + Ry <= D
            @test koszul(x ∧ y) == koszul(x) ∧ y + bitsign(Rx) * x ∧ koszul(y)
        end
        if 2 <= Rx
            @test iszero(koszul(koszul(x)))
        end

        if D >= 1
            Rz = rand(0:D)
            p = rand(0:5)
            function mkterm()
                dims = rand(1:D, p)
                powers = zeros(D)
                for d in dims
                    powers[d] += 1
                end
                return Term{D,T}(SVector{D,Int}(powers), T(rand(-10:10)))
            end
            function mkpoly()
                n = rand(0:5)
                return Poly{D,T}(Term{D,T}[mkterm() for i in 1:n])
            end
            function mkform()
                N = binomial(Val(D), Val(Rz))
                return Form{D,Rz}(ntuple(_ -> mkpoly(), N))
            end
            z = mkform()

            # Douglas Arnold, Richard Falk, Ragnar Winther, "Finite
            # element exterior calculus, homological techniques, and
            # applications", Acta Numerica 15, 1-155 (2006),
            # DOI:10.1017/S0962492906210018, eqn. (3.9)
            if 1 <= Rz <= D - 1
                # This holds only for homogenous polynomial forms,
                # i.e. forms where all polynomials have the same
                # degree `p`
                @test deriv(koszul(z)) + koszul(deriv(z)) == (Rz + p) * z
            end
        end
    end
end

################################################################################

@testset "Convert polynomial forms to vectors D=$D R=$R" for D in
                                                             0:min(5, Dmax),
R in 0:D

    T = Int

    for iter in 1:100
        x = rand(Form{D,R,Poly{D,T}})
        y = rand(Form{D,R,Poly{D,T}})
        n = zero(Form{D,R,Poly{D,T}})
        a = rand(-10:10)

        @test maxpower(n) == 0
        p = max(maxpower(x), maxpower(y))

        @test iszero(form2vec(n, p))
        @test form2vec(n + x, p) == form2vec(x, p)
        @test form2vec(x + y, p) == form2vec(x, p) + form2vec(y, p)
        @test form2vec(a * x, p) == a * form2vec(x, p)

        nrows = length(form2vec(n, p))
        @test forms2mat([x], p) == reshape(form2vec(x, p), nrows, 1)
    end

    for iter in 1:10
        ncols = rand(1:5)
        xs = [rand(Form{D,R,Poly{D,T}}) for j in 1:ncols]
        ys = [rand(Form{D,R,Poly{D,T}}) for j in 1:ncols]
        ns = [zero(Form{D,R,Poly{D,T}}) for j in 1:ncols]
        a = rand(-10:10)

        @test maxpower(ns) == 0
        p = max(maxpower(xs), maxpower(ys))

        @test iszero(forms2mat(ns, p))
        @test forms2mat(ns + xs, p) == forms2mat(xs, p)
        @test forms2mat(xs + ys, p) == forms2mat(xs, p) + forms2mat(ys, p)
        @test forms2mat(a * xs, p) == a * forms2mat(xs, p)
    end
end

################################################################################

@testset "Bases D=$D R=$R p=$p" for D in 0:Dmax, R in 0:D, p in 0:4
    T = Int

    # feec-icerm-lecture3, page 23
    # n = D
    # r = p
    # k = R

    fb = full_basis(Basis{D,R,T}, p)
    @test length(fb.forms) == binomial(p + D, p) * binomial(D, R)

    hb = homogeneous_basis(Basis{D,R,T}, p)
    @test length(hb.forms) == binomial(p + D - 1, p) * binomial(D, R)
end

################################################################################

@testset "Complexes D=$D p=$p" for D in 0:Dmax, p in 1:max(2, 5 - D)
    T = Int

    pc = polynomial_complex(Val(D), Int, p)
    tpc = trimmed_polynomial_complex(Val(D), Int, p)
    epc = extended_trimmed_polynomial_complex(Val(D), Int, p)

    @test Set(keys(pc)) == Set(0:D)
    @test Set(keys(tpc)) == Set(0:D)
    @test Set(keys(epc)) == Set(0:D)

    for R in 0:D
        @test pc[R] isa Basis{D,R,T}
        @test tpc[R] isa Basis{D,R,T}
        @test epc[R] isa Basis{D,R,T}

        fb = full_basis(Basis{D,R,T}, p)
        fb1 = full_basis(Basis{D,R,T}, p - 1)

        if R != 0 && R != D
            @test issubset(fb1, tpc[R])
            @test fb1 != tpc[R]
        end
        if R == 0
            @test tpc[R] == pc[R]
        elseif R == D
            @test tpc[R] == fb1
        else
            @test issubset(tpc[R], fb)
            @test tpc[R] != fb
        end
        @test issubset(tpc[R], epc[R])
        if R == 0
            @test epc[R] == pc[R]
        elseif R == D
            # This does not seem to hold
            # @test epc[R] == fb1
        else
            @test issubset(epc[R], fb)
            @test epc[R] != pc[R]
        end

        if R < D
            @test issubset(deriv(pc[R]), pc[R + 1])
            @test issubset(deriv(tpc[R]), tpc[R + 1])
            # This does not seem to hold
            # This also means that epc is not actually complex!
            # @test issubset(deriv(epc[R]), epc[R + 1])
        end
        if R > 0
            @test issubset(koszul(pc[R]), pc[R - 1])
            @test issubset(koszul(tpc[R]), tpc[R - 1])
            # This does not seem to hold
            # @test issubset(koszul(epc[R]), epc[R - 1])
        end
    end
end

################################################################################

@testset "Whitney forms D=$D R=$R" for D in 0:Dmax, R in 0:D
    T = Int

    ϕ = whitney(Basis{D + 1,R,Int})
    @test length(ϕ.forms) == binomial(D + 1, R + 1)

    for n in 1:length(ϕ.forms)
        ϕₙ = ϕ.forms[n]
        ϕ′ = Basis{D + 1,R,T}([ϕ.forms[1:(n - 1)]; ϕ.forms[(n + 1):end]])
        @test !(ϕₙ in ϕ′)
    end
end

@testset "Convert from barycentric to Cartesian coordinates D=$D R=$R" for D in
                                                                           0:Dmax,
R in 0:D

    T = Int

    ϕ = whitney(Basis{D + 1,R,Int})
    @test length(ϕ.forms) == binomial(D + 1, R + 1)
    ϕx = barycentric2cartesian(ϕ)

    for n in 1:length(ϕx.forms)
        ϕₙ = ϕx.forms[n]
        ϕx′ = Basis{D,R,T}([ϕx.forms[1:(n - 1)]; ϕx.forms[(n + 1):end]])
        @test !(ϕₙ in ϕx′)
    end
end

@testset "Compare Whitney basis to trimmed p=1 basis D=$D R=$R" for D in 0:Dmax,
R in 0:D

    T = Int
    p = 1                       # Whitney basis is equivalent to p=1

    ϕλ = whitney(Basis{D + 1,R,Int})
    ϕ = barycentric2cartesian(ϕλ)

    tpc = trimmed_polynomial_complex(Val(D), Int, p)
    tpcR = tpc[R]

    @test ϕ == tpcR
end
