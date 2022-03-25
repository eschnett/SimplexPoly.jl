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

maximum0(::SVector{0,T}) where {T} = zero(T)
maximum0(xs::SVector{N,T}) where {N,T} = maximum(xs)

################################################################################

@time begin
    ptypes() = [Pow, Exp]

    types(::Type{Pow}) = [Int64, Complex{Int64}]
    types(::Type{Exp}) = [Complex{Int64}]
    # bigtypes(::Type{Pow}) = [Int128, Complex{Int128}]
    # bigtypes(::Type{Exp}) = [Complex{Int128}]
    bigtypes(::Type{Pow}) = [BigInt, Complex{BigInt}]
    bigtypes(::Type{Exp}) = [Complex{BigInt}]

    # # Random terms
    # const maxrandpower = 5
    # function Base.rand(rng::AbstractRNG, ::Random.SamplerType{Term{P,D,T}}) where {P,D,T}
    #     return Term{P,D,T}(SVector{D,Int}(rand(rng, 0:maxrandpower, D)), T(rand(rng, -10:10)))
    # end

    @time @testset "Terms as vector space P=$P D=$D T=$T" for P in ptypes(), D in 0:Dmax, T in types(P)
        for iter in 1:100
            x = rand(Term{P,D,T})
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

    @time @testset "Terms as ring P=$P D=$D T=$T" for P in ptypes(), D in 0:Dmax, T in types(P)
        for iter in 1:100
            n = zero(Term{P,D,T})
            e = one(Term{P,D,T})
            x = rand(Term{P,D,T})
            x′ = map(c -> T(rand(-10:10)), x)
            y = rand(Term{P,D,T})
            z = rand(Term{P,D,T})
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

    @time @testset "Derivatives of terms P=Pow D=$D T=$T" for D in 1:Dmax, T in types(Pow)
        e = one(Term{Pow,D,T})
        for dir in 1:D
            @test iszero(deriv(e, dir))
        end
        @test integral(e) == 1 // factorial(D)

        for iter in 1:100
            x = rand(Term{Pow,D,T})
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

            d1 = deriv(x * y, dir)
            d2 = deriv(x, dir) * y + x * deriv(y, dir)
            @test iszero(d1) && iszero(d2) || d1 == d2

            κx = koszul(x, dir)
            @test κx.powers[dir] == x.powers[dir] + 1

            @test koszul(x + y, dir) == koszul(x, dir) + koszul(y, dir)
            @test koszul(a * x, dir) == a * koszul(x, dir)

            @test integral(n) == 0
            @test integral(x + y) == integral(x) + integral(y)
            @test integral(a * x) == a * integral(x)
        end
    end

    @time @testset "Derivatives of terms P=Exp D=$D T=$T" for D in 1:Dmax, T in types(Exp)
        e = one(Term{Exp,D,T})
        for dir in 1:D
            @test iszero(deriv(e, dir))
        end
        # @test integral(e) == 1 // factorial(D)

        for iter in 1:100
            x = rand(Term{Exp,D,T})
            n = map(c -> zero(T), x)
            y = map(c -> T(rand(-10:10)), x)
            a = T(rand(-10:10))

            dir = rand(1:D)

            dx = deriv(x, dir)
            if x.powers[dir] == 0
                @test dx.coeff == 0
            else
                @test dx.powers[dir] == x.powers[dir]
            end

            @test deriv(x + y, dir) == deriv(x, dir) + deriv(y, dir)
            @test deriv(a * x, dir) == a * deriv(x, dir)

            d1 = deriv(x * y, dir)
            d2 = deriv(x, dir) * y + x * deriv(y, dir)
            @test iszero(d1) && iszero(d2) || d1 == d2

            # κx = koszul(x, dir)
            # @test κx.powers[dir] == x.powers[dir]
            # 
            # @test koszul(x + y, dir) == koszul(x, dir) + koszul(y, dir)
            # @test koszul(a * x, dir) == a * koszul(x, dir)

            # @test integral(n) == 0
            # @test integral(x + y) == integral(x) + integral(y)
            # @test integral(a * x) == a * integral(x)
        end
    end

    ################################################################################

    # # Random polynomials
    # function Base.rand(rng::AbstractRNG, ::Random.SamplerType{Poly{P,D,T}}) where {P,D,T}
    #     n = rand(0:5)
    #     return Poly{P,D,T}(rand(rng, Term{P,D,T}, n))
    # end

    @time @testset "Polynomials P=$P D=$D T=$T" for P in ptypes(), D in 0:Dmax, T in types(P)
        e = one(Poly{P,D,T})
        for dir in 1:D
            @test iszero(deriv(e, dir))
        end
        if P ≡ Pow
            @test integral(e) == 1 // factorial(D)
        end

        for iter in 1:100
            n = zero(Poly{P,D,T})
            e = one(Poly{P,D,T})
            x = rand(Poly{P,D,T})
            y = rand(Poly{P,D,T})
            z = rand(Poly{P,D,T})
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

            @test n ⋅ x == 0
            x2 = x ⋅ x
            if iszero(x)
                @test x2 == 0
            else
                @test isreal(x2)
                @test real(x2) > 0
            end
            @test x ⋅ y == conj(y ⋅ x)
            @test a * (x ⋅ y) == x ⋅ (a * y)
            @test x ⋅ (y + z) == x ⋅ y + x ⋅ z

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
                if P ≡ Pow
                    @test deriv(unit(Poly{P,D,T}, dir), dir) == e
                elseif P ≡ Exp
                    @test deriv(unit(Poly{P,D,T}, dir), dir) == im * unit(Poly{P,D,T}, dir)
                end
                @test deriv(x + y, dir) == deriv(x, dir) + deriv(y, dir)
                @test deriv(a * x, dir) == a * deriv(x, dir)

                @test deriv(x * y, dir) == deriv(x, dir) * y + x * deriv(y, dir)

                if P ≡ Pow
                    κx = koszul(x, dir)
                    @test koszul(n, dir) == n
                    @test koszul(e, dir) == unit(Poly{P,D,T}, dir)
                    @test koszul(x + y, dir) == koszul(x, dir) + koszul(y, dir)
                    @test koszul(a * x, dir) == a * koszul(x, dir)

                    # This is not true
                    # @test koszul(x * y, dir) == koszul(x, dir) * y + x * koszul(y, dir)
                end

                if P ≡ Pow
                    @test integral(n) == 0
                    @test integral(e) == 1 // factorial(D)
                    @test integral(x + y) == integral(x) + integral(y)
                    @test integral(a * x) == a * integral(x)
                end
            end
        end
    end

    ################################################################################

    # # Random polynomial spaces
    # function Base.rand(rng::AbstractRNG, ::Random.SamplerType{PolySpace{P,D,T}}) where {P,D,T}
    #     n = rand(0:5)
    #     ps = rand(rng, Poly{P,D,T}, n)
    #     ps = filter(p -> !iszero(p), ps)
    #     return PolySpace{P,D,T}(ps)
    # end

    @time @testset "Polynomial spaces P=$P D=$D T=$T" for P in [Pow], D in 0:Dmax, T in types(P)
        for iter in 1:100
            ps = rand(PolySpace{P,D,T})
            qs = rand(PolySpace{P,D,T})
            rs = rand(PolySpace{P,D,T})
            ns = PolySpace{P,D,T}()
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
                @test deriv(union(ps, qs), dir) == union(deriv(ps, dir), deriv(qs, dir))
                @test koszul(union(ps, qs), dir) == union(koszul(ps, dir), koszul(qs, dir))
            end
        end
    end

    ################################################################################
    ################################################################################
    ################################################################################

    @time @testset "Polynomial forms as vector space D=$D R=$R P=$P T=$T" for D in 0:Dmax, R in 0:D, P in ptypes(), T in types(P)
        for iter in 1:100
            n = zero(Form{D,R,Poly{P,D,T}})
            x = rand(Form{D,R,Poly{P,D,T}})
            x′ = rand(Form{D,R,Poly{P,D,T}})
            y = rand(Form{D,R,Poly{P,D,T}})
            z = rand(Form{D,R,Poly{P,D,T}})
            a = T(rand(-10:10))
            b = T(rand(-10:10))
            pn = zero(Poly{P,D,T})
            pe = one(Poly{P,D,T})
            p = rand(Poly{P,D,T})
            q = rand(Poly{P,D,T})

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

    @time @testset "Polynomial forms as ring P=$P D=$D T=$T" for P in ptypes(), D in 0:Dmax, T in types(P)
        for iter in 1:100
            Rn = rand(0:D)
            Rx = rand(0:D)
            Ry = rand(0:D)
            Rz = rand(0:D)
            n = zero(Form{D,Rn,Poly{P,D,T}})
            e = one(Form{D,0,Poly{P,D,T}})
            x = rand(Form{D,Rx,Poly{P,D,T}})
            x′ = rand(Form{D,Rx,Poly{P,D,T}})
            y = rand(Form{D,Ry,Poly{P,D,T}})
            z = rand(Form{D,Rz,Poly{P,D,T}})
            a = T(rand(-10:10))
            p = rand(Poly{P,D,T})

            @test e == e
            @test e != n
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

    @time @testset "Derivatives of polynomial forms P=$P D=$D T=$T" for P in ptypes(), D in 0:Dmax, T in types(P)
        for iter in 1:100
            Rx = rand(0:D)
            Ry = rand(0:D)
            n = zero(Form{D,Rx,Poly{P,D,T}})
            x = rand(Form{D,Rx,Poly{P,D,T}})
            x2 = rand(Form{D,Rx,Poly{P,D,T}})
            y = rand(Form{D,Ry,Poly{P,D,T}})
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

            if P ≡ Pow
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
                    return Term{P,D,T}(SVector{D,Int}(powers), T(rand(-10:10)))
                end
                function mkpoly()
                    q = rand(0:5)
                    return Poly{P,D,T}(Term{P,D,T}[mkterm() for i in 1:q])
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
                    if P ≡ Pow
                        @test deriv(koszul(z)) + koszul(deriv(z)) == (Rz + p) * z
                    end
                end
            end

            if P ≡ Pow
                @test iszero(integral(n)::Form{D,Rx})
                if Rx <= D - 1
                    @test integral(x + x2) == integral(x) + integral(x2)
                    @test integral(a * x) == a * integral(x)
                end
            end

            @test iszero(n ⋅ x)

            xx = x ⋅ x
            @test isreal(xx)
            if iszero(x)
                @test xx == 0
            else
                @test real(xx) > 0
            end
            @test x ⋅ x2 isa Number
            @test x ⋅ (a * x2) == a * (x ⋅ x2)
            @test x ⋅ x2 == conj(x2 ⋅ x)
        end
    end

    ################################################################################

    @time @testset "Polynomial tensor forms as vector space D=$D R1=$R1 R2=$R2 P=$P T=$T" for D in 0:min(Dmax, 4),
                                                                                              R1 in 0:D,
                                                                                              R2 in 0:D,
                                                                                              P in ptypes(),
                                                                                              T in types(P)

        for iter in 1:100
            n = zero(TensorForm{D,R1,R2,Poly{P,D,T}})
            x = rand(TensorForm{D,R1,R2,Poly{P,D,T}})
            x′ = rand(TensorForm{D,R1,R2,Poly{P,D,T}})
            y = rand(TensorForm{D,R1,R2,Poly{P,D,T}})
            z = rand(TensorForm{D,R1,R2,Poly{P,D,T}})
            a = T(rand(-10:10))
            b = T(rand(-10:10))
            pn = zero(Poly{P,D,T})
            pe = one(Poly{P,D,T})
            p = rand(Poly{P,D,T})
            q = rand(Poly{P,D,T})

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

    @time @testset "Polynomial tensor forms as ring P=$P D=$D T=$T" for P in ptypes(), D in 0:min(Dmax, 4), T in types(P)
        for iter in 1:100
            R1n = rand(0:D)
            R1x = rand(0:D)
            R1y = rand(0:D)
            R1z = rand(0:D)
            R2n = rand(0:D)
            R2x = rand(0:D)
            R2y = rand(0:D)
            R2z = rand(0:D)
            n = zero(TensorForm{D,R1n,R2n,Poly{P,D,T}})
            e = one(TensorForm{D,0,0,Poly{P,D,T}})
            x = rand(TensorForm{D,R1x,R2x,Poly{P,D,T}})
            x′ = rand(TensorForm{D,R1x,R2x,Poly{P,D,T}})
            y = rand(TensorForm{D,R1y,R2y,Poly{P,D,T}})
            z = rand(TensorForm{D,R1z,R2z,Poly{P,D,T}})
            a = T(rand(-10:10))
            p = rand(Poly{P,D,T})

            @test e == e
            @test e != n
            @test iszero(n)
            @test isone(e)
            @test !iszero(e)
            # @test !isone(n)

            if R1n + R1x ≤ D && R2n + R2x ≤ D
                @test iszero(n ∧ x)
                @test iszero(x ∧ n)
            end
            @test e ∧ x == x
            @test x ∧ e == x
            if R1x + R1y ≤ D && R2x + R2y ≤ D
                @test x ∧ y == bitsign(R1x * R1y) * bitsign(R2x * R2y) * (y ∧ x)
            end
            if R1x + R1y + R1z ≤ D && R2x + R2y + R2z ≤ D
                @test x ∧ (y ∧ z) == (x ∧ y) ∧ z
            end
            if R1x + R1y ≤ D && R2x + R2y ≤ D
                @test (a * x) ∧ y == a * (x ∧ y)
                @test x ∧ (y * a) == (x ∧ y) * a
                @test (p * x) ∧ y == p * (x ∧ y)
                @test x ∧ (y * p) == (x ∧ y) * p
                @test (x + x′) ∧ y == x ∧ y + x′ ∧ y
            end
        end
    end

    @time @testset "Derivatives of polynomial tensor forms P=$P D=$D T=$T" for P in ptypes(), D in 0:min(Dmax, 4), T in types(P)
        for iter in 1:100
            R1x = rand(0:D)
            R1y = rand(0:D)
            R2x = rand(0:D)
            R2y = rand(0:D)
            n = zero(TensorForm{D,R1x,R2x,Poly{P,D,T}})
            x = rand(TensorForm{D,R1x,R2x,Poly{P,D,T}})
            x2 = rand(TensorForm{D,R1x,R2x,Poly{P,D,T}})
            y = rand(TensorForm{D,R1y,R2y,Poly{P,D,T}})
            a = T(rand(-10:10))

            if R1x ≤ D - 1
                @test deriv1(x + x2) == deriv1(x) + deriv1(x2)
                @test deriv1(a * x) == a * deriv1(x)
            end
            if R1x ≤ D && R1y ≤ D && R1x + R1y ≤ D - 1 && R2x + R2y ≤ D
                @test deriv1(x ∧ y) == deriv1(x) ∧ y + bitsign(R1x) * x ∧ deriv1(y)
            end
            if R1x ≤ D - 2
                @test iszero(deriv1(deriv1(x)))
            end

            if R2x ≤ D - 1
                @test deriv2(x + x2) == deriv2(x) + deriv2(x2)
                @test deriv2(a * x) == a * deriv2(x)
            end
            if R1x + R1y ≤ D && R2x + R2y ≤ D - 1
                @test deriv2(x ∧ y) == deriv2(x) ∧ y + bitsign(R2x) * x ∧ deriv2(y)
            end
            if R2x ≤ D - 2
                @test iszero(deriv2(deriv2(x)))
            end

            if P ≡ Pow
                if 1 ≤ R1x
                    @test koszul1(x + x2) == koszul1(x) + koszul1(x2)
                    @test koszul1(a * x) == a * koszul1(x)
                end
                if 1 ≤ R1x && 1 ≤ R1y && R1x + R1y ≤ D && R1x + R1y ≤ D - 1 && R2x + R2y ≤ D
                    @test koszul1(x ∧ y) == koszul1(x) ∧ y + bitsign(R1x) * x ∧ koszul1(y)
                end
                if 2 ≤ R1x
                    @test iszero(koszul1(koszul1(x)))
                end

                if 1 ≤ R2x
                    @test koszul2(x + x2) == koszul2(x) + koszul2(x2)
                    @test koszul2(a * x) == a * koszul2(x)
                end
                if R1x + R1y ≤ D && 1 ≤ R2x && 1 ≤ R2y && R2x + R2y ≤ D && R2x + R2y ≤ D - 1
                    @test koszul2(x ∧ y) == koszul2(x) ∧ y + bitsign(R2x) * x ∧ koszul2(y)
                end
                if 2 ≤ R2x
                    @test iszero(koszul2(koszul2(x)))
                end
            end

            # if D >= 1
            #     Rz = rand(0:D)
            #     p = rand(0:5)
            #     function mkterm()
            #         dims = rand(1:D, p)
            #         powers = zeros(D)
            #         for d in dims
            #             powers[d] += 1
            #         end
            #         return Term{P,D,T}(SVector{D,Int}(powers), T(rand(-10:10)))
            #     end
            #     function mkpoly()
            #         q = rand(0:5)
            #         return Poly{P,D,T}(Term{P,D,T}[mkterm() for i in 1:q])
            #     end
            #     function mkform()
            #         N = binomial(Val(D), Val(Rz))
            #         return Form{D,Rz}(ntuple(_ -> mkpoly(), N))
            #     end
            #     z = mkform()
            # 
            #     # Douglas Arnold, Richard Falk, Ragnar Winther, "Finite
            #     # element exterior calculus, homological techniques, and
            #     # applications", Acta Numerica 15, 1-155 (2006),
            #     # DOI:10.1017/S0962492906210018, eqn. (3.9)
            #     if 1 <= Rz <= D - 1
            #         # This holds only for homogenous polynomial forms,
            #         # i.e. forms where all polynomials have the same
            #         # degree `p`
            #         if P ≡ Pow
            #             @test deriv(koszul(z)) + koszul(deriv(z)) == (Rz + p) * z
            #         end
            #     end
            # end

            # if P ≡ Pow
            #     @test iszero(integral(n)::Form{D,Rx})
            #     if Rx <= D - 1
            #         @test integral(x + x2) == integral(x) + integral(x2)
            #         @test integral(a * x) == a * integral(x)
            #     end
            # end

            @test iszero(n ⋅ x)

            xx = x ⋅ x
            xx::T
            # @test isreal(xx)
            @test xx == conj(xx)
            if iszero(x)
                @test xx == 0
            else
                @test real(xx) > 0
            end
            @test x ⋅ x2 isa Number
            @test x ⋅ (a * x2) == a * (x ⋅ x2)
            @test x ⋅ x2 == conj(x2 ⋅ x)
        end
    end

    ################################################################################

    @time @testset "Convert polynomial forms to vectors D=$D R=$R P=$P T=$T" for D in 0:min(5, Dmax),
                                                                                 R in 0:D,
                                                                                 P in ptypes(),
                                                                                 T in types(P)

        for iter in 1:100
            x = rand(Form{D,R,Poly{P,D,T}})
            y = rand(Form{D,R,Poly{P,D,T}})
            n = zero(Form{D,R,Poly{P,D,T}})
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
            xs = [rand(Form{D,R,Poly{P,D,T}}) for j in 1:ncols]
            ys = [rand(Form{D,R,Poly{P,D,T}}) for j in 1:ncols]
            ns = [zero(Form{D,R,Poly{P,D,T}}) for j in 1:ncols]
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

    @time @testset "Convert polynomial tensor forms to vectors D=$D R1=$R1 R2=$R2 P=$P T=$T" for D in 0:min(4, Dmax),
                                                                                                 R1 in 0:D,
                                                                                                 R2 in 0:D,
                                                                                                 P in ptypes(),
                                                                                                 T in types(P)

        for iter in 1:100
            x = rand(TensorForm{D,R1,R2,Poly{P,D,T}})
            y = rand(TensorForm{D,R1,R2,Poly{P,D,T}})
            n = zero(TensorForm{D,R1,R2,Poly{P,D,T}})
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
            xs = [rand(TensorForm{D,R1,R2,Poly{P,D,T}}) for j in 1:ncols]
            ys = [rand(TensorForm{D,R1,R2,Poly{P,D,T}}) for j in 1:ncols]
            ns = [zero(TensorForm{D,R1,R2,Poly{P,D,T}}) for j in 1:ncols]
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

    function test_basis(basis::Basis{P,D,R,T}) where {P,D,R,T}
        prebasis = zero(basis)
        for f in basis.forms
            @test f ∉ prebasis
            push!(prebasis.forms, f)
        end
    end

    @time @testset "Bases P=$P D=$D R=$R T=$T p=$p" for P in ptypes(),
                                                        D in 0:min(Dmax, 3),
                                                        R in 0:D,
                                                        T in types(P),
                                                        p in 0:max(2, 5 - D)

        z = zero(Basis{P,D,R,T})
        b1 = rand(Basis{P,D,R,T})
        b2 = rand(Basis{P,D,R,T})
        b3 = rand(Basis{P,D,R,T})

        @test isempty(z)

        @test z ⊆ z
        @test z ⊆ b1
        @test b1 ⊆ b1

        @test z ∩ b1 == z
        @test b1 ∩ b1 == b1
        @test b1 ∩ b2 == b2 ∩ b1
        @test (b1 ∩ b2) ∩ b3 == b1 ∩ (b2 ∩ b3)
        @test ∩(b1, b2, b3) == (b1 ∩ b2) ∩ b3
        @test (b1 ∩ b2) ⊆ b1
        @test (b1 ∩ b2) ⊆ b2

        @test z ∪ b1 == b1
        @test b1 ∪ b1 == b1
        @test b1 ∪ b2 == b2 ∪ b1
        @test (b1 ∪ b2) ∪ b3 == b1 ∪ (b2 ∪ b3)
        @test ∪(b1, b2, b3) == (b1 ∪ b2) ∪ b3
        @test b1 ⊆ (b1 ∪ b2)
        @test b2 ⊆ (b1 ∪ b2)

        @test b1 ∪ (b2 ∩ b3) == (b1 ∪ b2) ∩ (b1 ∪ b3)
        @test b1 ∩ (b2 ∪ b3) == (b1 ∩ b2) ∪ (b1 ∩ b3)

        @test setdiff(b1, z) == b1
        @test setdiff(z, b1) == z
        @test setdiff(b1, b1) == z
        @test setdiff(setdiff(b1, b2), b3) == setdiff(setdiff(b1, b3), b2)
        @test setdiff(b1, setdiff(b2, b1)) == b1
        @test setdiff(b1, b2, b3) == setdiff(setdiff(b1, b2), b3)
        @test setdiff(b1, b2) ⊆ b1
        @test isdisjoint(setdiff(b1, b2), b2)

        @test setdiff(b1, b2 ∪ b3) == setdiff(b1, b2, b3)
        @test setdiff(b1 ∩ b2, b3) == setdiff(b1, b3) ∩ setdiff(b2, b3)
        @test setdiff(b1 ∪ b2, b3) == setdiff(b1, b3) ∪ setdiff(b2, b3)
    end

    @time @testset "Special bases P=$P D=$D R=$R T=$T p=$p" for P in ptypes(),
                                                                D in 0:Dmax,
                                                                R in 0:D,
                                                                T in types(P),
                                                                p in 0:max(2, 5 - D)
        # feec-icerm-lecture3, page 23
        # n = D
        # r = p
        # k = R

        fb = full_basis(Basis{P,D,R,T}, p)
        @test length(fb.forms) == binomial(p + D, p) * binomial(D, R)
        if D < 6
            test_basis(fb)
        end

        hb = homogeneous_basis(Basis{P,D,R,T}, p)
        @test length(hb.forms) == binomial(p + D - 1, p) * binomial(D, R)
        if D < 6
            test_basis(hb)
        end
    end

    ################################################################################

    function test_basis(basis::TensorBasis{P,D,R1,R2,T}) where {P,D,R1,R2,T}
        prebasis = zero(basis)
        for f in basis.forms
            @test f ∉ prebasis
            push!(prebasis.forms, f)
        end
    end

    @time @testset "Tensor bases P=$P D=$D R1=$R1 R2=$R2 T=$T p=$p" for P in [Pow],
                                                                        D in 0:min(Dmax, 3),
                                                                        R1 in 0:D,
                                                                        R2 in 0:D,
                                                                        T in bigtypes(P),
                                                                        p in 0:max(2, 5 - D)

        z = zero(TensorBasis{P,D,R1,R2,T})
        b1 = rand(TensorBasis{P,D,R1,R2,T})
        b2 = rand(TensorBasis{P,D,R1,R2,T})
        b3 = rand(TensorBasis{P,D,R1,R2,T})

        @test isempty(z)

        @test z ⊆ z
        @test z ⊆ b1
        @test b1 ⊆ b1

        @test z ∩ b1 == z
        @test b1 ∩ b1 == b1
        @test b1 ∩ b2 == b2 ∩ b1
        @test (b1 ∩ b2) ∩ b3 == b1 ∩ (b2 ∩ b3)
        @test ∩(b1, b2, b3) == (b1 ∩ b2) ∩ b3
        @test (b1 ∩ b2) ⊆ b1
        @test (b1 ∩ b2) ⊆ b2

        @test z ∪ b1 == b1
        @test b1 ∪ b1 == b1
        @test b1 ∪ b2 == b2 ∪ b1
        @test (b1 ∪ b2) ∪ b3 == b1 ∪ (b2 ∪ b3)
        @test ∪(b1, b2, b3) == (b1 ∪ b2) ∪ b3
        @test b1 ⊆ (b1 ∪ b2)
        @test b2 ⊆ (b1 ∪ b2)

        @test b1 ∪ (b2 ∩ b3) == (b1 ∪ b2) ∩ (b1 ∪ b3)
        @test b1 ∩ (b2 ∪ b3) == (b1 ∩ b2) ∪ (b1 ∩ b3)

        @test setdiff(b1, z) == b1
        @test setdiff(z, b1) == z
        @test setdiff(b1, b1) == z
        @test setdiff(setdiff(b1, b2), b3) == setdiff(setdiff(b1, b3), b2)
        @test setdiff(b1, setdiff(b2, b1)) == b1
        @test setdiff(b1, b2, b3) == setdiff(setdiff(b1, b2), b3)
        @test setdiff(b1, b2) ⊆ b1
        @test isdisjoint(setdiff(b1, b2), b2)

        @test setdiff(b1, b2 ∪ b3) == setdiff(b1, b2, b3)
        @test setdiff(b1 ∩ b2, b3) == setdiff(b1, b3) ∩ setdiff(b2, b3)
        @test setdiff(b1 ∪ b2, b3) == setdiff(b1, b3) ∪ setdiff(b2, b3)
    end

    @time @testset "Special tensor bases P=$P D=$D R1=$R1 R2=$R2 T=$T p=$p" for P in [Pow],
                                                                                D in 0:3,
                                                                                R1 in 0:D,
                                                                                R2 in 0:D,
                                                                                T in types(P),
                                                                                p in 0:max(2, 4 - D)
        # feec-icerm-lecture3, page 23
        # n = D
        # r = p
        # k = R

        fb = full_basis(TensorBasis{P,D,R1,R2,T}, p)
        @test length(fb.forms) == binomial(p + D, p) * binomial(D, R1) * binomial(D, R2)
        test_basis(fb)

        hb = homogeneous_basis(TensorBasis{P,D,R1,R2,T}, p)
        @test length(hb.forms) == binomial(p + D - 1, p) * binomial(D, R1) * binomial(D, R2)
        test_basis(hb)
    end

    ################################################################################

    function tpc_length(D::Int, p::Int, R::Int)
        @assert D ≥ 0 && p ≥ 1 && 0 ≤ R ≤ D
        return binomial(p + D, p + R) * binomial(p + R - 1, R)
    end
    function pc_length(D::Int, p::Int, R::Int)
        @assert D ≥ 0 && p ≥ 1 && 0 ≤ R ≤ D
        return binomial(p - R + D, p) * binomial(p, R)
    end

    @time @testset "Complexes P=$P D=$D T=$T p=$p" for P in [Pow], D in 0:min(Dmax, 4), T in types(P), p in 1:max(2, 4 - D)
        pc = polynomial_complex(P, Val(D), T, p)
        tpc = trimmed_polynomial_complex(P, Val(D), T, p)
        epc = extended_trimmed_polynomial_complex(P, Val(D), T, p)
        mpc = maximal_polynomial_complex(P, Val(D), T, p)
        mqpc = maximal_polynomial_complex(P, Val(D), T, p; pnorm=maximum0)

        @test Set(keys(pc)) == Set(0:D)
        @test Set(keys(tpc)) == Set(0:D)
        @test Set(keys(epc)) == Set(0:D)
        @test Set(keys(mpc)) == Set(0:D)
        @test Set(keys(mqpc)) == Set(0:D)

        for R in 0:D
            @test pc[R] isa Basis{P,D,R,T}
            @test tpc[R] isa Basis{P,D,R,T}
            @test epc[R] isa Basis{P,D,R,T}
            @test mpc[R] isa Basis{P,D,R,T}
            @test mqpc[R] isa Basis{P,D,R,T}

            test_basis(pc[R])
            test_basis(tpc[R])
            test_basis(epc[R])
            test_basis(mpc[R])
            test_basis(mqpc[R])

            @test length(pc[R]) == pc_length(D, p, R)
            @test length(tpc[R]) == tpc_length(D, p, R)

            fb = full_basis(Basis{P,D,R,T}, p)
            fb1 = full_basis(Basis{P,D,R,T}, p - 1)

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
                # This also means that epc is not actually a complex!
                # @test issubset(deriv(epc[R]), epc[R + 1])
                @test issubset(deriv(mpc[R]), mpc[R + 1])
                @test issubset(deriv(mqpc[R]), mqpc[R + 1])
            end
            # These properties do not hold
            # @test ⋆pc[R] == pc[D - R]
            # @test ⋆tpc[R] == tpc[D - R]
            # @test ⋆mpc[R] == mpc[D - R]
            # @test ⋆mqpc[R] == mqpc[D - R]
            if R > 0
                @test issubset(⋆deriv(⋆pc[R]), pc[R - 1])
                @test issubset(⋆deriv(⋆tpc[R]), tpc[R - 1])
                # This does not seem to hold
                # This also means that epc is not actually a complex!
                # @test issubset(deriv(epc[R]), epc[R + 1])
                @test issubset(⋆deriv(⋆mpc[R]), mpc[R - 1])
                # This property does not seem to hold
                # @test issubset(⋆deriv(⋆mqpc[R]), mqpc[R - 1])
            end
            if R > 0
                @test issubset(koszul(pc[R]), pc[R - 1])
                @test issubset(koszul(tpc[R]), tpc[R - 1])
                # This does not seem to hold
                # @test issubset(koszul(epc[R]), epc[R - 1])
                # @test issubset(koszul(mpc[R]), mpc[R - 1])
                # @test issubset(koszul(mqpc[R]), mqpc[R - 1])
            end
        end
    end

    ################################################################################

    # function pdc_length(D::Int, p::Int, R1::Int, R2::Int)
    #     @assert D ≥ 0 && p ≥ 1 && 0 ≤ R1 ≤ D && 0 ≤ R2 ≤ D
    #     return binomial(p - (R1 + R2) + D, p) * binomial(p, R1) * binomial(p, R2)
    # end

    @time @testset "Double complexes P=$P D=$D T=$T p=$p" for P in [Pow], D in 0:min(Dmax, 3), T in types(P), p in 1:max(2, 3 - D)
        pdc = polynomial_double_complex(P, Val(D), T, p)
        qpdc = quad_polynomial_double_complex(P, Val(D), T, p)
        # tpdc = trimmed_polynomial_double_complex(P, Val(D), T, p)
        mpdc = maximal_polynomial_double_complex(P, Val(D), T, p)
        mqpdc = maximal_polynomial_double_complex(P, Val(D), T, p)
        @test Set(keys(pdc)) == Set((d1, d2) for d1 in 0:D, d2 in 0:D)
        @test Set(keys(qpdc)) == Set((d1, d2) for d1 in 0:D, d2 in 0:D)
        # @test Set(keys(tpdc)) == Set((d1, d2) for d1 in 0:D, d2 in 0:D)
        @test Set(keys(mpdc)) == Set((d1, d2) for d1 in 0:D, d2 in 0:D)
        @test Set(keys(mqpdc)) == Set((d1, d2) for d1 in 0:D, d2 in 0:D)

        for R1 in 0:D, R2 in 0:D
            @test pdc[(R1, R2)] isa TensorBasis{P,D,R1,R2,T}
            @test qpdc[(R1, R2)] isa TensorBasis{P,D,R1,R2,T}
            # @test tpdc[(R1, R2)] isa TensorBasis{P,D,R1,R2,T}
            @test mpdc[(R1, R2)] isa TensorBasis{P,D,R1,R2,T}
            @test mqpdc[(R1, R2)] isa TensorBasis{P,D,R1,R2,T}
            test_basis(pdc[(R1, R2)])
            test_basis(qpdc[(R1, R2)])
            # test_basis(tpdc[(R1, R2)])
            test_basis(mpdc[(R1, R2)])
            test_basis(mqpdc[(R1, R2)])
            # @test length(pdc[(R1, R2)]) == pdc_length(D, p, R1, R2)

            if R1 < D
                @test issubset(deriv1(pdc[(R1, R2)]), pdc[(R1 + 1, R2)])
                # @test issubset(deriv1(qpdc[(R1, R2)]), qpdc[(R1 + 1, R2)])
                # @test issubset(deriv1(tpdc[(R1, R2)]), tpdc[(R1 + 1, R2)])
                @test issubset(deriv1(mpdc[(R1, R2)]), mpdc[(R1 + 1, R2)])
                @test issubset(deriv1(mqpdc[(R1, R2)]), mqpdc[(R1 + 1, R2)])
            end
            if R2 < D
                @test issubset(deriv2(pdc[(R1, R2)]), pdc[(R1, R2 + 1)])
                # @test issubset(deriv2(qpdc[(R1, R2)]), qpdc[(R1, R2 + 1)])
                # @test issubset(deriv2(tpdc[(R1, R2)]), tpdc[(R1, R2 + 1)])
                @test issubset(deriv2(mpdc[(R1, R2)]), mpdc[(R1, R2 + 1)])
                @test issubset(deriv2(mqpdc[(R1, R2)]), mqpdc[(R1, R2 + 1)])
            end
            if R1 > 0
                @test issubset(⋆deriv1(⋆pdc[(R1, R2)]), pdc[(R1 - 1, R2)])
            end
            if R2 > 0
                @test issubset(⋆deriv2(⋆pdc[(R1, R2)]), pdc[(R1, R2 - 1)])
            end
            # This can fail because `koszul[12]` do not generate the constant function
            if R1 > 0
                @test issubset(koszul1(pdc[(R1, R2)]), pdc[(R1 - 1, R2)])
                # @test issubset(koszul1(qpdc[(R1, R2)]), qpdc[(R1 - 1, R2)])
                # @test issubset(koszul1(tpdc[(R1, R2)]), tpdc[(R1 - 1, R2)])
                # @test issubset(koszul1(mpdc[(R1, R2)]), mpdc[(R1 - 1, R2)])
            end
            if R2 > 0
                @test issubset(koszul2(pdc[(R1, R2)]), pdc[(R1, R2 - 1)])
                # @test issubset(koszul2(qpdc[(R1, R2)]), qpdc[(R1, R2 - 1)])
                # @test issubset(koszul2(tpdc[(R1, R2)]), tpdc[(R1, R2 - 1)])
                # @test issubset(koszul2(mpdc[(R1, R2)]), mpdc[(R1, R2 - 1)])
            end
        end
    end

    @time @testset "Double complexes P=$P D=$D T=$T p=$p" for P in [Pow],
                                                              D in 0:min(Dmax, 3),
                                                              T in bigtypes(P),
                                                              p in 1:max(2, 3 - D)
        # pdc = polynomial_double_complex(P, Val(D), T, p)
        # pdc = maximal_polynomial_double_complex(P, Val(D), T, p; pnorm=sum)
        mspdc = maximal_polynomial_double_complex(P, Val(D), T, p; pnorm=sum)
        mqpdc = maximal_polynomial_double_complex(P, Val(D), T, p; pnorm=maximum0)
        @test Set(keys(mspdc)) == Set((d1, d2) for d1 in 0:D, d2 in 0:D)
        @test Set(keys(mqpdc)) == Set((d1, d2) for d1 in 0:D, d2 in 0:D)

        for R1 in 0:D, R2 in 0:D
            @test mspdc[(R1, R2)] isa TensorBasis{P,D,R1,R2,T}
            @test mqpdc[(R1, R2)] isa TensorBasis{P,D,R1,R2,T}
            test_basis(mspdc[(R1, R2)])
            test_basis(mqpdc[(R1, R2)])
            # @test length(mspdc[(R1, R2)]) == mspdc_length(D, p, R1, R2)

            if R1 < D
                @test deriv1(mspdc[(R1, R2)]) ⊆ mspdc[(R1 + 1, R2)]
                @test deriv1(mqpdc[(R1, R2)]) ⊆ mqpdc[(R1 + 1, R2)]
            end
            if R2 < D
                @test deriv2(mspdc[(R1, R2)]) ⊆ mspdc[(R1, R2 + 1)]
                @test deriv2(mqpdc[(R1, R2)]) ⊆ mqpdc[(R1, R2 + 1)]
            end
            # This property does not seem to hold
            # @test ⋆mspdc[(R1, R2)] == mspdc[(D - R1, D - R2)]
            # @test ⋆mqpdc[(R1, R2)] == mqpdc[(D - R1, D - R2)]
            if R1 > 0
                @test ⋆deriv1(⋆mspdc[(R1, R2)]) ⊆ mspdc[(R1 - 1, R2)]
                # This property does not seem to hold for mqdpc
                # @test issubset(⋆deriv1(⋆mqpdc[(R1, R2)]), mqpdc[(R1 - 1, R2)])
            end
            if R2 > 0
                @test ⋆deriv2(⋆mspdc[(R1, R2)]) ⊆ mspdc[(R1, R2 - 1)]
                # This property does not seem to hold for mqdpc
                # @test issubset(⋆deriv2(⋆mqpdc[(R1, R2)]), mqpdc[(R1, R2 - 1)])
            end
            # This can fail because `koszul[12]` do not generate the constant function
            # if R1 > 0
            #     @test issubset(koszul1(mspdc[(R1, R2)]), mspdc[(R1 - 1, R2)])
            # end
            # if R2 > 0
            #     @test issubset(koszul2(mspdc[(R1, R2)]), mspdc[(R1, R2 - 1)])
            # end
        end
    end

    ################################################################################

    @time @testset "Whitney forms P=$P, D=$D R=$R T=$T" for P in [Pow], D in 0:Dmax, R in 0:D, T in types(P)
        ϕ = whitney(Basis{P,D + 1,R,T})
        @test length(ϕ.forms) == binomial(D + 1, R + 1)

        for n in 1:length(ϕ.forms)
            ϕₙ = ϕ.forms[n]
            ϕ′ = Basis{P,D + 1,R,T}([ϕ.forms[1:(n - 1)]; ϕ.forms[(n + 1):end]])
            @test !(ϕₙ in ϕ′)
        end
    end

    @time @testset "Convert from barycentric to Cartesian coordinates P=$P D=$D R=$R T=$T" for P in [Pow],
                                                                                               D in 0:Dmax,
                                                                                               R in 0:D,
                                                                                               T in types(P)

        ϕ = whitney(Basis{P,D + 1,R,T})
        @test length(ϕ.forms) == binomial(D + 1, R + 1)
        ϕx = barycentric2cartesian(ϕ)

        for n in 1:length(ϕx.forms)
            ϕₙ = ϕx.forms[n]
            ϕx′ = Basis{P,D,R,T}([ϕx.forms[1:(n - 1)]; ϕx.forms[(n + 1):end]])
            @test !(ϕₙ in ϕx′)
        end
    end

    @time @testset "Compare Whitney basis to trimmed p=1 basis P=$P D=$D R=$R T=$T" for P in [Pow],
                                                                                        D in 0:Dmax,
                                                                                        R in 0:D,
                                                                                        T in types(P)

        p = 1                       # Whitney basis is equivalent to p=1

        ϕλ = whitney(Basis{P,D + 1,R,T})
        ϕ = barycentric2cartesian(ϕλ)

        tpc = trimmed_polynomial_complex(P, Val(D), T, p)
        tpcR = tpc[R]

        @test ϕ == tpcR
    end
end                             # @time
