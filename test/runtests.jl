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
function Base.rand(rng::AbstractRNG,
                   ::Random.SamplerType{Term{D,T}}) where {D,T}
    return Term{D,T}(SVector{D,Int}(rand(rng, 0:10, D)), T(rand(rng, -10:10)))
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
        if !(x + (y + z) == (x + y) + z)
            @show x y z (x + y) (y + z) (x + (y + z)) ((x + y) + z)
        end
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
            if !(deriv(unit(Poly{D,T}, dir), dir) == e)
                @show D T dir unit(Poly{D,T}, dir) deriv(unit(Poly{D,T}, dir),
                                                         dir)
            end
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

################################################################################
