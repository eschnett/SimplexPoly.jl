# SimplexPoly

Provide various operations of polynomials and polynomial differential
forms that live on simplices.

* [Documentation](https://eschnett.github.io/SimplexPoly.jl/dev/): Future
  documentation
* [GitHub](https://github.com/eschnett/SimplexPoly.jl): Source code repository
* [![GitHub
  CI](https://github.com/eschnett/SimplexPoly.jl/workflows/CI/badge.svg)](https://github.com/eschnett/SimplexPoly.jl/actions)
* [![Codecov](https://codecov.io/gh/eschnett/SimplexPoly.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/eschnett/SimplexPoly.jl)

## Overview

This package provides three main datatypes:

- `Poly{D,T}`: Polynomials of type `T` of `D` unknowns
- `Form{D,R,Poly{D,T}}`: Polynomial differential forms of rank `R`
- `Basis{D,R,T}`: A basis for a subspace of polynomial differential
  forms

This packagae provides the usual arithmetic operations (scaling,
adding, multiplying) for polynomials, as well as derivative and
[Koszul](https://en.wikipedia.org/wiki/Koszul_complex) operators. (The
Koszul operator is similar to an antiderivative.) For subspaces, there
is a function to determine whether a particular polynomial form is
contained in it.

Differential forms are handled via the package `DifferentialForms.jl`.

This package can also calculate the PᵣΛᵏ and Pᵣ⁻Λᵏ families of finite
elements for the [Finite element exterior calculus
(FEEC)](https://en.wikipedia.org/wiki/Finite_element_exterior_calculus).
See [Periodic Table of the Finite
Elements](http://www-users.math.umn.edu/~arnold/femtable/) for an
overview, and [Periodic Table of the Finite
Elements](https://www-users.math.umn.edu/~arnold/papers/periodic-table.pdf)
for some details.

## Examples

### Polynomial complex PᵣΛᵏ

Polynomial complex PᵣΛᵏ for polynomial order r=2 in n=2 dimensions.
This complex consists of the spaces (P₂Λ⁰, P₁Λ¹, P₀Λ²) for scalars,
1-forms, and 2-forms, respectively. In this complex, higher rank forms
are represented via lower degree polynomials. All polynomials depend
on x and y.

```Julia
julia> using SimplexPoly

julia> pc = polynomial_complex(Val(2), Int, 2);

julia> pc[0]
Basis{2,0,Int64}[
    ⟦(1 * [0, 0])⟧,
    ⟦(1 * [0, 1])⟧,
    ⟦(1 * [0, 2])⟧,
    ⟦(1 * [1, 0])⟧,
    ⟦(1 * [1, 1])⟧,
    ⟦(1 * [2, 0])⟧]
```

These are the basis elements for scalar functions. There are 6 basis
polynomials. The factor `1 *` can be ignored. The vectors describe the
polynomial terms, i.e. the (quadratic) polynomials are: 1, y, y², x,
xy, x².

```Julia
julia> pc[1]
Basis{2,1,Int64}[
    ⟦(), (1 * [0, 0])⟧,
    ⟦(), (1 * [0, 1])⟧,
    ⟦(), (1 * [1, 0])⟧,
    ⟦(1 * [0, 0]), ()⟧,
    ⟦(1 * [0, 1]), ()⟧,
    ⟦(1 * [1, 0]), ()⟧]
```

These are the basis elements for 1-forms. Each basis element has two
components (for dx and dy). The 6 (linear) basis polynomials are: dy,
y dy, x dy, dx, y dx, x dx.

```Julia
julia> pc[2]
Basis{2,2,Int64}[⟦(1 * [0, 0])⟧]
```

These is the basis element for 2-forms. The only (constant) basis
polynomial is: dx ∧ dy.

TODO: This is wrong; there should be 3 basis polynomials.

### Trimmed polynomial complex Pᵣ⁻Λᵏ

Polynomial complex Pᵣ⁻Λᵏ for polynomial order r=2 in n=2 dimensions.
This complex consists of the spaces (P₂⁻Λ⁰, P₂⁻Λ¹, P₂⁻Λ²) for scalars,
1-forms, and 2-forms, respectively. In this complex, forms of all
ranks are represented via polynomials of approximately the same
degree. All polynomials depend on x and y.

```Julia
julia> using SimplexPoly

julia> tpc = trimmed_polynomial_complex(Val(2), Int, 2);

julia> tpc[0]
Basis{2,0,Int64}[
    ⟦(1 * [0, 0])⟧,
    ⟦(1 * [0, 1])⟧,
    ⟦(1 * [0, 2])⟧,
    ⟦(1 * [1, 0])⟧,
    ⟦(1 * [1, 1])⟧,
    ⟦(1 * [2, 0])⟧]
```

This is the same space as P₂Λ⁰ described above with the (quadratic)
polynomials: 1, y, y², x, xy, x².

```Julia
julia> tpc[1]
Basis{2,1,Int64}[
    ⟦(), (1 * [0, 0])⟧,
    ⟦(), (1 * [0, 1])⟧,
    ⟦(), (1 * [1, 0])⟧,
    ⟦(1 * [0, 0]), ()⟧,
    ⟦(1 * [0, 1]), ()⟧,
    ⟦(-1 * [0, 2]), (1 * [1, 1])⟧,
    ⟦(1 * [1, 0]), ()⟧,
    ⟦(-1 * [1, 1]), (1 * [2, 0])⟧]
```

These are the basis elements for 1-forms. Each basis element has two
components (for dx and dy). This space contains some, but not all
quadratic polynomials. The 8 basis polynomials are: dy, y dy, x dy,
dx, y dx, -x² dx + x y dy, x dx, - x y dx + y² dy.

```Julia
julia> tpc[2]
Basis{2,2,Int64}[
    ⟦(1 * [0, 0])⟧,
    ⟦(1 * [0, 1])⟧,
    ⟦(1 * [1, 0])⟧]
```

These are the basis element for 2-forms. The 3 (linear) basis
polynomials are: dx ∧ dy, y dx ∧ dy, x dx ∧ dy.

## Related work

See also <https://github.com/JuliaMath/Polynomials.jl> and
<https://github.com/scheinerman/SimplePolynomials.jl>; these support
only one-dimensional polynomials.
