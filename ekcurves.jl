module EKCurves

using LinearAlgebra
using Polynomials: Polynomial, roots

using Gtk
import Graphics

include("trig-coeffs.jl")


# Parameters
maxiter = 400
distance_tolerance = 1e-4
warn_on_convergence_failure = false
warn_on_multiple_solutions = false

# GUI parameters
width = 500
height = 500
resolution = 50
curvature_scaling = 2000
curvature_sparsity = 10
point_size = 5
rectangle_scale = 1.4

# GUI variables
show_controls = false
show_curvature = true
just_curve = false
closed_curve = true
curve_type = 0
alpha = 2/3
incremental = false
current_file = nothing

# Dirty hack
global radios
alpha_index = 1
selected_base = 0

# Global variables
# `controls`, `curve` and `curvature` are vectors of curve-data
# The first `nr_closed` curve is assumed to be closed, the rest is open
# `points` and `max_curvature` contain the points of all curves without distinction
global points
global controls
global curve
global curvature
global max_curvatures
global nr_closed
global selected
global selected_alpha

# Designs
designs = ["bear", "deer", "elephant", "bird", "plane", "pumpkin", "rose", "dinosaur"]


# Interpolation

"""
    interpolate(points; closed, cubic, alpha)

Interpolates an (ϵ)-κ-curve on the given points.
The result is given as `(c, t)`, where `c` contains the curves (as a vector of control points),
and `t` are the parameters of the interpolated points.
"""
function interpolate(points; closed = true, cubic = false, alpha = 2/3)
    n = length(points) - (closed ? 0 : 2)

    c = map(p -> [nothing, p, nothing], closed ? points : points[2:end-1])
    if (!closed)
        c[1][1] = points[1]
        c[n][3] = points[end]
        points = points[2:end-1]
    end

    λ = fill(0.5, n)
    t = Vector{Float64}(undef, n)

    # Setup right-hand side for the linear equation system
    rhs = Matrix{Float64}(undef, n, 2)
    for i in 1:n
        rhs[i,:] = points[i]
    end

    update_endpoints!(c, λ, closed)

    for iteration in 1:maxiter
        λ = compute_lambdas(c, closed)
        update_endpoints!(c, λ, closed)
        if cubic
            for i in 1:n
                a = (i + (closed ? 0 : 1)) in selected ? selected_alpha : alpha
                t[i] = compute_parameter(c[i], points[i], a)
            end
        else
            t = [compute_parameter(c[i], points[i]) for i in 1:n]
        end

        if iteration == maxiter
            max_error = maximum(1:n) do i
                a = (i + (closed ? 0 : 1)) in selected ? selected_alpha : alpha
                cp = cubic ? create_cubic(c[i], a) : c[i]
                norm(bezier_eval(cp, t[i]) - points[i])
            end
            warn_on_convergence_failure && @warn "Did not converge - err: $max_error"
            break
        end

        if cubic
            x = compute_central_cps(c, λ, t, rhs, closed, alpha)
        else
            x = compute_central_cps(c, λ, t, rhs, closed)
        end
        max_deviation = 0
        for i in 1:n
            max_deviation = max(max_deviation, norm(x[i,:] - c[i][2]))
            c[i][2] = x[i,:]
        end

        max_deviation < distance_tolerance && break
    end

    update_endpoints!(c, λ, closed)
    (c, t)
end

"""
    update_endpoints!(c, λ, closed)

Destructively updates the endpoints of each curve segment, based on the λ values.
"""
function update_endpoints!(c, λ, closed)
    n = length(c)
    for i in 1:n
        ip = mod1(i + 1, n)
        if closed || i < n
            c[i][3] = (1 - λ[i]) * c[i][2] + λ[i] * c[ip][2]
        end
        if closed || ip > 1
            c[ip][1] = c[i][3]
        end
    end
end

"""
    compute_lambdas(c, closed)

Computes λ values based on control point triangle areas.
"""
function compute_lambdas(c, closed)
    n = length(c)
    map(1:n) do i
        !closed && i == n && return 0 # not used
        ip = mod1(i + 1, n)
        a = (i + (closed ? 0 : 1)) in selected ? selected_alpha : alpha
        ap = (ip + (closed ? 0 : 1)) in selected ? selected_alpha : alpha
        tmp = sqrt((1 - a) * abs(Δ(c[i][1], c[i][2], c[ip][2])))
        denom = (tmp + a / ap * sqrt((1 - ap) * abs(Δ(c[i][2], c[ip][2], c[ip][3]))))
        if denom < 1e-10
            denom += 1e-10
        end
        tmp / denom
    end
end

"""
    Δ(a, b, c)

Computes the signed area of the triangle defined by the points `a`, `b` and `c`.
"""
Δ(a, b, c) = det([b - a  c - a]) / 2

"""
    compute_parameter(curve, p)

Computes the parameter where a quadratic curve with the same endpoints
takes its largest curvature value, and also interpolates the given `p` point.
"""
function compute_parameter(curve, p)
    coeffs = [-norm(curve[1] - p) ^ 2,
              dot(3 * curve[1] - 2 * p - curve[3], curve[1] - p),
              3 * dot(curve[3] - curve[1], curve[1] - p),
              norm(curve[3] - curve[1]) ^ 2]
    unit_root(coeffs)
end

"""
    unit_root(coeffs)

Finds a real root of the polynomial in `[0, 1]`.
The coefficients are given in the order `[1, x, x^2, ...]`.
"""
function unit_root(coeffs)
    p = Polynomial(coeffs)
    find_suitable_root(roots(p))
end

"""
    find_suitable_root(roots)

Given a vector of complex numbers, selects one that is read and in `[0,1]`.
When the global variable  `warn_on_multiple_solutions` is `true`,
this function displays warnings on failure.

When multiple solutions exist, it will return the first acceptable one.
When there is none, it returns 0.5.
"""
function find_suitable_root(roots, ϵ = 1e-8)
    if warn_on_multiple_solutions
        sol = []
        for xi in roots
            if abs(imag(xi)) < ϵ && -ϵ <= real(xi) <= 1 + ϵ
                push!(sol, xi)
            end
        end
        isempty(sol) && (@warn "No suitable solution"; return 0.5)
        length(sol) > 1 && @warn "Multiple solutions: $sol"
        clamp(real(sol[1]), 0, 1)
    else
        for xi in roots
            abs(imag(xi)) < ϵ && -ϵ <= real(xi) <= 1 + ϵ && return clamp(real(xi), 0, 1)
        end
        0.5
    end
end

"""
    compute_central_cps(c, λ, t, points, closed)

Computes the central control points of the quadratic Bezier curves `c`
in such a way that `c(t[i]) = points[i]`, where `points` is a matrix of size `(n, 2)`.
The control points satisfy `c[i][3] = (1-λ[i]) c[i][2] + λ[i] c[i+1][2]`.
The result is also a matrix of size `(n, 2)`.
"""
function compute_central_cps(c, λ, t, points, closed)
    n = length(c)
    A = zeros(n, n)
    fixed = zeros(n, 2)
    for i in 1:n
        im = mod1(i - 1, n)
        ip = mod1(i + 1, n)
        if closed || i > 1
            A[i,im] = (1 - λ[im]) * (1 - t[i]) ^ 2
        else
            fixed[1,:] -= c[1][1] * (1 - t[i]) ^ 2
        end
        if closed || i < n
            A[i,ip] = λ[i] * t[i] ^ 2
        else
            fixed[n,:] -= c[n][3] * t[i] ^ 2
        end
        if closed || 1 < i < n
            A[i,i] = λ[im] * (1 - t[i]) ^ 2 + (2 - (1 + λ[i]) * t[i]) * t[i]
        elseif n == 1
            A[i,i] = 2 * (1 - t[i]) * t[i]
        elseif i == 1
            A[i,i] = (2 - (1 + λ[i]) * t[i]) * t[i]
        else # i == n
            A[i,i] = λ[im] * (1 - t[i]) ^ 2 + 2 * (1 - t[i]) * t[i]
        end
    end
    A \ (points + fixed)
end


# Bezier evaluation

"""
    bernstein(n, u)

Computes the Bernstein polynomials of degree `n` at the parameter `u`.
"""
function bernstein(n, u)
    coeff = [1.0]
    for j in 1:n
        saved = 0.0
        for k in 1:j
            tmp = coeff[k]
            coeff[k] = saved + tmp * (1.0 - u)
            saved = tmp * u
        end
        push!(coeff, saved)
    end
    coeff
end

"""
    bernstein_all(n, u)

Computes all Bernstein polynomials up to degree `n` at the parameter `u`.
"""
function bernstein_all(n, u)
    result = [[1.0]]
    coeff = [1.0]
    for j in 1:n
        saved = 0.0
        for k in 1:j
            tmp = coeff[k]
            coeff[k] = saved + tmp * (1.0 - u)
            saved = tmp * u
        end
        push!(coeff, saved)
        push!(result, copy(coeff))
    end
    result
end

"""
    bezier_derivative_controls(curve, d)

Computes the control points for the `d`-th derivative computation.
The Bezier curve is given by its control points.
"""
function bezier_derivative_controls(curve, d)
    n = length(curve) - 1
    dcp = [copy(curve)]
    for k in 1:d
        tmp = n - k + 1
        cp = []
        for i in 1:tmp
            push!(cp, (dcp[k][i+1] - dcp[k][i]) * tmp)
        end
        push!(dcp, cp)
    end
    dcp
end

"""
    bezier_eval(curve, u)

Evaluates a Bezier curve, given by its control points, at the parameter `u`.
"""
function bezier_eval(curve, u)
    n = length(curve) - 1
    coeff = bernstein(n, u)
    sum(curve .* coeff)
end

"""
    bezier_eval(curve, u, d)

Evaluates a Bezier curve, given by its control points, at the parameter `u`, with `d` derivatives.
"""
function bezier_eval(curve, u, d)
    result = []
    n = length(curve) - 1
    du = min(d, n)
    coeff = bernstein_all(n, u)
    dcp = bezier_derivative_controls(curve, du)
    for k in 0:du
        push!(result, sum(dcp[k+1] .* coeff[n-k+1]))
    end
    for k in n+1:d
        push!(result, [0, 0])
    end
    result
end

"""
    bezier_curvature(curve, u)

Computes the curvature at the given parameter.
"""
function bezier_curvature(curve, u)
    der = bezier_eval(curve, u, 2)
    det([der[2] der[3]]) / norm(der[2]) ^ 3
end

"""
    bezier_curvature_derivatives(curve, u)

Computes the first two derivatives of the curvature at `u`.
"""
function bezier_curvature_derivatives(curve, u)
    _, d1, d2, d3, d4 = bezier_eval(curve, u, 4)

    d1d2 = dot(d1, d2)
    d1d3 = dot(d1, d3)
    d2d2 = dot(d2, d2)

    d1xd2 = det([d1 d2])
    d1xd3 = det([d1 d3])
    d1xd4 = det([d1 d4])
    d2xd3 = det([d2 d3])

    n = norm(d1)

    r1 = d1xd3 / n ^ 3 - 3 * d1xd2 * d1d2 / n ^ 5

    w0 = d2xd3 + d1xd4
    w1 = 3 * d1xd3 * d1d2
    w2 = 3 * (d1xd3 * d1d2 + d1xd2 * (d2d2 + d1d3))
    w3 = 15 * d1xd2 * d1d2 ^ 2
    r2 = w0 / n ^ 3 - w1 / n ^ 5 - w2 / n ^ 5 + w3 / n ^ 7

    (r1, r2)
end


# Cubic version

"""
    compute_parameter(curve, p, alpha)

Computes the parameter where a cubic curve with the same endpoints
takes its largest curvature value, and also interpolates the given `p` point.
"""
function compute_parameter(curve, p, alpha)
    a = alpha
    px, py = p
    x0, y0 = curve[1]
    x2, y2 = curve[3]
    # coeffs generated by ek-curvature.mac (using Maxima)
    coeffs = [-3*a*y0^2+4*y0^2+6*a*py*y0-8*py*y0-3*a*x0^2+4*x0^2+6*a*px*x0-8*px*x0-3*a*py^2+4*py^2-3*a*px^2+4*px^2,
              18*a^2*y0*y2-36*a*y0*y2+18*y0*y2-18*a^2*py*y2+36*a*py*y2-18*py*y2-18*a^2*y0^2+48*a*y0^2-30*y0^2+18*a^2*py*y0-60*a*py*y0+42*py*y0+18*a^2*x0*x2-36*a*x0*x2+18*x0*x2-18*a^2*px*x2+36*a*px*x2-18*px*x2-18*a^2*x0^2+48*a*x0^2-30*x0^2+18*a^2*px*x0-60*a*px*x0+42*px*x0+12*a*py^2-12*py^2+12*a*px^2-12*px^2,
              (-144*a^2*y0*y2)+258*a*y0*y2-114*y0*y2+144*a^2*py*y2-258*a*py*y2+114*py*y2+144*a^2*y0^2-276*a*y0^2+126*y0^2-144*a^2*py*y0+294*a*py*y0-138*py*y0-144*a^2*x0*x2+258*a*x0*x2-114*x0*x2+144*a^2*px*x2-258*a*px*x2+114*px*x2+144*a^2*x0^2-276*a*x0^2+126*x0^2-144*a^2*px*x0+294*a*px*x0-138*px*x0-18*a*py^2+12*py^2-18*a*px^2+12*px^2,
              54*a^3*y2^2-162*a^2*y2^2+162*a*y2^2-54*y2^2-108*a^3*y0*y2+774*a^2*y0*y2-1032*a*y0*y2+380*y0*y2-450*a^2*py*y2+708*a*py*y2-272*py*y2+54*a^3*y0^2-612*a^2*y0^2+882*a*y0^2-334*y0^2+450*a^2*py*y0-732*a*py*y0+288*py*y0+54*a^3*x2^2-162*a^2*x2^2+162*a*x2^2-54*x2^2-108*a^3*x0*x2+774*a^2*x0*x2-1032*a*x0*x2+380*x0*x2-450*a^2*px*x2+708*a*px*x2-272*px*x2+54*a^3*x0^2-612*a^2*x0^2+882*a*x0^2-334*x0^2+450*a^2*px*x0-732*a*px*x0+288*px*x0+12*a*py^2-8*py^2+12*a*px^2-8*px^2,
              (-405*a^3*y2^2)+1080*a^2*y2^2-945*a*y2^2+270*y2^2+810*a^3*y0*y2-2880*a^2*y0*y2+2910*a*y0*y2-900*y0*y2+720*a^2*py*y2-1020*a*py*y2+360*py*y2-405*a^3*y0^2+1800*a^2*y0^2-1965*a*y0^2+630*y0^2-720*a^2*py*y0+1020*a*py*y0-360*py*y0-405*a^3*x2^2+1080*a^2*x2^2-945*a*x2^2+270*x2^2+810*a^3*x0*x2-2880*a^2*x0*x2+2910*a*x0*x2-900*x0*x2+720*a^2*px*x2-1020*a*px*x2+360*px*x2-405*a^3*x0^2+1800*a^2*x0^2-1965*a*x0^2+630*x0^2-720*a^2*px*x0+1020*a*px*x0-360*px*x0,
              1296*a^3*y2^2-3114*a^2*y2^2+2454*a*y2^2-636*y2^2-2592*a^3*y0*y2+6822*a^2*y0*y2-5700*a*y0*y2+1536*y0*y2-594*a^2*py*y2+792*a*py*y2-264*py*y2+1296*a^3*y0^2-3708*a^2*y0^2+3246*a*y0^2-900*y0^2+594*a^2*py*y0-792*a*py*y0+264*py*y0+1296*a^3*x2^2-3114*a^2*x2^2+2454*a*x2^2-636*x2^2-2592*a^3*x0*x2+6822*a^2*x0*x2-5700*a*x0*x2+1536*x0*x2-594*a^2*px*x2+792*a*px*x2-264*px*x2+1296*a^3*x0^2-3708*a^2*x0^2+3246*a*x0^2-900*x0^2+594*a^2*px*x0-792*a*px*x0+264*px*x0,
              (-2268*a^3*y2^2)+5004*a^2*y2^2-3648*a*y2^2+880*y2^2+4536*a^3*y0*y2-10206*a^2*y0*y2+7560*a*y0*y2-1848*y0*y2+198*a^2*py*y2-264*a*py*y2+88*py*y2-2268*a^3*y0^2+5202*a^2*y0^2-3912*a*y0^2+968*y0^2-198*a^2*py*y0+264*a*py*y0-88*py*y0-2268*a^3*x2^2+5004*a^2*x2^2-3648*a*x2^2+880*x2^2+4536*a^3*x0*x2-10206*a^2*x0*x2+7560*a*x0*x2-1848*x0*x2+198*a^2*px*x2-264*a*px*x2+88*px*x2-2268*a^3*x0^2+5202*a^2*x0^2-3912*a*x0^2+968*x0^2-198*a^2*px*x0+264*a*px*x0-88*px*x0,
              2268*a^3*y2^2-4698*a^2*y2^2+3240*a*y2^2-744*y2^2-4536*a^3*y0*y2+9396*a^2*y0*y2-6480*a*y0*y2+1488*y0*y2+2268*a^3*y0^2-4698*a^2*y0^2+3240*a*y0^2-744*y0^2+2268*a^3*x2^2-4698*a^2*x2^2+3240*a*x2^2-744*x2^2-4536*a^3*x0*x2+9396*a^2*x0*x2-6480*a*x0*x2+1488*x0*x2+2268*a^3*x0^2-4698*a^2*x0^2+3240*a*x0^2-744*x0^2,
              (-1215*a^3*y2^2)+2430*a^2*y2^2-1620*a*y2^2+360*y2^2+2430*a^3*y0*y2-4860*a^2*y0*y2+3240*a*y0*y2-720*y0*y2-1215*a^3*y0^2+2430*a^2*y0^2-1620*a*y0^2+360*y0^2-1215*a^3*x2^2+2430*a^2*x2^2-1620*a*x2^2+360*x2^2+2430*a^3*x0*x2-4860*a^2*x0*x2+3240*a*x0*x2-720*x0*x2-1215*a^3*x0^2+2430*a^2*x0^2-1620*a*x0^2+360*x0^2,
              270*a^3*y2^2-540*a^2*y2^2+360*a*y2^2-80*y2^2-540*a^3*y0*y2+1080*a^2*y0*y2-720*a*y0*y2+160*y0*y2+270*a^3*y0^2-540*a^2*y0^2+360*a*y0^2-80*y0^2+270*a^3*x2^2-540*a^2*x2^2+360*a*x2^2-80*x2^2-540*a^3*x0*x2+1080*a^2*x0*x2-720*a*x0*x2+160*x0*x2+270*a^3*x0^2-540*a^2*x0^2+360*a*x0^2-80*x0^2]
    unit_root(coeffs)
end

"""
    compute_central_cps(c, λ, t, points, closed, alpha)

Computes the "central" control points of the cubic Bezier curves `c`
in such a way that `c(t[i]) = points[i]`, where `points` is a matrix of size `(n, 2)`.
The control points satisfy `c[i][3] = (1-λ[i]) c[i][2] + λ[i] c[i+1][2]`.
The result is also a matrix of size `(n, 2)`.
"""
function compute_central_cps(c, λ, t, points, closed, alpha)
    n = length(c)
    A = zeros(n, n)
    fixed = zeros(n, 2)
    for i in 1:n
        a = (i + (closed ? 0 : 1)) in selected ? selected_alpha : alpha
        im = mod1(i - 1, n)
        ip = mod1(i + 1, n)
        if closed || i > 1
            A[i,im] = (1 - λ[im]) * ((1 - t[i]) ^ 3 + 3 * (1 - t[i]) ^ 2 * t[i] * (1 - a))
        else
            fixed[1,:] -= c[1][1] * ((1 - t[i]) ^ 3 + 3 * (1 - t[i]) ^ 2 * t[i] * (1 - a))
        end
        if closed || i < n
            A[i,ip] = λ[i] * (t[i] ^ 3 + 3 * (1 - t[i]) * t[i] ^ 2 * (1 - a))
        else
            fixed[n,:] -= c[n][3] * (t[i] ^ 3 + 3 * (1 - t[i]) * t[i] ^ 2 * (1 - a))
        end
        if closed || 1 < i < n
            A[i,i] =
                λ[im] * ((1 - t[i]) ^ 3 + 3 * (1 - t[i]) ^ 2 * t[i] * (1 - a)) +
                (1 - λ[i]) * (t[i] ^ 3 + 3 * (1 - t[i]) * t[i] ^ 2 * (1 - a)) +
                a * 3 * (1 - t[i]) * t[i]
        elseif n == 1
            A[i,i] = a * 3 * (1 - t[i]) * t[i]
        elseif i == 1
            A[i,i] =
                (1 - λ[i]) * (t[i] ^ 3 + 3 * (1 - t[i]) * t[i] ^ 2 * (1 - a)) +
                a * 3 * (1 - t[i]) * t[i]
        else # i == n
            A[i,i] =
                λ[im] * ((1 - t[i]) ^ 3 + 3 * (1 - t[i]) ^ 2 * t[i] * (1 - a)) +
                a * 3 * (1 - t[i]) * t[i]
        end
    end
    A \ (points + fixed)
end

"""
    create_cubic(points, ratio)

Creates the control points of a cubic Bezier curve based on 3 `points`
and the given `ratio`. When `ratio == 2/3`, this will be the same curve
as the quadratic Bezier curve defined by the same points.
"""
function create_cubic(points, ratio)
    [points[1],
     points[1] * (1 - ratio) + points[2] * ratio,
     points[2] * ratio + points[3] * (1 - ratio),
     points[3]]
end


# Interpolation with trigonometric curves

"""
    interpolate(points; closed, alpha)

Interpolates a trigonometric ϵ-κ-curve on the given points.
The result is given as `(c, t)`, where `c` contains the curves (as a vector of control points),
and `t` are the parameters of the interpolated points.
"""
function interpolate_trig(points; closed = true, alpha = 1)
    n = length(points) - (closed ? 0 : 2)

    c = map(p -> [nothing, p, nothing], closed ? points : points[2:end-1])
    if (!closed)
        c[1][1] = points[1]
        c[n][3] = points[end]
        points = points[2:end-1]
    end

    λ = fill(0.5, n)
    t = Vector{Float64}(undef, n)

    # Setup right-hand side for the linear equation system
    rhs = Matrix{Float64}(undef, n, 2)
    for i in 1:n
        rhs[i,:] = points[i]
    end

    update_endpoints!(c, λ, closed)

    for iteration in 1:maxiter
        λ = compute_lambdas(c, closed)
        update_endpoints!(c, λ, closed)
        t = [compute_parameter_trig(c[i], points[i], alpha) for i in 1:n]

        if iteration == maxiter
            max_error = maximum(1:n) do i
                norm(trig_eval(c[i], alpha, t[i], 0) - points[i])
            end
            warn_on_convergence_failure && @warn "Did not converge - err: $max_error"
            break
        end

        x = compute_central_cps_trig(c, λ, t, rhs, closed, alpha)
        max_deviation = 0
        for i in 1:n
            max_deviation = max(max_deviation, norm(x[i,:] - c[i][2]))
            c[i][2] = x[i,:]
        end

        max_deviation < distance_tolerance && break
    end

    update_endpoints!(c, λ, closed)
    (c, t)
end

"""
    compute_parameter_trig(curve, p, alpha)

Computes the parameter where a trigonometric curve with the same endpoints
takes its largest curvature value, and also interpolates the given `p` point.
"""
compute_parameter_trig(curve, p, alpha) = solve_trig(coeffs_trig(curve, p, alpha))

"""
    solve_trig(coeffs)

Finds a root of a polynomial based on `x=exp(π*im*t/2)`,
such that t is real and is in `[0, 1]`.
The coefficients are given in the order `[1, x, x^2, ...]`.
"""
function solve_trig(coeffs)
    p = Polynomial(coeffs)
    find_suitable_root([log(r) * 2 / (im * π) for r in roots(p)])
end

"""
    compute_central_cps_trig(c, λ, t, points, closed, alpha)

Computes the central control points of the "quadratic" trigonometric curves `c`
in such a way that `c(t[i]) = points[i]`, where `points` is a matrix of size `(n, 2)`.
The control points satisfy `c[i][3] = (1-λ[i]) c[i][2] + λ[i] c[i+1][2]`.
The result is also a matrix of size `(n, 2)`.
"""
function compute_central_cps_trig(c, λ, t, points, closed, alpha)
    n = length(c)
    A = zeros(n, n)
    fixed = zeros(n, 2)
    for i in 1:n
        im = mod1(i - 1, n)
        ip = mod1(i + 1, n)
        S = sin(π * t[i] / 2)
        C = cos(π * t[i] / 2)
        if closed || i > 1
            A[i,im] = (1 + (alpha - 1) * S ^ 2 - alpha * S) * (1 - λ[im])
        else
            fixed[1,:] -= c[1][1] * (1 + (alpha - 1) * S ^ 2 - alpha * S)
        end
        if closed || i < n
            A[i,ip] = (1 + (alpha - 1) * C ^ 2 - alpha * C) * λ[i]
        else
            fixed[n,:] -= c[n][3] * (1 + (alpha - 1) * C ^ 2 - alpha * C)
        end
        if closed || 1 < i < n
            A[i,i] =
                λ[im] * (1 + (alpha - 1) * S ^ 2 - alpha * S) +
                alpha * (S + C - 1) +
                (1 - λ[i]) * (1 + (alpha - 1) * C ^ 2 - alpha * C)
        elseif n == 1
            A[i,i] = alpha * (S + C - 1)
        elseif i == 1
            A[i,i] =
                alpha * (S + C - 1) +
                (1 - λ[i]) * (1 + (alpha - 1) * C ^ 2 - alpha * C)
        else # i == n
            A[i,i] =
                λ[im] * (1 + (alpha - 1) * S ^ 2 - alpha * S) +
                alpha * (S + C - 1)
        end
    end
    A \ (points + fixed)
end


# Trigonometric curve evaluation

"""
    trig_eval(curve, alpha, u, d)

Evaluates a trigonometric curve, given by its control points,
at the parameter `u`, with `d` derivatives.
`alpha` is a parameter of the basis functions.
"""
function trig_eval(curve, alpha, u, d)
    S = sin(π * u / 2)
    C = cos(π * u / 2)
    b = [1 - (1 - alpha) * S ^ 2 - alpha * S,
         alpha * (S + C - 1),
         1 - (1 - alpha) * C ^ 2 - alpha * C]
    p = sum(curve .* b)
    d == 0 && return p
    der = [p]
    db1 = [-π * (1 - alpha) * S * C - π / 2 * alpha * C,
           π / 2 * alpha * (C - S),
           π * (1 - alpha) * C * S + π / 2 * alpha * S]
    push!(der, sum(curve .* db1))
    d == 1 && return der
    db2 = [π ^ 2 / 2 * (1 - alpha) * (S ^ 2 - C ^ 2) + π ^ 2 / 4 * alpha * S,
           -alpha * π ^ 2 / 4 * (S + C),
           π ^ 2 / 2 * (1 - alpha) * (C ^ 2 - S ^ 2) + π ^ 2 / 4 * alpha * C]
    push!(der, sum(curve .* db2))
    der
end

"""
    trig_curvature(curve, u)

Computes the curvature at the given parameter.
"""
function trig_curvature(curve, alpha, u)
    der = trig_eval(curve, alpha, u, 2)
    det([der[2] der[3]]) / norm(der[2]) ^ 3
end



# I/O

function load_design(filename)
    open_curves = []
    closed_curves = []

    read_point(f) = [parse(Float64, s) for s in split(readline(f))]
    function read_curve(f)
        cpts = []
        n = parse(Int, readline(f))
        for i in 1:n
            push!(cpts, read_point(f))
        end
        cpts
    end

    open(filename) do f
        n = parse(Int, readline(f))
        for i in 1:n
            push!(closed_curves, read_curve(f))
        end
        n = parse(Int, readline(f))
        for i in 1:n
            push!(open_curves, read_curve(f))
        end
    end

    (open_curves, closed_curves)
end


# Graphics

function draw_polygon(ctx, poly, closep = false)
    if isempty(poly)
        return
    end
    Graphics.new_path(ctx)
    Graphics.move_to(ctx, poly[1][1], poly[1][2])
    for p in poly[2:end]
        Graphics.line_to(ctx, p[1], p[2])
    end
    if closep && length(poly) > 2
        Graphics.line_to(ctx, poly[1][1], poly[1][2])
    end
    Graphics.stroke(ctx)
end

draw_callback = @guarded (canvas) -> begin
    ctx = Graphics.getgc(canvas)

    # White background
    Graphics.rectangle(ctx, 0, 0, Graphics.width(canvas), Graphics.height(canvas))
    Graphics.set_source_rgb(ctx, 1, 1, 1)
    Graphics.fill(ctx)

    if !just_curve && show_controls
        # Control polygon
        Graphics.set_source_rgb(ctx, 1, 0, 1)
        Graphics.set_line_width(ctx, 2.0)
        foreach(i -> draw_polygon(ctx, controls[i], true), 1:nr_closed)
        foreach(i -> draw_polygon(ctx, controls[i], false), nr_closed+1:length(controls))
    end

    # Generated curve
    Graphics.set_source_rgb(ctx, 0.8, 0.3, 0)
    Graphics.set_line_width(ctx, 2.0)
    foreach(i -> draw_polygon(ctx, curve[i], true), 1:nr_closed)
    foreach(i -> draw_polygon(ctx, curve[i], false), nr_closed+1:length(curve))

    just_curve && return

    # Curvature comb
    if show_curvature
        Graphics.set_source_rgb(ctx, 0, 0, 1)
        Graphics.set_line_width(ctx, 1.0)
        foreach(i -> draw_polygon(ctx, curvature[i], true), 1:nr_closed)
        foreach(i -> draw_polygon(ctx, curvature[i], false), nr_closed+1:length(curvature))
        for j in 1:length(curvature)
            for i in 1:length(curve[j])
                i % curvature_sparsity != 1 && i != length(curvature[j]) && continue
                Graphics.new_path(ctx)
                Graphics.move_to(ctx, curve[j][i][1], curve[j][i][2])
                Graphics.line_to(ctx, curvature[j][i][1], curvature[j][i][2])
                Graphics.stroke(ctx)
            end
        end
    end

    if show_curvature
        # Maximum curvature points
        for p in max_curvatures[1:end]
            Graphics.set_source_rgb(ctx, 1, 0, 0)
            Graphics.arc(ctx, p[1], p[2], point_size - 1, 0, 2pi)
            Graphics.fill(ctx)
        end
    end

    # Input points
    for i in 1:length(points)
        p = points[i]
        if show_curvature
            Graphics.set_source_rgb(ctx, 0, 1, 0)
            Graphics.arc(ctx, p[1], p[2], point_size, 0, 2pi)
            Graphics.fill(ctx)
        end
        if (i - selected_base) in selected
            Graphics.set_source_rgb(ctx, 0, 0, 1)
            Graphics.set_line_width(ctx, 2.0)
        else
            Graphics.set_source_rgb(ctx, 0, 0, 0)
            Graphics.set_line_width(ctx, 1.0)
        end
        rect = [p + [-point_size, -point_size] * rectangle_scale,
                p + [-point_size,  point_size] * rectangle_scale,
                p + [ point_size,  point_size] * rectangle_scale,
                p + [ point_size, -point_size] * rectangle_scale]
        draw_polygon(ctx, rect, true)
    end
end


# GUI

function clear_variables!()
    global points = []
    global controls = []
    global curve = []
    global curvature = []
    global max_curvatures = []
    global current_file = nothing
    global nr_closed = 0
    global selected = []
    global selected_alpha = alpha
end

function generate_curve()
    if !incremental
        global controls = []
        global curve = []
        global curvature = []
        global max_curvatures = []
        global nr_closed = 0
    end

    if length(points) < 3
        if length(points) == 2
            push!(controls, [points[1], (points[1] + points[2]) / 2, points[2]])
            push!(curve, [points[1], points[2]])
            push!(curvature, [points[1], points[2]])
        end
        return
    end
    local cpts, t, eval_fn, curvature_fn

    if curve_type == 0
        # Original quadratic
        cpts, t = interpolate(points, closed=closed_curve, cubic=false)
        eval_fn, curvature_fn = bezier_eval, bezier_curvature
    elseif curve_type == 1
        # Cubic
        cpts, t = interpolate(points, closed=closed_curve, cubic=true, alpha=alpha)
        for i in 1:length(cpts)
            a = (i + (closed_curve ? 0 : 1)) in selected ? selected_alpha : alpha
            cpts[i] = create_cubic(cpts[i], a)
        end
        eval_fn, curvature_fn = bezier_eval, bezier_curvature
    elseif curve_type == 2
        # Trigonometric
        cpts, t = interpolate_trig(points, closed=closed_curve, alpha=alpha * 2)
        eval_fn = (curve, u, d=0) -> trig_eval(curve, alpha * 2, u, d)
        curvature_fn = (curve, u) -> trig_curvature(curve, alpha * 2, u)
    end

    push!(controls, vcat(cpts...))
    push!(curve, [])
    push!(curvature, [])
    nr_closed += closed_curve ? 1 : 0

    for c in cpts
        for u in range(0, stop=1, length=resolution) # endpoints are drawn twice
            der = eval_fn(c, u, 1)
            p = der[1]
            n = normalize([der[2][2], -der[2][1]])
            k = curvature_fn(c, u)
            push!(curve[end], p)
            push!(curvature[end], p + n * k * curvature_scaling)
        end
    end
    append!(max_curvatures, [eval_fn(cpts[i], t[i]) for i in 1:length(t)])
end

function display_design(canvas, filename)
    open_curves, closed_curves = load_design(filename)
    old_closed = closed_curve
    old_selected = selected
    old_selected_alpha = selected_alpha
    clear_variables!()
    global selected = old_selected
    global incremental = true
    global closed_curve = true
    all_points = []
    for c in closed_curves
        global points = c
        append!(all_points, points)
        generate_curve()
    end
    global closed_curve = false
    for c in open_curves
        global points = c
        if c === open_curves[end]
            global selected_alpha = old_selected_alpha
            global selected_base = length(all_points)
        end
        append!(all_points, points)
        generate_curve()
    end
    global points = all_points
    draw(canvas)
    clear_variables!()
    global selected = old_selected
    global selected_alpha = old_selected_alpha
    points = open_curves[end]
    global closed_curve = old_closed
    global incremental = false
    global current_file = filename
end

mousedown_handler = @guarded (canvas, event) -> begin
    p = [event.x, event.y]
    global clicked = findfirst(points) do q
        norm(p - q) < 10
    end
    if clicked === nothing
        global current_file = nothing
        push!(points, p)
        clicked = length(points)
        refresh(canvas, true)
    else
        global selected
        if event.state & 1 == 1     # shift
            push!(selected, clicked)
            refresh(canvas, true)
        elseif event.state & 4 == 4 # control
            selected = [clicked]
            refresh(canvas, true)
        elseif !isempty(selected)
            global selected = []
            global radios[alpha_index].active[Bool] = true
        end
    end
end

mousemove_handler = @guarded (canvas, event) -> begin
    global clicked
    points[clicked] = [event.x, event.y]
    generate_curve()
    draw(canvas)
end

function refresh(canvas, generate = false)
    if current_file === nothing
        if generate
            generate_curve()
        end
        draw(canvas)
    else
        display_design(canvas, current_file)
    end
end

function setup_gui()
    clear_variables!()

    win = GtkWindow("ϵ-κ-curves")
    vbox = GtkBox(:v)

    # Canvas widget
    canvas = GtkCanvas(width, height)
    canvas.mouse.button1press = mousedown_handler
    canvas.mouse.button1motion = mousemove_handler
    draw(draw_callback, canvas)
    push!(win, vbox)
    push!(vbox, canvas)

    # Reset button
    reset = GtkButton("Start Over")
    signal_connect(reset, "clicked") do _
        clear_variables!()
        draw(canvas)
    end
    hbox = GtkBox(:h)
    hbox.spacing[Int] = 5
    push!(vbox, hbox)
    push!(hbox, reset)

    # Show controls checkbox
    controlsp = GtkCheckButton("Controls")
    controlsp.active[Bool] = show_controls
    signal_connect(controlsp, "toggled") do cb
        global show_controls = cb.active[Bool]
        refresh(canvas)
    end
    push!(hbox, controlsp)

    # Show curvature checkbox
    curvaturep = GtkCheckButton("Curvature")
    curvaturep.active[Bool] = show_curvature
    signal_connect(curvaturep, "toggled") do cb
        global show_curvature = cb.active[Bool]
        refresh(canvas)
    end
    push!(hbox, curvaturep)

    # Closed checkbox
    closedp = GtkCheckButton("Closed curve")
    closedp.active[Bool] = closed_curve
    signal_connect(closedp, "toggled") do cb
        global closed_curve = cb.active[Bool]
        refresh(canvas, true)
    end
    push!(hbox, closedp)

    # Cubic checkbox
    changetype = GtkComboBoxText()
    curve_types = ["Original", "Cubic", "Trig.basis"]
    foreach(t -> push!(changetype, t), curve_types)
    changetype.active[Int] = curve_type
    push!(hbox, changetype)

    hbox = GtkBox(:h)
    push!(vbox, hbox)

    # Alpha choices
    push!(hbox, GtkLabel("Alpha: "))
    global radios = [GtkRadioButton("N/A") for _ in 1:7]
    function alpha_handler(r)
        !r.active[Bool] && return
        if r.label[String] != "N/A"
            if isempty(selected)
                global alpha = float(eval(Meta.parse(r.label[String])))
                global selected_alpha = alpha
                for i in 1:length(radios)
                    rb = radios[i]
                    rb.label[String] == "N/A" && continue
                    if float(eval(Meta.parse(rb.label[String]))) == alpha
                        global alpha_index = i
                        break
                    end
                end
            else
                global selected_alpha = float(eval(Meta.parse(r.label[String])))
            end
        end
        refresh(canvas, true)
    end
    for r in radios
        r.group[GtkRadioButton] = radios[1]
        signal_connect(alpha_handler, r, "toggled")
        push!(hbox, r)
    end
    radios[1].active[Bool] = true

    # Update alphas based on curve type
    alpha_choices =
        [["N/A", "N/A", "N/A", "N/A", "N/A", "N/A", "N/A"],
         ["2/3", "0.7", "0.75", "0.8", "0.85", "0.9", "0.95"],
         ["0.55", "0.6", "0.65", "0.7", "0.75", "0.8", "0.85"]]
    function type_handler(ct)
        i = ct.active[Int]
        global curve_type = i
        ac = alpha_choices[i+1]
        for j in 1:length(ac)
            radios[j].label[String] = ac[j]
            if radios[j].active[Bool]
                alpha_handler(radios[j])
            end
        end
        refresh(canvas, true)
    end
    signal_connect(type_handler, changetype, "changed")
    type_handler(changetype)

    hbox = GtkBox(:h)
    hbox.spacing[Int] = 10
    push!(vbox, hbox)

    # Loading designs
    push!(hbox, GtkLabel("Select design:"))
    combo = GtkComboBoxText()
    foreach(d -> push!(combo, d), designs)
    combo.active[Int] = 0
    push!(hbox, combo)
    load = GtkButton("Load")
    signal_connect(load, "clicked") do _
        i = combo.active[Int] + 1
        display_design(canvas, "$(designs[i]).pts")
    end
    push!(hbox, load)

    # Just curve checkbox
    just = GtkCheckButton("Show only the curve")
    just.active[Bool] = just_curve
    signal_connect(just, "toggled") do cb
        global just_curve = cb.active[Bool]
        refresh(canvas)
    end
    push!(hbox, just)

    showall(win)
end

run() = begin setup_gui(); nothing end

end # module
