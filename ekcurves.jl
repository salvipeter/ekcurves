module EKCurves

using LinearAlgebra
using Polynomials: Polynomial, roots

using Gtk
import Graphics


# Parameters
maxiter = 1000
distance_tolerance = 1e-4
warn_for_convergence_failure = false

# GUI parameters
width = 500
height = 500
resolution = 50
curvature_scaling = 2000
curvature_sparsity = 10
point_size = 5
rectangle_scale = 1.4

# GUI variables
closed_curve = true
quartic_curve = false
alpha = 2/3

# Global variables
points = []
curve = []
curvature = []
max_curvatures = []


# Interpolation

"""
    interpolate(points; closed = true)

Interpolates an (ϵ)-κ-curve on the given points.
The result is given as `(c, t)`, where `c` contains the curves (as a vector of control points),
and `t` are the parameters of the interpolated points.
"""
function interpolate(points; closed = true)
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
        t = map(i -> compute_parameter(c[i], points[i]), 1:n)

        if iteration == maxiter
            max_error = maximum(i -> norm(bezier_eval(c[i], t[i]) - points[i]), 1:n)
            warn_for_convergence_failure && @warn "Did not converge - err: $max_error"
            break
        end

        x = compute_central_cps(c, λ, t, rhs, closed)
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
        tmp = sqrt(abs(Δ(c[i][1], c[i][2], c[ip][2])))
        denom = (tmp + sqrt(abs(Δ(c[i][2], c[ip][2], c[ip][3]))))
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

Computes the parameter where the given quadratic curve takes its largest curvature value.
`p` is a point to interpolate.
"""
function compute_parameter(curve, p)
    a = [-norm(curve[1] - p) ^ 2,
         dot(3 * curve[1] - 2 * p - curve[3], curve[1] - p),
         3 * dot(curve[3] - curve[1], curve[1] - p),
         norm(curve[3] - curve[1]) ^ 2]
    solve_cubic(a)
end

"""
    solve_cubic(a)

Solves the cubic equation, and chooses a real solution in `[0, 1]`.
The coefficients are given in the order `[1, x, x^2, x^3]`.
"""
function solve_cubic(coeffs)
    ϵ = 1e-8
    p = Polynomial(coeffs)
    x = roots(p)
    for xi in x
        abs(imag(xi)) < ϵ && -ϵ <= real(xi) <= 1 + ϵ && return clamp(real(xi), 0, 1)
    end
    @error "No suitable solution"
end

"""
    compute_central_cps(c, λ, t, points, closed)

Computes the central control points of the qudratic Bezier curves `c`
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
    mapreduce(k -> curve[k] * coeff[k], +, 1:n+1)
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
        push!(result, mapreduce(j -> dcp[k+1][j] * coeff[n-k+1][j], +, 1:n-k+1))
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

    # Input polygon
    # Graphics.set_source_rgb(ctx, 1, 0, 0)
    # Graphics.set_line_width(ctx, 1.0)
    # draw_polygon(ctx, points, closed_curve)

    # Generated curve
    Graphics.set_source_rgb(ctx, 0.8, 0.3, 0)
    Graphics.set_line_width(ctx, 2.0)
    draw_polygon(ctx, curve, closed_curve)

    # Curvature comb
    Graphics.set_source_rgb(ctx, 0, 0, 1)
    Graphics.set_line_width(ctx, 1.0)
    draw_polygon(ctx, curvature, closed_curve)
    for i in 1:length(curvature)
        i % curvature_sparsity != 0 && continue
        Graphics.new_path(ctx)
        Graphics.move_to(ctx, curve[i][1], curve[i][2])
        Graphics.line_to(ctx, curvature[i][1], curvature[i][2])
        Graphics.stroke(ctx)
    end

    # Maximum curvature points
    for p in max_curvatures[1:end]
        Graphics.set_source_rgb(ctx, 1, 0, 0)
        Graphics.arc(ctx, p[1], p[2], point_size, 0, 2pi)
        Graphics.fill(ctx)
    end

    # Input points
    for p in points[1:end]
        Graphics.set_source_rgb(ctx, 0, 1, 0)
        Graphics.arc(ctx, p[1], p[2], point_size, 0, 2pi)
        Graphics.fill(ctx)
        Graphics.set_source_rgb(ctx, 0, 0, 0)
        rect = [p + [-point_size, -point_size] * rectangle_scale,
                p + [-point_size,  point_size] * rectangle_scale,
                p + [ point_size,  point_size] * rectangle_scale,
                p + [ point_size, -point_size] * rectangle_scale]
        draw_polygon(ctx, rect, true)
    end
end


# GUI

function generate_curve()
    length(points) < 3 && return
    cpts, t = interpolate(points, closed=closed_curve)
    global curve = []
    global curvature = []
    for c in cpts
        for u in range(0, stop=1, length=resolution) # endpoints are drawn twice
            der = bezier_eval(c, u, 1)
            p = der[1]
            n = normalize([der[2][2], -der[2][1]])
            k = bezier_curvature(c, u)
            push!(curve, p)
            push!(curvature, p + n * k * curvature_scaling)
        end
    end
    global max_curvatures = map(i -> bezier_eval(cpts[i], t[i]), 1:length(t))
end

mousedown_handler = @guarded (canvas, event) -> begin
    p = [event.x, event.y]
    global clicked = findfirst(points) do q
        norm(p - q) < 10
    end
    if clicked === nothing
        push!(points, p)
        clicked = length(points)
        generate_curve()
        draw(canvas)
    end
end

mousemove_handler = @guarded (canvas, event) -> begin
    global clicked
    points[clicked] = [event.x, event.y]
    generate_curve()
    draw(canvas)
end

function setup_gui()
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
        global points = []
        global curve = []
        global curvature = []
        global max_curvatures = []
        draw(canvas)
    end
    hbox = GtkBox(:h)
    hbox.spacing[Int] = 10
    push!(vbox, hbox)
    push!(hbox, reset)

    # Closed checkbox
    closedp = GtkCheckButton("Closed curve")
    closedp.active[Bool] = closed_curve
    signal_connect(closedp, "toggled") do cb
        global closed_curve = cb.active[Bool]
        generate_curve()
        draw(canvas)
    end
    push!(hbox, closedp)

    # Quartic checkbox
    quarticp = GtkCheckButton("Quartic")
    quarticp.active[Bool] = quartic_curve
    signal_connect(quarticp, "toggled") do cb
        global quartic_curve = cb.active[Bool]
        generate_curve()
        draw(canvas)
    end
    push!(hbox, quarticp)

    hbox = GtkBox(:h)
    push!(vbox, hbox)

    # Alpha slider
    push!(hbox, GtkLabel("Alpha: "))
    choices = ["0.1", "0.2", "1/3", "0.5", "2/3", "0.8", "0.9"]
    choices_float = [0.1, 0.2, 1/3, 0.5, 2/3, 0.8, 0.9]
    radios = [GtkRadioButton(choice) for choice in choices]
    for r in radios
        r.group[GtkRadioButton] = radios[1]
        signal_connect(r, "toggled") do _
            !r.active[Bool] && return
            global alpha = choices_float[findfirst(rb -> rb === r, radios)]
            generate_curve()
            draw(canvas)
        end
        push!(hbox, r)
    end
    radios[5].active[Bool] = true

    generate_curve()
    showall(win)
end

run() = begin setup_gui(); nothing end

end # module
