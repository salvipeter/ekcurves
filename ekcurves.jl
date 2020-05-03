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
show_controls = false
show_curvature = true
just_curve = false
closed_curve = true
cubic_curve = false
alpha = 2/3
incremental = false

# Global variables
points = []
controls = []
curve = []
curvature = []
max_curvatures = []

# Designs
designs = ["bear", "deer", "elephant", "bird", "plane", "pumpkin", "rose"]


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
            t = map(i -> compute_parameter(c[i], points[i], alpha), 1:n)
        else
            t = map(i -> compute_parameter(c[i], points[i]), 1:n)
        end

        if iteration == maxiter
            max_error = maximum(1:n) do i
                cp = cubic ? create_cubic(c[i], alpha) : c[i]
                norm(bezier_eval(cp, t[i]) - points[i])
            end
            warn_for_convergence_failure && @warn "Did not converge - err: $max_error"
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

Computes the parameter where the cubic curve created from the given quadratic one
takes its largest curvature value.
`p` is a point to interpolate. Uses second-degree approximation by Taylor series.
"""
function compute_parameter(curve, p, alpha)
    # Maximal curvature parameter of the quadratic curve
    tmp = curve[1] - 2 * curve[2] + curve[3]
    t = dot(curve[1] - curve[2], tmp) / norm(tmp) ^ 2

    # Create the cubic curve and evaluate with 2 derivatives
    cpts = create_cubic(curve, alpha)
    der = bezier_eval(cpts, t, 2)

    # Taylor expansion around t, in Bernstein form
    taylor = [der[1],
              der[1] + der[2] / 2,
              der[1] + der[2] + der[3] / 2]

    # Reparameterization
    # (since the Taylor expansion is around t, the curve is now in [-t, 1-t])
    tmp1 = -t * (taylor[1] - 2 * taylor[2] + taylor[3])
    tmp2 = 2 * t * (taylor[1] - taylor[2]) - t * tmp1
    taylor += [tmp2, tmp1 + tmp2, 2 * tmp1 + tmp2]

    compute_parameter(taylor, p)
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
        im = mod1(i - 1, n)
        ip = mod1(i + 1, n)
        if closed || i > 1
            A[i,im] = (1 - λ[im]) * ((1 - t[i]) ^ 3 + 3 * (1 - t[i]) ^ 2 * t[i] * (1 - alpha))
        else
            fixed[1,:] -= c[1][1] * ((1 - t[i]) ^ 3 + 3 * (1 - t[i]) ^ 2 * t[i] * (1 - alpha))
        end
        if closed || i < n
            A[i,ip] = λ[i] * (t[i] ^ 3 + 3 * (1 - t[i]) * t[i] ^ 2 * (1 - alpha))
        else
            fixed[n,:] -= c[n][3] * (t[i] ^ 3 + 3 * (1 - t[i]) * t[i] ^ 2 * (1 - alpha))
        end
        if closed || 1 < i < n
            A[i,i] =
                λ[im] * ((1 - t[i]) ^ 3 + 3 * (1 - t[i]) ^ 2 * t[i] * (1 - alpha)) +
                (1 - λ[i]) * (t[i] ^ 3 + 3 * (1 - t[i]) * t[i] ^ 2 * (1 - alpha)) +
                alpha * 3 * (1 - t[i]) * t[i]
        elseif n == 1
            A[i,i] = alpha * 3 * (1 - t[i]) * t[i]
        elseif i == 1
            A[i,i] =
                (1 - λ[i]) * (t[i] ^ 3 + 3 * (1 - t[i]) * t[i] ^ 2 * (1 - alpha)) +
                alpha * 3 * (1 - t[i]) * t[i]
        else # i == n
            A[i,i] =
                λ[im] * ((1 - t[i]) ^ 3 + 3 * (1 - t[i]) ^ 2 * t[i] * (1 - alpha)) +
                alpha * 3 * (1 - t[i]) * t[i]
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


# I/O

function load_design(filename)
    open_curves = []
    closed_curves = []

    read_point(f) = map(s -> parse(Float64, s), split(readline(f)))
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

    if !incremental
        # White background
        Graphics.rectangle(ctx, 0, 0, Graphics.width(canvas), Graphics.height(canvas))
        Graphics.set_source_rgb(ctx, 1, 1, 1)
        Graphics.fill(ctx)
    end

    # Input polygon
    # Graphics.set_source_rgb(ctx, 1, 0, 0)
    # Graphics.set_line_width(ctx, 1.0)
    # draw_polygon(ctx, points, closed_curve)

    if !just_curve && show_controls
        # Control polygon
        Graphics.set_source_rgb(ctx, 1, 0, 1)
        Graphics.set_line_width(ctx, 2.0)
        draw_polygon(ctx, controls, closed_curve)
    end

    # Generated curve
    Graphics.set_source_rgb(ctx, 0.8, 0.3, 0)
    Graphics.set_line_width(ctx, 2.0)
    draw_polygon(ctx, curve, closed_curve)

    just_curve && return

    # Curvature comb
    if show_curvature
        Graphics.set_source_rgb(ctx, 0, 0, 1)
        Graphics.set_line_width(ctx, 1.0)
        draw_polygon(ctx, curvature, closed_curve)
        for i in 1:length(curvature)
            i % curvature_sparsity != 1 && i != length(curvature) && continue
            Graphics.new_path(ctx)
            Graphics.move_to(ctx, curve[i][1], curve[i][2])
            Graphics.line_to(ctx, curvature[i][1], curvature[i][2])
            Graphics.stroke(ctx)
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
    for p in points[1:end]
        if show_curvature
            Graphics.set_source_rgb(ctx, 0, 1, 0)
            Graphics.arc(ctx, p[1], p[2], point_size, 0, 2pi)
            Graphics.fill(ctx)
        end
        Graphics.set_source_rgb(ctx, 0, 0, 0)
        Graphics.set_line_width(ctx, 1.0)
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
end

function generate_curve()
    if length(points) < 3
        if length(points) == 2
            global controls = [points[1], (points[1] + points[2]) / 2, points[2]]
            global curve = [points[1], points[2]]
            global curvature = []
        end
        return
    end
    cpts, t = interpolate(points, closed=closed_curve, cubic=cubic_curve, alpha=alpha)
    if cubic_curve
        cpts = map(c -> create_cubic(c, alpha), cpts)
    end
    global controls = vcat(cpts...)
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
        draw(canvas)
    end
    push!(hbox, controlsp)

    # Show curvature checkbox
    curvaturep = GtkCheckButton("Curvature")
    curvaturep.active[Bool] = show_curvature
    signal_connect(curvaturep, "toggled") do cb
        global show_curvature = cb.active[Bool]
        draw(canvas)
    end
    push!(hbox, curvaturep)

    # Closed checkbox
    closedp = GtkCheckButton("Closed curve")
    closedp.active[Bool] = closed_curve
    signal_connect(closedp, "toggled") do cb
        global closed_curve = cb.active[Bool]
        generate_curve()
        draw(canvas)
    end
    push!(hbox, closedp)

    # Cubic checkbox
    cubicp = GtkCheckButton("Cubic")
    cubicp.active[Bool] = cubic_curve
    signal_connect(cubicp, "toggled") do cb
        global cubic_curve = cb.active[Bool]
        generate_curve()
        draw(canvas)
    end
    push!(hbox, cubicp)

    hbox = GtkBox(:h)
    push!(vbox, hbox)

    # Alpha choices
    push!(hbox, GtkLabel("Alpha: "))
    choices = ["2/3", "0.7", "0.75", "0.8", "0.85", "0.9", "0.95"]
    choices_float = [2/3, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
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
    radios[1].active[Bool] = true

    hbox = GtkBox(:h)
    hbox.spacing[Int] = 10
    push!(vbox, hbox)

    # Loading designs
    push!(hbox, GtkLabel("Select design:"))
    combo = GtkComboBoxText()
    foreach(d -> push!(combo, d), designs)
    combo.active[Int] = 1
    push!(hbox, combo)
    load = GtkButton("Load")
    signal_connect(load, "clicked") do _
        i = combo.active[Int] + 1
        open_curves, closed_curves = load_design("$(designs[i]).pts")
        old_closed = closed_curve
        clear_variables!()
        draw(canvas)
        global incremental = true
        global closed_curve = true
        for c in closed_curves
            global points = c
            generate_curve()
            draw(canvas)
        end
        global closed_curve = false
        for c in open_curves
            global points = c
            generate_curve()
            draw(canvas)
        end
        clear_variables!()
        global closed_curve = old_closed
        global incremental = false
    end
    push!(hbox, load)

    # Just curve checkbox
    just = GtkCheckButton("Show only the curve")
    just.active[Bool] = just_curve
    signal_connect(just, "toggled") do cb
        global just_curve = cb.active[Bool]
        draw(canvas)
    end
    push!(hbox, just)

    generate_curve()
    showall(win)
end

run() = begin setup_gui(); nothing end

end # module
