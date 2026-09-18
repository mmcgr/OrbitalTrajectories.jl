# Define a circle by radius r
circle(x, y, r) = ellipse_by_eccentricity(x, y, a = r, e = 0)

# Define an ellipse with: semi-major axis a, eccentricity e
ellipse_by_eccentricity(x, y; a, e) = ellipse_by_axis(x, y, a = a, b = √(a^2 * (1 - e^2)))

function ellipse_by_axis(x, y; a, b, num=500)
    # Define an ellipse with:
    #   semi-major axis a
    #   semi-minor axis b
    θ = LinRange(0, 2π, num)
    return @. (x + a * cos(θ), y + b * sin(θ))
end

function get_x_y(model::M) where { M <: Abstract_DynamicalModel }
    us = unknowns(model.ode)
    x = findfirst(isequal(model.x), us)
    y = findfirst(isequal(model.y), us)
    return (x, y)
end

xyz_to_idx(_, i::Int) = i
function xyz_to_idx(_, var::Symbol)
    if var == :x
        return 1
    elseif var == :y
        return 2
    elseif var == :z
        return 3
    else
        error("Unknown variable: $var")
    end
end

function xyz_to_idx(model::M, var::Num) where { M <: Abstract_DynamicalModel }
    if var == model.x
        return 1
    elseif var == model.y
        return 2
    elseif var == model.z
        return 3
    else
        error("Unknown variable: $var")
    end
end

@recipe function f(trajs::AbstractArray{<:Trajectory})
    trajs, trajs[1].frame
end

@recipe function f(trajs::AbstractArray{<:Trajectory}, frame::Abstract_ReferenceFrame)
    values = get(plotattributes, :values, nothing)
    plot_cbar = !isnothing(values) && length(values) == length(trajs)

    user_xlim = pop!(plotattributes, :xlims, nothing)
    user_ylim = pop!(plotattributes, :ylims, nothing)

    legend --> false # XXX: Legend seems to mess up the plot badly!

    # Work out the maximum extent of the orbit
    xlim = (Inf, -Inf)
    ylim = (Inf, -Inf)

    for (i, traj) in enumerate(trajs)
        current_xlim, current_ylim = get_margin_lims(convert_to_frame(traj, frame), plotattributes)
        if isnothing(user_xlim)
            xlim = (min(xlim[1], current_xlim[1]), max(xlim[2], current_xlim[2]))
        else
            xlim = user_xlim
        end
        if isnothing(user_ylim)
            ylim = (min(ylim[1], current_ylim[1]), max(ylim[2], current_ylim[2]))
        else
            ylim = user_ylim
        end

        @series begin
            label := false
            if plot_cbar
                seriescolor --> cgrad(:thermal)
                arrow --> false
                linewidth --> 3
                label --> false
                colorbar --> :left
                line_z --> values[i]
            else
                seriescolor --> :blue
                arrow --> true
                linewidth --> 1.5
                label --> "Trajectory"
            end
            xlims := xlim
            ylims := ylim
            if i < length(trajs)
                nomodel := true
            end
            traj, frame
        end
    end
end

# ------------------
# NEW STYLE PLOTTING
# ------------------

@recipe function f(traj::Trajectory)
    trace_vars = get(plotattributes, :trace, get(plotattributes, :trace_stability, false))
    if trace_vars === false
        # Plot the frame
        if !get(plotattributes, :nomodel, false)
            @series begin
                seriesalpha := 1.0
                (traj.model, traj.frame)
            end
            framestyle --> :zerolines
        else
            framestyle --> :none
        end

        # Plot the trajectory
        idxs --> get_x_y(traj.model)
        denseplot --> get(plotattributes, :denseplot, true)

        xlim, ylim = get_margin_lims(traj, plotattributes)
        xlims --> xlim
        ylims --> ylim
        
        # Formatting
        arrow --> true
        dpi --> 150
        size --> (400, 500)
        legend --> false
        xaxis --> (rotation=45)
        aspect_ratio --> 1
        ticks --> false
    end

    traj.sol
end

@recipe function f(sol::SciMLBase.ODESolution{T}) where {T <: ForwardDiff.Dual}
    # TODO: Fix the type dispatch on this, since it's doing piracy.

    trace_vars = get(plotattributes, :trace, false)
    trace_stability = get(plotattributes, :trace_stability, false)
    denseplot = get(plotattributes, :denseplot, true)
    plotdensity = get(plotattributes, :plotdensity, 1000)

    tspan = ForwardDiff.value.(denseplot ? range(sol.t[begin], sol.t[end], length=plotdensity) : sol.t)
    tspan_norm = @. (tspan - tspan[begin]) / (tspan[end] - tspan[begin])
    u_vals = sol.(tspan)

    if trace_vars
        STMs = hcat([reshape(v, length(v)) for v in extract_STMs(u_vals)]...)'
        @series begin
            label --> ""
            legend --> false
            tspan_norm, STMs
        end
    elseif trace_stability
        stability_indices = norm.(extract_stability(u_vals))'
        sorted_indices = hcat(map(sort, eachslice(stability_indices, dims=1))...)[4:6,:]'
        max_eigenvalues = map(maximum, eachslice(sorted_indices, dims=1))
        @series begin
            label --> ""
            legend --> false
            tspan_norm, max_eigenvalues
        end
    else
        vars = get(plotattributes, :vars, (3, 2))
        ([[u[v].value for u in u_vals] for v in vars]...,)
    end
end

@recipe function f(::Abstract_DynamicalModel, ::Abstract_ReferenceFrame)
    nothing
end

@recipe function f(model::Abstract_DynamicalModel, ::SynodicFrame)
    # User arguments
    nolabels = get(plotattributes, :nolabels, get(plotattributes, :nomodel, false))
    plot_libration = get(plotattributes, :libration_points, true)
    circ_props = R3BPSystemProperties(primary_body(model), secondary_body(model))

    vars = get(plotattributes, :idxs, get_x_y(model))
    idx1 = xyz_to_idx(model, vars[1])
    idx2 = xyz_to_idx(model, vars[2])

    @series begin
        seriestype := :shape
        seriescolor := get(plotattributes, :primary_color, :blue)
        linecolor := :black
        linewidth := 0.5
        label := nolabels ? "" : titlecase(String(primary_body(model)))
        line_z := nothing

        primary_pos = (-circ_props.μ, 0.0, 0.0)
        ellipse_by_axis(primary_pos[idx1], primary_pos[idx2];
                        a = ustrip(circ_props.R1[idx1] / circ_props.L),
                        b = ustrip(circ_props.R1[idx2] / circ_props.L))
    end

    @series begin
        seriestype := :shape
        seriescolor := get(plotattributes, :secondary_color, :brown)
        linecolor := :black
        linewidth := 0.5
        label := nolabels ? "" : titlecase(String(secondary_body(model)))
        line_z := nothing

        secondary_pos = (1 - circ_props.μ, 0., 0.)
        ellipse_by_axis(secondary_pos[idx1], secondary_pos[idx2];
                        a = ustrip(circ_props.R2[idx1] / circ_props.L),
                        b = ustrip(circ_props.R2[idx2] / circ_props.L))
    end

    @series begin
        seriestype := :hline
        seriescolor := :black
        linestyle := :dash
        linewidth := 0.75
        line_z := nothing
        label := ""
        [0]
    end

    if get(plotattributes, :origin_secondary, true)
        @series begin
            seriestype := :vline
            seriescolor := :black
            linestyle := :dash
            linewidth := 0.75
            label := nolabels ? "" : "Origin of $(titlecase(String(secondary_body(model))))"
            line_z := nothing
            if vars[1] == 1
                [1 - circ_props.μ]
            else
                [0.0]
            end
        end
    end

    @series begin
        # Find the libration points
        L = libration_points(circ_props)
        if !isnothing(L) && plot_libration
            seriestype := :scatter
            markercolor := :yellow
            markerstrokewidth := 1
            markerstrokecolor := :red
            markershape := :diamond
            markersize := 3
            label := nolabels ? "" : "Libration points"
            line_z := nothing
            [l[idx1] for l in L], [l[idx2] for l in L]
        else
            [], []
        end
    end

end

@recipe function f(traj::Trajectory, frame::Abstract_ReferenceFrame)
    convert_to_frame(traj, frame)
end

@recipe function f(state::State)
    seriestype := :scatter
    # TODO: Is this used, and should it respect :idxs?
    u0 = ordered_u0(state)
    ([u0[1]], [u0[2]])
end

@recipe function f(state::State, frame::Abstract_ReferenceFrame)
    @series begin
        state.model, frame
    end
    convert_to_frame(state, frame)
end

function get_margin_lims(sol::Trajectory, plotattributes)
    margins = get(plotattributes, :padding, 0.10)
    a, b = get(plotattributes, :idxs, get_x_y(sol.model))
    a = xyz_to_idx(sol.model, a)
    b = xyz_to_idx(sol.model, b)

    # Work out the maximum extent of the orbit
    x, y = (ForwardDiff.value.(sol.sol[a,:]), ForwardDiff.value.(sol.sol[b,:]))
    xlim = (minimum(x), maximum(x))
    ylim = (minimum(y), maximum(y))

    # Add some margins to the figure
    xdiff = margins * (xlim[2] - xlim[1])
    ydiff = margins * (ylim[2] - ylim[1])

    xlims = (xlim[1] - xdiff, xlim[2] + xdiff)
    ylims = (ylim[1] - ydiff, ylim[2] + ydiff)

    return xlims, ylims
end