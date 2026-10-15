export State, Trajectory
export primary_body, secondary_body
export collision, check_distance, crashed

#------------------#
# ORBITAL PROBLEMS #
#------------------#

struct State{
        M <: Abstract_DynamicalModel,
        F <: Abstract_ReferenceFrame,
        uType,
        tType,
        isinplace,
        O <: SciMLBase.AbstractODEProblem{uType, tType, isinplace},
    } <: SciMLBase.AbstractODEProblem{uType, tType, isinplace}
    model::M
    frame::F  # Reference frame that the problem's u0 is defined in
    prob::O
end

State(model::Abstract_DynamicalModel, u0::AbstractArray, tspan) = State(model, default_reference_frame(model), u0, tspan)
State(model::Abstract_DynamicalModel, reference_frame::Abstract_ReferenceFrame, u0::AbstractArray{Float64}, tspan) =
    State(model, reference_frame, varmap(model, u0), tspan)
# Fallback for u0 arrays that are not Float64
State(model::Abstract_DynamicalModel, reference_frame::Abstract_ReferenceFrame, u0::AbstractArray, tspan) =
    State(model, reference_frame, ODEProblem(model, u0, tspan, parameters(model)))
State(model::Abstract_DynamicalModel, reference_frame::Abstract_ReferenceFrame, d::Vector{Pair{Num, Float64}}, tspan) =
    State(model, reference_frame, ODEProblem(model.ode.ode_system, d, tspan))

struct Trajectory{
        M <: Abstract_DynamicalModel,
        F <: Abstract_ReferenceFrame,
        T,
        N,
        A,
        O <: SciMLBase.AbstractTimeseriesSolution{T, N, A},
    } <: SciMLBase.AbstractTimeseriesSolution{T, N, A}
    model::M
    frame::F  # Reference frame that the solution is defined in
    sol::O
end

State(traj::Trajectory) = State(traj.model, traj.frame, traj.sol.prob)

primary_body(state::State) = primary_body(state.model)
primary_body(traj::Trajectory) = primary_body(traj.model)
secondary_body(state::State) = secondary_body(state.model)
secondary_body(traj::Trajectory) = secondary_body(traj.model)

SciMLBase.remake(state::State; model = state.model, frame = state.frame, kwargs...) =
    State(model, frame, remake(state.prob; kwargs...))
ModelingToolkit.parameters(state::State) = ModelingToolkit.parameters(state.model)

# Generate a map from expected unknowns using an initial state array.
# The input u0 is expected to be in the order [x, y, z, Dx(x), Dx(y), Dx(z)]
function varmap(model::M, u0::AbstractArray{Float64}) where {M <: Abstract_DynamicalModel}
    @assert length(u0) == length(unknowns(model.ode.ode_system))
    return vcat(u0_input_map(model, u0), parameter_map(model))
end

to_base_order(state::State, u0::AbstractArray) = to_base_order(state.model, u0)
to_base_order!(state::State, u0::AbstractArray) = to_base_order!(state.model, u0)

to_model_order(state::State, xyz::AbstractArray) = to_model_order(state.model, xyz)
to_model_order!(state::State, xyz::AbstractArray) = to_model_order!(state.model, xyz)

# Return u0 in the order [x, y, z, Dx(x), Dx(y), Dx(z)]
base_ordered_u0(state::State) = to_base_order(state, state.u0)

#---------------#
# INTERPOLATION #
#---------------#

# Interpolation
(traj::Trajectory)(t::Number) = State(traj.model, traj.frame, remake(traj.sol.prob; u0 = traj.sol(t), tspan = (t, t)))
function (traj::Trajectory)(t::AbstractArray{<:Number})
    new_sol = SciMLBase.build_solution(traj.prob, traj.alg, t, traj.sol(t); interp = traj.interp, retcode = traj.retcode)
    return Trajectory(traj.model, traj.frame, new_sol)
end

# Indexing
Base.getindex(state::State, idx...) = getindex(state.prob.u0, idx...)
Base.getindex(state::State, idx::Union{Tuple, AbstractArray}) = getindex(state.prob, idx)
Base.getindex(traj::Trajectory, idx::Int...) = State(traj.model, traj.frame, getindex(traj.sol, idx...), (traj.sol.t[idx...], traj.sol.t[idx...]))
Base.getindex(traj::Trajectory, idx::Int) = State(traj.model, traj.frame, remake(traj.sol.prob; u0 = getindex(traj.sol.u, idx), tspan = (traj.sol.t[idx], traj.sol.t[idx])))
Base.getindex(traj::Trajectory, idx::AbstractArray{Int}) = State(traj.model, traj.frame, getindex(traj.sol.u, idx), (traj.sol.t[idx], traj.sol.t[idx]))
Base.axes(state::State, idx...) = axes(state.prob, idx...)
Base.axes(traj::Trajectory, idx...) = axes(traj.sol, idx...)
Base.firstindex(traj::Trajectory) = firstindex(traj.sol.u)
Base.firstindex(traj::Trajectory, idx) = firstindex(traj.sol, idx)
Base.lastindex(traj::Trajectory) = lastindex(traj.sol.u)
Base.lastindex(traj::Trajectory, idx) = lastindex(traj.sol, idx)
Base.size(traj::Trajectory) = size(traj.sol)

function Base.getproperty(x::State, b::Symbol)
    if b === :model
        return getfield(x, :model)
    elseif b === :frame
        return getfield(x, :frame)
    elseif b === :prob
        return getfield(x, :prob)
    end

    prob = getfield(x, :prob)
    if b === :u0
        return getfield(prob, :u0)
    end

    return getproperty(prob, b)
end

function Base.getproperty(x::Trajectory, b::Symbol)
    if b === :model
        return getfield(x, :model)
    elseif b === :frame
        return getfield(x, :frame)
    elseif b === :sol
        return getfield(x, :sol)
    end

    sol = getfield(x, :sol)
    if b === :u
        return getfield(sol, :u)
    elseif b === :t
        return getfield(sol, :t)
    end

    return getproperty(sol, b)
end

function Base.propertynames(x::State, private::Bool = false)
    prob_fields = propertynames(getfield(x, :prob), private)
    return (:model, :frame, :prob, prob_fields...)
end

function Base.propertynames(x::Trajectory, private::Bool = false)
    sol_fields = propertynames(getfield(x, :sol), private)
    return (:model, :frame, :sol, sol_fields...)
end

#---------#
# DISPLAY #
#---------#

Base.show(io::IO, A::State) = println(io, summary(A))
Base.summary(state::State) = string(
    SciMLBase.TYPE_COLOR, nameof(typeof(state)), SciMLBase.NO_COLOR, " in ",
    SciMLBase.TYPE_COLOR, state.model, SciMLBase.NO_COLOR, " in ",
    SciMLBase.TYPE_COLOR, state.frame, "\n",
    "    ", SciMLBase.NO_COLOR, "Underlying ", summary(state.prob), "\n",
    "    ", SciMLBase.NO_COLOR, "tspan = ", state.prob.tspan, "\n",
    "    u0    = ", state.prob.u0
)
Base.show(io::IO, _::MIME"text/plain", A::Trajectory) = show(io, A) # XXX: Required because also defined in SciMLBase
function Base.show(io::IO, A::Trajectory)
    println(
        io, string(
            SciMLBase.TYPE_COLOR, nameof(typeof(A)), SciMLBase.NO_COLOR, " in ",
            SciMLBase.TYPE_COLOR, A.model, SciMLBase.NO_COLOR, " in ",
            SciMLBase.TYPE_COLOR, A.frame, SciMLBase.NO_COLOR
        )
    )
    return println(
        io, string(
            "    retcode  = $(A.retcode) [$(length(A.t)) timesteps]\n",
            "    t        = ($(A.t[begin]), $(A.t[end]))\n",
            "    u[begin] = $(A.u[begin])\n",
            "    u[end]   = $(A.u[end])\n"
        )
    )
end

#-------#
# SOLVE #
#-------#

const DEFAULT_ALG = Vern7();

function SciMLBase.solve(
        state::State,
        alg::OrdinaryDiffEqAlgorithm = DEFAULT_ALG;
        reltol = 1.0e-10,
        abstol = 1.0e-10,
        kwargs...
    )
    return SciMLBase.__solve(state, alg; reltol, abstol, kwargs...)
end

# The __solve() method does the actual heavy lifting, including converting to a Trajectory.
function SciMLBase.__solve(
        state::State,
        alg::OrdinaryDiffEqAlgorithm = DEFAULT_ALG;
        reltol = 1.0e-10,
        abstol = 1.0e-10,
        kwargs...
    )
    default_frame = default_reference_frame(state.model)
    real_state = convert_to_frame(state, default_frame)

    # Call the underlying solver
    raw_sol = solve(real_state.prob, alg; reltol, abstol, kwargs...)
    return Trajectory(state.model, default_frame, raw_sol)
end
