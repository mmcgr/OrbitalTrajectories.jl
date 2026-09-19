export State, Trajectory
export primary_body, secondary_body
export collision, check_distance, crashed

#------------------#
# ORBITAL PROBLEMS #
#------------------#

function _unknowns_default_ordering(model::M) where M<:Abstract_DynamicalModel
    iv = independent_variable(model.ode.ode_system)
    D = Differential(iv)

    us = [model.x, model.y, model.z, D(model.x), D(model.y), D(model.z)]
    return us
end

# Generate a map from expected unknowns using an initial state array.
# The input u0 is expected to be in the order [x, y, z, Dx(x), Dx(y), Dx(z)]
function u0_map(model::M, u0::AbstractArray{Float64}) where M<:Abstract_DynamicalModel
    us = _unknowns_default_ordering(model)
    d = Dict{Num, Float64}(@. (us .=> u0))

    merge!(d, parameter_map(model))

    return d
end

struct State{
    M<:Abstract_DynamicalModel,
    F<:Abstract_ReferenceFrame,
    uType,
    tType,
    isinplace,
    O<:SciMLBase.AbstractODEProblem{uType, tType, isinplace}
} <: SciMLBase.AbstractODEProblem{uType, tType, isinplace}
    model :: M
    frame :: F  # Reference frame that the problem's u0 is defined in
    prob :: O
end

State(model::Abstract_DynamicalModel, u0::AbstractArray, tspan) = State(model, default_reference_frame(model), u0, tspan)
State(model::Abstract_DynamicalModel, reference_frame::Abstract_ReferenceFrame, u0::AbstractArray{Float64}, tspan) =
    State(model, reference_frame, u0_map(model, u0), tspan)
# Fallback for u0 arrays that are not Float64
State(model::Abstract_DynamicalModel, reference_frame::Abstract_ReferenceFrame, u0::AbstractArray, tspan) =
    State(model, reference_frame, ODEProblem(model, u0, tspan, parameters(model)))
# TODO: Convince ODEProblem to work better with DynamicModel + dict.
State(model::Abstract_DynamicalModel, reference_frame::Abstract_ReferenceFrame, d::Dict{Num, Float64}, tspan) =
    State(model, reference_frame, ODEProblem(model.ode.ode_system, d, tspan))

struct Trajectory{M<:Abstract_DynamicalModel,F<:Abstract_ReferenceFrame,T,N,A,O<:SciMLBase.AbstractTimeseriesSolution{T,N,A},} <: SciMLBase.AbstractTimeseriesSolution{T,N,A}
    model :: M
    frame :: F  # Reference frame that the solution is defined in
    sol :: O
end

State(traj::Trajectory) = State(traj.model, traj.frame, traj.sol.prob)

primary_body(state::State) = primary_body(state.model)
primary_body(traj::Trajectory) = primary_body(traj.model)
secondary_body(state::State) = secondary_body(state.model)
secondary_body(traj::Trajectory) = secondary_body(traj.model)

SciMLBase.remake(state::State; kwargs...) = State(state.model, state.frame, remake(state.prob; kwargs...))
ModelingToolkit.parameters(state::State) = ModelingToolkit.parameters(state.model)

# Given a state and a state array u0 ordered by the states uknowns, return u0 ordered by
# [x, y, z, Dx(x), Dx(y), Dx(z)]
function order_u0(state::State, u0::AbstractArray)
    return order_u0!(state, copy(u0))
end

function _expected_vars(sys::M) where M<:Abstract_DynamicalModel
    sys = sys.ode.ode_system
    return [sys.x, sys.y, sys.z, sys.xˍf, sys.yˍf, sys.zˍf]
end

# Given a state and a state array u0 ordered by the states unknowns, return u0 ordered by
# [x, y, z, Dx(x), Dx(y), Dx(z)]
order_u0!(state::State, u0::AbstractArray) =
    order_u0!(state.model, u0)

function order_u0!(model::M, u0::AbstractArray) where {M<:Abstract_DynamicalModel}
    sys = model.ode.ode_system
    expected_us = _expected_vars(model)
    # expected_us = _unknowns_default_orderinging(model)
    ordering = [findfirst(uu->isequal(u, uu), unknowns(sys)) for u in expected_us]
    permute!(u0, ordering)
    return u0
end

# Given a state and a state array u0 ordered by the states unknowns, return u0 invordered by
# [x, y, z, Dx(x), Dx(y), Dx(z)]
invorder_u0!(state::State, u0::AbstractArray) =
    invorder_u0!(state.model, u0)

function invorder_u0!(model::M, u0::AbstractArray) where {M <: Abstract_DynamicalModel}
    sys = model.ode.ode_system
    expected_us = _expected_vars(model)
    # expected_us = _unknowns_default_orderinging(model)
    ordering = [findfirst(uu->isequal(u, uu), unknowns(sys)) for u in expected_us]
    invpermute!(u0, ordering)
    return u0
end

# Return u0 in the order [x, y, z, Dx(x), Dx(y), Dx(z)]
function ordered_u0(state::State)
    return order_u0(state, state.prob.u0)
end

#---------------#
# INTERPOLATION #
#---------------#

# Interpolation
(traj::Trajectory)(t::Number) = State(traj.model, traj.frame, traj.sol(t), (t, t))
function (traj::Trajectory)(t::AbstractArray{<:Number})
    new_sol = SciMLBase.build_solution(traj.prob, traj.alg, t, traj.sol(t); interp=traj.interp, retcode=traj.retcode)
    return Trajectory(traj.model, traj.frame, new_sol)
end

# Indexing
Base.getindex(state::State, idx...) = getindex(state.prob.u0, idx...)
Base.getindex(state::State, idx::Union{Tuple, AbstractArray}) = getindex(state.prob, idx)
Base.getindex(traj::Trajectory, idx::Int...) = State(traj.model, traj.frame, getindex(traj.sol, idx...), (traj.sol.t[idx...], traj.sol.t[idx...]))
Base.getindex(traj::Trajectory, idx::Int) = State(traj.model, traj.frame, getindex(traj.sol.u, idx), (traj.sol.t[idx], traj.sol.t[idx]))
Base.getindex(traj::Trajectory, idx::AbstractArray{Int}) = State(traj.model, traj.frame, getindex(traj.sol.u, idx), (traj.sol.t[idx], traj.sol.t[idx]))
Base.axes(state::State, idx...) = axes(state.prob, idx...)
Base.axes(traj::Trajectory, idx...) = axes(traj.sol, idx...)
Base.firstindex(traj::Trajectory) = firstindex(traj.sol.u)
Base.firstindex(traj::Trajectory, idx) = firstindex(traj.sol, idx)
Base.lastindex(traj::Trajectory) = lastindex(traj.sol.u)
Base.lastindex(traj::Trajectory, idx) = lastindex(traj.sol, idx)
Base.size(traj::Trajectory) = size(traj.sol)

function Base.getproperty(x::State, b::Symbol)
    if hasfield(State, b)
        return getfield(x, b)
    end
    return getproperty(x.prob, b)
end

function Base.getproperty(x::Trajectory, b::Symbol)
    if hasfield(Trajectory, b)
        return getfield(x, b)
    end
    return getproperty(x.sol, b)
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
    "    u0    = ", state.prob.u0)
Base.show(io::IO, _::MIME"text/plain", A::Trajectory) = show(io, A) # XXX: Required because also defined in SciMLBase
function Base.show(io::IO, A::Trajectory)
    println(io, string(
        SciMLBase.TYPE_COLOR, nameof(typeof(A)), SciMLBase.NO_COLOR, " in ",
        SciMLBase.TYPE_COLOR, A.model, SciMLBase.NO_COLOR, " in ",
        SciMLBase.TYPE_COLOR, A.frame, SciMLBase.NO_COLOR))
    println(io, string(
        "    retcode  = $(A.retcode) [$(length(A.t)) timesteps]\n",
        "    t        = ($(A.t[begin]), $(A.t[end]))\n",
        "    u[begin] = $(A.u[begin])\n",
        "    u[end]   = $(A.u[end])\n"
    ))
end

#-------#
# SOLVE #
#-------#

const DEFAULT_ALG = Vern7();

function SciMLBase.solve(state::State, alg::OrdinaryDiffEqAlgorithm=DEFAULT_ALG; reltol=1e-10, abstol=1e-10, kwargs...)
    return SciMLBase.__solve(state, alg; reltol, abstol, kwargs...)
end

# The __solve() method does the actual heavy lifting, including converting to a Trajectory.
function SciMLBase.__solve(state::State, alg::OrdinaryDiffEqAlgorithm=DEFAULT_ALG; reltol=1e-10, abstol=1e-10, userdata=Dict(), callback=nothing, kwargs...)
    default_frame = default_reference_frame(state.model)
    real_state = convert_to_frame(state, default_frame)

    # Pass the default state into the underlying solver
    # TODO: Remove the need for this in DiffCorrectAxisymmetric
    # TODO: Removed to avoid deprecation warning in SciMLBase.solve()
    #  - buuut presumably DiffCorrectAxisymmetric is now sad
    # userdata = deepcopy(userdata)
    # userdata[:real_state] = real_state

    # Copy the callbacks (for thread-safety)
    callback = deepcopy(callback)

    # Call the underlying solver
    raw_sol = solve(real_state.prob, alg; reltol, abstol, callback, kwargs...)
    return Trajectory(state.model, default_frame, raw_sol)
end