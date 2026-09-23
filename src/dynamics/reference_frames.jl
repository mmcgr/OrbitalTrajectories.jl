export InertialFrame, SynodicFrame, convert_to_frame

#------------------#
# REFERENCE FRAMES #
#------------------#

struct InertialFrame <: Abstract_ReferenceFrame end
struct SynodicFrame{IsNormalised} <: Abstract_ReferenceFrame
    SynodicFrame(isnormalised::Bool=true) = new{isnormalised}()
end

default_reference_frame(model::Abstract_DynamicalModel) = error("$(model) has no default reference frame assigned!")
default_synodic_reference_frame(::Abstract_DynamicalModel) = SynodicFrame()

#----------------------------------#
# CONVERT BETWEEN REFERENCE FRAMES #
#----------------------------------#
convert_to_frame(state::State, frame::Abstract_ReferenceFrame) = error("Cannot convert $(nameof(typeof(state))){$(state.model),$(state.frame)} to $(frame).")
convert_to_frame(state::State{<:Abstract_DynamicalModel,T}, ::T) where {T<:Abstract_ReferenceFrame} = state
convert_to_frame(traj::Trajectory{<:Abstract_DynamicalModel,T}, ::T) where {T<:Abstract_ReferenceFrame} = traj
function convert_to_frame(traj::Trajectory, frame::F) where {F<:Abstract_ReferenceFrame}
    prob0 = traj.prob
    times = traj.t

    u1 = deepcopy(traj.u)
    convert_u!(u1, times, traj, frame)

    # TODO: Convert this into a type so that we can make an interp_summary(::interpType) dispatch (to print out converted sols)
    function interp_and_convert(times, idxs, deriv::Type{Val{0}}, p, continuity::Symbol=:left)
        # The interpolation needs to use all indices, even when only two will be displayed
        u = deepcopy(traj.interp(times, nothing, deriv, p, continuity))
        convert_u!(u, times, traj, frame)
        return u
    end

    new_sol = SciMLBase.build_solution(prob0, traj.alg, times, u1; interp=interp_and_convert, retcode=traj.retcode)

    return Trajectory(traj.model, frame, new_sol)
end

function convert_u!(u::AbstractArray{Arr, 1}, times::AbstractArray, traj::Trajectory, frame::Abstract_ReferenceFrame) where {Arr <: AbstractArray{<:Any, 1}}
    for (i, t) in enumerate(times)
        convert_u!(u[i], t, traj, frame)
    end
end
function convert_u!(
    u::AbstractArray{A, 2},
    times::AbstractArray,
    traj::Trajectory,
    frame::Abstract_ReferenceFrame) where {A <: Any}
    for (i, t) in enumerate(times)
        convert_u!(u[:,i], t, traj, frame)
    end
end
# Convert a single state vector to a different reference frame
# The state vector should be in the expected order for the given model
# The result is in the same order
function convert_u!(u::AbstractArray{A, 1}, t::Float64, traj::Trajectory, frame::Abstract_ReferenceFrame) where {A <: Any}
    prob1 = remake(traj.sol.prob; u0=u, tspan=(t, t))
    tmp_state = State(traj.model, traj.frame, prob1)
    new_state = convert_to_frame(tmp_state, frame)
    # TODO: return state order
    #u.= new_state.u0
    u .= ordered_u0(new_state)
end

const CRASHED_RETCODE = SciMLBase.ReturnCode.Terminated
function collision(system::Abstract_DynamicalModel, body, dist=bodvrd(String(body), "RADII")[1]; radii=1., interp_points=10)
    diam = radii * dist
    ContinuousCallback((integrator) -> terminate!(integrator, CRASHED_RETCODE); interp_points) do u, t, integrator
        check_distance(u, t, system, body, diam)
    end
end
check_distance(_, _, system::Abstract_DynamicalModel, _, _=nothing) = error("check_distance not defined for $(nameof(typeof(system)))")
crashed(sol::Trajectory) = crashed(sol.sol)
crashed(sol::ODESolution) = sol.retcode == CRASHED_RETCODE