using OrbitalTrajectories
using OrbitalTrajectories.Dynamics: convert_to_frame, convert_u!, order_u0!, ordered_u0
using DynamicQuantities
using SciMLBase: solve

using Test

# Values taken from Table 1 of [Pellegrini 2016, On the Computation and Accuracy of Trajectory State Transition Matrices]
# The u0 from the paper is given relative to secondary body. Here it has been converted to primary body
test_cases = [(
    system = (:Jupiter, :Europa),
    u0     = [0.8982275190401223, 0.0, 0.0, 0.0, -0.01806028472285857, 0.0],
    μ      = 2.528009215182033e-5,
    tspan = (0.0, 25.13898226959327),
    λ_max  = 2.468621114047195,
), (
    system = (:Jupiter, :Europa),
    u0     = [1.0486505808029702, 0.0, 0.0, 0.0, -0.09354853663949217, 0.0],
    μ      = 2.528009215182033e-5,
    tspan  = (0.0, 70.53945041512506),
    λ_max  = 2.804814246519340e7,
), (
    system = (:Earth, :Moon),
    u0     = [0.974785880885315, 0.0, 0.07129515195874, 0.0, -0.526306975588415, 0.0],
    μ      = 0.01215509906405700,
    tspan  = (0.0, 2.517727406553485),
    λ_max  = 17.632688124231755,
)]

@testset "CR3BP" begin
    for case in test_cases
        cr3bp = CR3BP(case.system...; μ=case.μ)
        state = State(cr3bp, SynodicFrame(false), case.u0, case.tspan)
	
        @test case.u0 ≈ ordered_u0(state)

        state1 = convert_to_frame(state, SynodicFrame(false))
        @test state1 === state

        # Invalid
        # state2 = convert_to_frame(state, SynodicFrame(true))
        # expected_u0 = u0
        # @test expected_u0 ≈ state.u0

        state3 = convert_to_frame(state, InertialFrame())
        expected_u0 = copy(case.u0)
        @test expected_u0 ≈ ordered_u0(state)
    end
end

@testset "ER3BP" begin
    for case in test_cases
        er3bp = ER3BP(case.system...; μ=case.μ)
        state = State(er3bp, SynodicFrame(false), case.u0, case.tspan)
	
        @test case.u0 ≈ ordered_u0(state)

        state1 = convert_to_frame(state, SynodicFrame(false))
        @test state1 === state

        # Invalid
        # state2 = convert_to_frame(state, SynodicFrame(true))
        # expected_u0 = copy(case.u0)
        # @test expected_u0 ≈ state.u0

        state3 = convert_to_frame(state, InertialFrame())
        expected_u0 = copy(case.u0)
        @test expected_u0 ≈ ordered_u0(state)
    end
end

@testset "Ephemeris" begin
    for case in test_cases
        # The EphemerisNBP model requires a timespan in seconds, rather than radians.
        tspan_epoch = ustrip.(u"s", case.tspan .* R3BPSystemProperties(case.system...).T)
        eph = EphemerisNBP(case.system..., :Sun)
        state = State(eph, SynodicFrame(false), case.u0, tspan_epoch)
	
        @test case.u0 ≈ ordered_u0(state)

        state1 = convert_to_frame(state, SynodicFrame(false))
        @test state1 === state

        # Invalid
        state2 = convert_to_frame(state, SynodicFrame(true))
        expected_u0 = copy(case.u0)
        @test expected_u0 ≈ ordered_u0(state)

        state3 = convert_to_frame(state, InertialFrame())
        expected_u0 = copy(case.u0)
        @test expected_u0 ≈ ordered_u0(state)
    end
end

@testset "convert_to_frame(Trajectory)" begin
    case = first(test_cases)
    model = ER3BP(case.system...)
    state = State(model, SynodicFrame(true), case.u0, case.tspan)
    traj = solve(state)
    @test traj.sol.prob.u0 ≈ state.u0

    traj_st = convert_to_frame(traj, SynodicFrame(true))
    @test traj_st === traj
    positions = [
        (1, case.u0),
        (2, [0.8981308252677037, -0.00041042484954003153, 0.0, -0.008479036989339448, -0.017865020174065993, 0.0]),
        (20, [0.5855475189888679, 0.20889786432411458, 0.0, -0.5766929247032657, 0.4842745971159068, 0.0]),
    ]

    for (i, pos) in positions
        expected_u0 = order_u0!(traj.model, copy(pos))
        @test traj_st.sol[:,i] ≈ expected_u0 atol=1e-4
    end

    traj_i = convert_to_frame(traj, InertialFrame())
    positions = [
        (1, [0.8982275190401223, 0.0, 0.0, 0.0, 0.8801672343172637, 0.0]),
        (2, [0.8979066002253975, 0.02007199582183506, 0.0, -0.028141407045196427, 0.8798528577714169, 0.0]),
        (20, [-0.019070893263767097, 0.6214020564363437, 0.0, -1.259320136610032, -0.41926726731292824, 0.0])
    ]

    for (i, pos) in positions
        expected_u0 = order_u0!(traj_i.model, copy(pos))
        @test traj_i.sol[:,i] ≈ expected_u0 atol=1e-4
    end
end

@testset "convert_u!(AbstractArray, ER3BP)" begin
    case = first(test_cases)
    model = ER3BP(case.system...)
    state = State(model, SynodicFrame(true), case.u0, case.tspan)
    traj = solve(state)
    positions = [
        (0.0, [0.1, 0.2, 0.3, 0.4, 0.5, 0.6], [0.1, 0.2, 0.3, 0.2, 0.6, 0.6]),
        (1.0, [0.1, 0.2, 0.3, 0.4, 0.5, 0.6], [-0.11426396637476534, 0.19220755965441763, 0.3, -0.39682212971110997, 0.4924755804824632, 0.6]),
        (10.0, [0.1, 0.2, 0.3, 0.4, 0.5, 0.6], [0.024897069270228712, -0.22221641690422747, 0.3, 0.15859836071833136, -0.6122471396237454, 0.6]),
        (10.0, [1.0, 0.0, 0.0, 0.0, 0.0, 0.0], [-0.8390715290764524, -0.5440211108893698, 0.0, 0.5440211108893698, -0.8390715290764524, 0.0]),
    ]
    # Translate from x,y,z to the order used by the state
    for position in positions
        order_u0!(state, position[2])
        order_u0!(state, position[3])
    end
    for (t, u0, expected_newu0) in positions
        actual_newu0 = copy(u0)
        convert_u!(actual_newu0, t, traj, InertialFrame())
        @test expected_newu0 ≈ actual_newu0 atol=1e-4
    end
end

@testset "convert_u!(AbstractArray, CR3BP)" begin
    case = first(test_cases)
    model = CR3BP(case.system...)
    state = State(model, SynodicFrame(true), case.u0, case.tspan)
    traj = solve(state)
    positions = [
        (0.0, [0.1, 0.2, 0.3, 0.4, 0.5, 0.6], [0.1, 0.2, 0.3, 0.2, 0.6, 0.6]),
        (1.0, [0.1, 0.2, 0.3, 0.4, 0.5, 0.6], [-0.11426396637476534, 0.19220755965441763, 0.3, -0.39682212971110997, 0.4924755804824632, 0.6]),
        (10.0, [0.1, 0.2, 0.3, 0.4, 0.5, 0.6], [0.024897069270228712, -0.22221641690422747, 0.3, 0.15859836071833136, -0.6122471396237454, 0.6]),
        (10.0, [1.0, 0.0, 0.0, 0.0, 0.0, 0.0], [-0.8390715290764524, -0.5440211108893698, 0.0, 0.5440211108893698, -0.8390715290764524, 0.0]),
    ]
    # Translate from x,y,z to the order used by the state
    for position in positions
        order_u0!(state, position[2])
        order_u0!(state, position[3])
    end
    for (t, u0, expected_newu0) in positions
        actual_newu0 = copy(u0)
        convert_u!(actual_newu0, t, traj, InertialFrame())
        @test expected_newu0 ≈ actual_newu0 atol=1e-4
    end
end

@testset "convert_u!(AbstractArray, EphemerisNBP)" begin
    case = first(test_cases)
    model = EphemerisNBP(case.system..., :Sun)
    state = State(model, SynodicFrame(true), case.u0, case.tspan)
    traj = solve(state)
    positions = [
        (0.0, [0.1, 0.2, 0.3, 0.4, 0.5, 0.6], [-2.5580305665462433e-5, -1.8040494905762604e-7, 4.404802348301499e-7, -0.04232225315270789, -0.01565430653851563, 0.041825275533164986]),
        (1.0, [0.1, 0.2, 0.3, 0.4, 0.5, 0.6], [-2.5580309443995475e-5, -1.8039869609204107e-7, 4.404802577647932e-7, -0.04232277341782379, -0.015653370836976772, 0.04182527334413663]),
        (10.0, [0.1, 0.2, 0.3, 0.4, 0.5, 0.6], [-2.5580343447018095e-5, -1.8034241573347003e-7, 4.404804638822698e-7, -0.04232745498841439, -0.01564494921954967, 0.04182525365057605]),
        (10.0, [1.0, 0.0, 0.0, 0.0, 0.0, 0.0], [-2.654952753577966e-5, 8.063436554669776e-7, -2.161122424968398e-8, 0.0024943302431482215, 1.2694315268154432e-6, 3.3443184345053577e-12]),
    ]
    # Translate from x,y,z to the order used by the state
    for position in positions
        order_u0!(state, position[2])
#        order_u0!(state, position[3])
    end
    for (t, u0, expected_newu0) in positions
        actual_newu0 = copy(u0)
        convert_u!(actual_newu0, t, traj, SynodicFrame())
        @test expected_newu0 ≈ actual_newu0 atol=1e-4
    end
end
