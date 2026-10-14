using OrbitalTrajectories
using OrbitalTrajectories.Dynamics: base_ordered_u0

using Test
using DifferentialEquations
# using BenchmarkTools

@testset "convert_to_frame" begin
    u0 = [0.8574053516112442, 0.0, 0.0, 0.0, 0.47, 0.0]
    system_nbp = EphemerisNBP(:earth, :moon)
    prob = State(system_nbp, SynodicFrame(), u0, (0.0, 3600.0 * 24 * 30))

    # Test that converting to and from inertial frames works
    inert_prob = convert_to_frame(prob, InertialFrame())
    synod_prob = convert_to_frame(inert_prob, SynodicFrame())
    @test base_ordered_u0(synod_prob) ≈ base_ordered_u0(prob)

    # Test that converting to/from non-normalised frames also works
    synod_nonorm_prob = convert_to_frame(prob, SynodicFrame(false))
    synod_norm_prob = convert_to_frame(synod_nonorm_prob, SynodicFrame(true))
    @test base_ordered_u0(synod_norm_prob) ≈ base_ordered_u0(prob)
    inert_prob2 = convert_to_frame(synod_nonorm_prob, InertialFrame())
    synod_nonorm_prob2 = convert_to_frame(inert_prob2, SynodicFrame(false))
    @test base_ordered_u0(synod_nonorm_prob2) ≈ base_ordered_u0(synod_nonorm_prob)

    # Roughly in decreasing order of allocations (30μs~40μs, allocations 8 to 38)
    # @btime convert_to_frame(prob, InertialFrame())  # norm -> inert
    # @btime convert_to_frame(inert_prob, SynodicFrame())  # inert -> norm
    # @btime convert_to_frame(prob, SynodicFrame(false))  # norm -> unorm
    # @btime convert_to_frame(synod_nonorm_prob, SynodicFrame(true))  # unorm -> norm
    # @btime convert_to_frame(synod_nonorm_prob, InertialFrame())  # unorm -> inert
    # @btime convert_to_frame(inert_prob2, SynodicFrame(false))  # inert -> unorm

    # Solve and test that converting the solution to synodic also matches
    sol = solve(prob)
    sol2 = convert_to_frame(sol, SynodicFrame())
    u0 = base_ordered_u0(prob)
    @test sol2.sol.u[begin] ≈ u0
end
