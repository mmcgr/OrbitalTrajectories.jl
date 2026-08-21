using Aqua
using Test

@testset "Stale dependencies" begin
    @test Aqua.test_stale_deps(OrbitalTrajectories)
end