using Aqua
using OrbitalTrajectories
using Test

@testset "Stale dependencies" begin
    Aqua.test_stale_deps(OrbitalTrajectories)
end
