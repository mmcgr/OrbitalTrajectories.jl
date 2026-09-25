using Aqua
using OrbitalTrajectories
using Test

@testset "Stale dependencies" begin
    Aqua.test_all(
        OrbitalTrajectories
    )
end
