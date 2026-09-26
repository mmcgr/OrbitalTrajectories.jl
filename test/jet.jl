using JET
using OrbitalTrajectories
using Test

@testset "Stale dependencies" begin
    JET.test_package(OrbitalTrajectories; target_modules=[OrbitalTrajectories])
end
