using Aqua
using OrbitalTrajectories
using Test

@testset "Stale dependencies" begin
    #Aqua.test_stale_deps(OrbitalTrajectories)
    Aqua.test_all(
        OrbitalTrajectories;
        # TODO: Are the piracies still needed?
        # treat_as_own = [
        #     LinearAlgebra.norm,
        #     Base.occursin,
        #     Base.pointer,
        #     Base.unsafe_convert,
        # ],
    )
end
