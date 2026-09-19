using Test
using SafeTestsets
@testset "OrbitalTrajectories.jl" begin
    @safetestset "SPICE Utils" begin include("spice_utils.jl") end
    @safetestset "NBP" begin include("nbp.jl") end
    @safetestset "3BP" begin include("3bp.jl") end
    @safetestset "Document tests" begin include("doctests.jl") end
    @safetestset "Sensitivity" begin include("sensitivity.jl") end
    @safetestset "Reference Frames" begin include("reference_frames.jl") end
    @safetestset "Aqua" begin include("aqua.jl") end
end
