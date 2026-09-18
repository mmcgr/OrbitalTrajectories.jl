using OrbitalTrajectories

using Test
using LinearAlgebra: eigvals, norm

@testset "Pellegrini2016" begin
    # Values taken from Table 1 of [Pellegrini 2016, On the Computation and Accuracy of Trajectory State Transition Matrices]
    test_cases = [(
        system = (:Jupiter, :Europa),
        u0     = [-0.1017472008677258, 0., 0., 0., -0.01806028472285857, 0.],
        μ      = 2.528009215182033e-5,
        t_p    = 25.13898226959327,
        λ_max  = 2.468621114047195,
    ), (
        system = (:Jupiter, :Europa),
        u0     = [0.04867586089512202, 0., 0., 0., -0.09354853663949217, 0.],
        μ      = 2.528009215182033e-5,
        t_p    = 70.53945041512506,
        λ_max  = 2.804814246519340e7,
    ), (
        system = (:Earth, :Moon),
        u0     = [-0.013059020050628, 0., 0.07129515195874, 0., -0.526306975588415, 0.],
        μ      = 0.01215509906405700,
        t_p    = 2.517727406553485,
        λ_max  = 17.632688124231755,
    )]

    for case in test_cases
        cr3bp = CR3BP(case.system...; μ=case.μ)
        cr3bp_handcoded = HandcodedCR3BP(case.system...; μ=case.μ)
        
        # The u0 from the paper is given relative to secondary body, so convert here to primary body
        u0 = deepcopy(case.u0)
        u0[1] += (1 - cr3bp.props.μ)

        # Initial state
        state_cr3bp = State(cr3bp, SynodicFrame(), u0, (0., case.t_p))
        state_cr3bp_handcoded = State(cr3bp_handcoded, SynodicFrame(), u0, (0., case.t_p))

        # Compute STMs
        STM_AD = sensitivity(AD, state_cr3bp)
        STM_VE = sensitivity(VE, state_cr3bp)
        STM_VE_handcoded = sensitivity(VE, state_cr3bp_handcoded)

        # Compute the maximum eigenvalues
        λ_max_AD = maximum(norm.(eigvals(Matrix(STM_AD))))
        @test λ_max_AD ≈ case.λ_max rtol=1e-4
        λ_max_VE = maximum(norm.(eigvals(STM_VE)))
        @test λ_max_VE ≈ case.λ_max rtol=1e-4
        λ_max_VE_handcoded = maximum(norm.(eigvals(STM_VE_handcoded)))
        @test λ_max_VE_handcoded ≈ case.λ_max rtol=1e-4
    end
end

@testset "EphemerisNBP STM computation" begin
    u0 = [0.8574053516112442, 0., 0., 0., 0.47, 0.]
    system_nbp = EphemerisNBP(:earth, :moon)
    prob = State(system_nbp, SynodicFrame(), u0, (0., 3600.0*24))

    # Compute final STM
    STM_AD = sensitivity(AD, prob)
    STM_FD = sensitivity(FD, prob)
    @test STM_AD ≈ STM_FD rtol=1e-5

    # Compute STM trace
    STM_AD_trace = sensitivity_trace(AD, prob)
    @test extract_STMs([STM_AD_trace.sol[:,end]])[1] ≈ STM_AD rtol=1e-5
end

@testset "EphemerisNBP STM computation - absolute" begin
    function permute_sensitivity(state, a)
        p = OrbitalTrajectories.Dynamics._model_ordering(state.model)
        return a[p, p]
    end
    u0 = [0.8574053516112442, 0.0, 0.0, 0.0, 0.47, 0.0]
    tspan = (0.0, 3600.0*24)
    model = EphemerisNBP(:earth, :moon)
    state = State(model, SynodicFrame(), u0, tspan)

    expected_STM = [
        1.2993178575812105 -0.09155050962480367 3.5806458411312535e-6 0.22157811033467473 0.0364090338286375 -2.3919101958202677e-6;
        -0.12444211119304659 0.9029220039487235 1.8459510781214097e-6 -0.05190941389539204 0.19629131615085213 -1.921640109660043e-5;
        -3.627295641960973e-6 6.295871157840424e-6 0.8430793401887287 -3.4258974957029967e-6 1.989127540705744e-5 0.19752722314179738;
        2.623575673448739 -1.1639253839131734 -1.2388062473258362e-6 1.1676099459724318 0.2987851893173867 -3.736126012314366e-5;
        -1.702822368639892 -0.5018402795952146 0.00011082272558727774 -0.5709232945047221 0.8671342315792283 -0.0001796795877059251;
        -0.00015756415038688563 7.198068156630559e-5 -1.3847003822298243 -6.132670891463197e-5 0.0001982044012257495 0.8595117098688569
    ]
    # Compute final STM
    STM_AD = sensitivity(AD, state)
    STM_AD .= STM_AD[:, OrbitalTrajectories.Dynamics._model_ordering(state.model)]
    @test STM_AD ≈ expected_STM rtol=1e-5


    STM_FD = sensitivity(FD, state)
    STM_FD .= STM_FD[:, OrbitalTrajectories.Dynamics._model_ordering(state.model)]
    @test STM_FD ≈ expected_STM rtol=1e-5

    # Compute STM trace
    STM_AD_trace = sensitivity_trace(AD, state)
    extracted_STM = extract_STMs([STM_AD_trace.sol[:,end]])[1]
    STM_FD .= STM_FD[:, OrbitalTrajectories.Dynamics._model_ordering(state.model)]
    extracted_STM .= extracted_STM[:, OrbitalTrajectories.Dynamics._model_ordering(state.model)]
    @test extracted_STM ≈ expected_STM rtol=1e-5
end
