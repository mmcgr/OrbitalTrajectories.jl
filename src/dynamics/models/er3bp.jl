export ER3BP
export elliptical_potential, centrifugal_potential

#---------------------------#
# ER3BP EQUATIONS OF MOTION #
#---------------------------#
struct _ER3BP_ODEFunctions{S, F, F2} <: Abstract_ModelODEFunctions
    ode_system::S
    ode_f::F
    ode_stm_f::F2
end

function ModelingToolkitBase.System(::Type{_ER3BP_ODEFunctions}; name = :ER3BP)
    @parameters  μ  # Mass fraction
    @parameters  e  # Eccentricity
    @independent_variables  f  # True anomaly
    @variables   x(f) y(f) z(f)
    D, D2 = Differential(f), Differential(f)^2
    Dx, Dy, Dz = Differential.((x, y, z))

    # [DeiTos2017, Eqs. 14]
    # NOTE: need to expand the RHS derivatives before calling order-lowering
    ω = elliptical_potential(μ, (x, y, z), f, e)
    eqs = expand_derivatives.(
        [
            D2(x) ~ +2D(y) + Dx(ω),
            D2(y) ~ -2D(x) + Dy(ω),
            D2(z) ~ + Dz(ω),
        ]
    )
    system = System(
        eqs,
        f,
        [x, y, z],
        [μ, e];
        name
    )
    return system
end

@doc "Centrifugal potential [DeiTos2017, Eq.13; Ichinomiya 2018, Eq. 2.2]"
function centrifugal_potential(μ, (x, y, z))
    r = distance_vectors(μ, (x, y, z))
    return 0.5 * (x^2 + y^2) + (1 - μ) / r[1] + μ / r[2] + 0.5 * (1 - μ) * μ
end

@doc "Elliptical potential in barycentric frame [DeiTos2017, Eq. 15]"
function elliptical_potential(μ, (x, y, z), f, e)
    Ω_3 = centrifugal_potential(μ, (x, y, z))
    return (Ω_3 - 0.5 * z^2 * e * cos(f)) / (1 + e * cos(f))
end

# Build the equations at pre-compile time
const ER3BP_ODEFunctions = _ER3BP_ODEFunctions()

#-------------#
# ER3BP MODEL #
#-------------#
struct ER3BP{O <: _ER3BP_ODEFunctions, P <: R3BPSystemProperties} <: Abstract_R3BPModel
    ode::O
    props::P
end
ER3BP(a::Symbol, b::Symbol; kwargs...) = ER3BP(ER3BP_ODEFunctions, R3BPSystemProperties(a, b; kwargs...))
ER3BP(system::Abstract_DynamicalModel; kwargs...) = ER3BP(ER3BP_ODEFunctions, R3BPSystemProperties(system; kwargs...))

function _known_unknowns(model::M) where {M <: Abstract_R3BPModel}
    @independent_variables  f  # True anomaly
    @variables   x(f) y(f) z(f)
    D = Differential(f)
    eqs = equations(model.ode.ode_system)

    # map Dx, Dy, Dz to the corresponding simplified system eqs
    sDx = eqs[findfirst(e->isequal(e.lhs, D(x)), eqs)].rhs
    sDy = eqs[findfirst(e->isequal(e.lhs, D(y)), eqs)].rhs
    sDz = eqs[findfirst(e->isequal(e.lhs, D(z)), eqs)].rhs

    return Num[x, y, z, sDx, sDy, sDz]
end

function u0_input_map(model::M, u0::AbstractArray) where {M <: Abstract_R3BPModel}
    return Vector(_known_unknowns(model) .=> u0)
end

# Map the state vector [x, y, z, Dx(f), Dy(f), Dz(f)] to the model's u0 ordering.
# e.g. xyz_to_u0(model, [1, 2, 3, 4, 5, 6]) might return [4, 1, 5, 2, 6, 3]
function get_model_order(model::M) where {M <: Abstract_R3BPModel}
    uns = unknowns(model.ode.ode_system)
    knowns = _known_unknowns(model)
    return [findfirst(isequal(k), uns) for k in knowns]
end

# XXX: The parameters overload below gives us a performance boost.
ModelingToolkit.parameters(model::ER3BP) = SVector(model.props.μ, model.props.e)
