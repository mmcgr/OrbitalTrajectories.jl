#-----------------------#
# GENERIC MODEL METHODS #
#-----------------------#

using Symbolics

# wrap_code: perform common substring elimination to improve performance of the resulting models.
@doc """Generic constructor for a DynamicalModel's underlying ODEFunctions."""
function (T::Type{<:Abstract_ModelODEFunctions})(args...; kwargs...)
    @mtkcompile ode = System(T, args...; kwargs...)

    # Generate the functions
    # TODO: Add support for tgrad (need to define derivative(get_pos) for EphemerisNBP)
    ode_f = ODEFunction(ode; jac = true, tgrad = false, eval_expression = false, eval_module = @__MODULE__)
    ode_stm_f = STM_ODEFunction(ode, ode_f)
    return T(ode, ode_f, ode_stm_f)
end

@doc """Generic model function."""
(model::Abstract_DynamicalModel)(args...; kwargs...) = model.ode.ode_f(args...; kwargs...)

@traitdef HasJacobian{X}
@traitimpl HasJacobian{X} < - has_jacobian(X)
has_jacobian(X::Type{<:SciMLBase.ODEFunction}) = !isnothing(fieldtype(X, :jac))

@doc """
    Function to generate an ODE Function that computes the State Transition
    Matrix simultaneously with the given function f. This requires f to have
    a computable Jacobian function.
"""
@traitfn function STM_ODEFunction(ode::ModelingToolkit.AbstractSystem, f::::HasJacobian; kwargs...)
    # TODO: Output just the variational equations, join as a SplitODEProblem
    # TODO: Memoize this function! It's very slightly slow for EphemerisNBP

    # Convert Variable -> Operation
    iv = independent_variable(ode)
    dvs = unknowns(ode)
    params = parameters(ode)

    # NOTE: Depends on the Jacobian, corresponding to A(t) matrix (for the
    # State Transition Matrix) [Parker & Anderson 2014].
    @variables ϕ(iv)[1:length(dvs), 1:length(dvs)]
    D = Differential(iv)

    # Get the Jacobian matrix (A(t))
    A = invokelatest(f.jac, Num.(dvs), Num.(params), Num(iv))

    # The State Transition Matrix (STM) ODE function is defined as follows, including the N^2 Jacobian equations +
    # the N first-order equations of motion. [Koon 2011]
    # NOTE: the Differential is defined element-wise and flattened to a list.
    stm_eqs = collect(simplify.(D.(ϕ) .~ A * ϕ))

    # Create the ODE system and generate its functions
    @mtkcompile stm_ode = System(
        [equations(ode)..., stm_eqs...], # Append the ODE equations.
        iv,
        [dvs..., ϕ...],  # Append the STM and motion state variables
        params;
        name = :stm_ode
    )
    stm_f = ODEFunction(stm_ode; sparse = true, eval_expression = false, eval_module = @__MODULE__, kwargs...)
    return stm_f
end

#---------#
# DISPLAY #
#---------#
Base.show(io::IO, x::Abstract_DynamicalModel) = show(io, typeof(x))
Base.show(io::IO, x::Type{<:Abstract_DynamicalModel}) = print(io, nameof(x))
Base.show(io::IO, x::Abstract_ModelODEFunctions) = print(io, nameof(x))
ModelingToolkit.varmap_to_vars(model::Abstract_DynamicalModel, varmap) = ModelingToolkit.varmap_to_vars(varmap, parameters(model))
SciMLBase.isinplace(::Abstract_DynamicalModel) = true
SciMLBase.isinplace(f::Abstract_DynamicalModel, _) = isinplace(f)
function Base.getproperty(x::M, b::Symbol) where {M <: Abstract_DynamicalModel}
    if b === :ode
        return getfield(x, :ode)
    elseif b === :props
        return getfield(x, :props)
    end

    return getproperty(getfield(x, :ode), b)
end
function Base.getproperty(x::F, b::Symbol) where {F <: Abstract_ModelODEFunctions}
    if b === :ode_system
        return getfield(x, :ode_system)
    elseif b === :ode_f
        return getfield(x, :ode_f)
    elseif b === :ode_stm_f
        return getfield(x, :ode_stm_f)
    end

    return getproperty(getfield(x, :ode_system), b)
end

function Base.propertynames(x::M, private::Bool = false) where {M <: Abstract_DynamicalModel}
    ode_fields = propertynames(getfield(x, :ode), private)
    return (:ode, :props, ode_fields...)
end

function Base.propertynames(x::F, private::Bool = false) where {F <: Abstract_ModelODEFunctions}
    ode_system_fields = propertynames(getfield(x, :ode_system), private)
    return (:ode_system, :ode_f, :ode_stm_f, ode_system_fields...)
end

ModelingToolkit.get_systems(::Abstract_ModelODEFunctions) = []
ModelingToolkit.get_eqs(f::Abstract_ModelODEFunctions) = ModelingToolkit.get_eqs(f.ode_system)
ModelingToolkit.get_unknowns(f::Abstract_ModelODEFunctions) = ModelingToolkit.get_unknowns(f.ode_system)
ModelingToolkit.get_ps(f::Abstract_ModelODEFunctions) = ModelingToolkit.get_ps(f.ode_system)
Base.nameof(f::Abstract_ModelODEFunctions) = nameof(typeof(f))

# Given a state and a state array u0 ordered by the states uknowns, return u0 ordered by
# [x, y, z, Dx(x), Dx(y), Dx(z)]
to_base_order!(model::M, u0::AbstractArray) where {M <: Abstract_DynamicalModel} = invpermute!(u0, get_model_order(model))
to_base_order(model::M, u0::AbstractArray) where {M <: Abstract_DynamicalModel} = to_base_order!(model, copy(u0))

# Given a state and an array u0 ordered by [x, y, z, Dx(x), Dx(y), Dx(z)], return u0 ordered by
# the states unknowns
to_model_order!(model::M, xyz::AbstractArray) where {M <: Abstract_DynamicalModel} = permute!(xyz, get_model_order(model))
to_model_order(model::M, xyz::AbstractArray) where {M <: Abstract_DynamicalModel} = to_model_order!(model, copy(xyz))

function parameter_map(model::M) where {M <: Abstract_DynamicalModel}
    v = Vector{Pair{Num, Float64}}()
    # Check the simpler model parameters first
    isempty(parameters(model)) && return v
    # Generate k => v pairs for the parameter map
    for p in parameters(model.ode.ode_system)
        push!(v, p => getproperty(model.props, p.name))
    end

    return v
end

#----------#
# INCLUDES #
#----------#
include("models/3bp.jl")
include("models/4bp.jl")
include("models/nbp.jl")
