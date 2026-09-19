using Symbolics: Num
using DynamicQuantities: Quantity
using ForwardDiff: ForwardDiff, Dual

# TODO: Check which of these are still in use and/or necessary.

# Avoid Plots compatibility issue with Quantify
# Base.occursin(s::Any, q::Quantity) = occursin(s, repr(q))
# Base.pointer(q::Quantity) = pointer(repr(q))

# # Ensure that ForwardDiff values can get sent to C calls
# # Base.unsafe_convert(T::Type{<:Any}, x::Dual) = T(ForwardDiff.value.(x))
# Base.unsafe_convert(::Type{Cwstring}, x::Dual) = Cwstring(ForwardDiff.value.(x))
