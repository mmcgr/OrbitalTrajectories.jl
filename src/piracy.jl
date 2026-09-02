using ModelingToolkit
using DynamicQuantities
using LinearAlgebra
using ForwardDiff

# TODO: Check which of these are still in use and/or necessary.

# Avoid Plots compatibility issue with Unitful
# Base.occursin(s::Any, q::Unitful.Quantity) = occursin(s, repr(q))
# Base.pointer(q::Unitful.Quantity) = pointer(repr(q))
Base.occursin(s::Any, q::Quantity) = occursin(s, repr(q))
Base.pointer(q::Quantity) = pointer(repr(q))

# Ensure that ForwardDiff values can get sent to C calls
Base.unsafe_convert(T::Type{<:Any}, x::ForwardDiff.Dual) = T(ForwardDiff.value.(x))

# Compute norms for arrays of symbolic variables (as needed by EphemerisNBP)
LinearAlgebra.norm(a::AbstractArray{<:ModelingToolkit.Num}) = sum(a .^ 2)^(1/2)




# --------------------------------#
# Common Subexpression Evaluation #
# --------------------------------#
# TODO: Convert to use the SymbolicUtils cse implementation (the updated and merged version of the below)
# instead of the one below
using SymbolicUtils.Code: Let, Func, MakeArray, SetArray, (←), LiteralExpr, AtIndex
using Symbolics: Equation

# Code below copied from Shashi Gowda's work in https://github.com/JuliaSymbolics/SymbolicUtils.jl/pull/200
# Under the MIT License as per conditions below:

# Copyright (c) 2020: Shashi Gowda, Yingbo Ma, Mason Protter, Julia Computing.

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

### Code-related utilities

using SymbolicUtils

using SymbolicUtils: Sym, Term, iscall

using SymbolicUtils.Rewriters

using OrderedCollections: OrderedDict

### Common subexprssion evaluation

newsym() = Sym{Number}(gensym("cse"))

function _cse(expr, dict=OrderedDict())
    r = @rule ~x::iscall => haskey(dict, ~x) ? dict[~x] : dict[~x] = newsym()
    return Postwalk(Chain([r]))(expr)
end

function cse(expr)
    !iscall(expr) && return expr
    dict=OrderedDict()
    final = _cse(expr, dict)
    return Let([var ← ex for (ex, var) in pairs(dict)], final)
end

function _cse(exprs::AbstractArray)
    dict = OrderedDict()
    final = map(ex->_cse(ex, dict), exprs)
    return ([var ← ex for (ex, var) in pairs(dict)], final)
end

function cse(x::MakeArray)
    assigns, expr = _cse(x.elems)
    return Let(assigns, MakeArray(expr, x.similarto, x.output_eltype))
end

function cse(x::SetArray)
    assigns, expr = _cse(x.elems)
    return Let(assigns, SetArray(x.inbounds, x.arr, expr))
end
# --------------------------------#
# End of copyrighted code         #
# --------------------------------#

function _cse(eq::Num, dict)
    final = _cse(Symbolics.value(eq), dict)
    return Num(final)
end

function _cse(eq::AtIndex, dict)
    final = _cse(eq.elem, dict)
    return AtIndex(eq.i, final)
end

function _cse(eq::Equation, dict)
    lhs = _cse(eq.lhs, dict)
    rhs = _cse(eq.rhs, dict)
    return Equation(lhs, rhs)
end

function cse(x::LiteralExpr)
    # XXX: Assumes that the LiteralExpr contains a LineNumber as the first argument
    return LiteralExpr(quote
        $(cse.(x.ex.args[2:end])...)
    end)
end

function cse(x::Func)
    body = cse(x.body)
    return Func(x.args, x.kwargs, body)
end