"""
A set of convenience wrapper functions for AST creation.
"""
#pylint: disable=protected-access
import numpy as np
from . import ast

def const(value):
    ## Convenience function for creating Number, Vector or MatrixConstants.
    if np.isscalar(value):
        return ast.NumberConstant(value)
    elif hasattr(value, 'ndim'):
        if value.ndim == 0:
            return ast.NumberConstant(value.item())
        if value.ndim == 1:
            return ast.VectorConstant(value)
        elif value.ndim == 2:
            return ast.MatrixConstant(value)
        else:
            raise ValueError("Only one or two-dimensional vectors are supported")
    elif hasattr(value, '__iter__') or hasattr(value, '__next__'):
        return const(np.asarray(value))
    else:
        raise ValueError("Invalid constant: {0}".format(value))

class _FuncCreator:
    ## An object similar to sqlalchemy.func
    def __getattr__(self, attrname):
        if attrname != '__wrapped__': # Otherwise test discovery crashes
            return getattr(ast, attrname, None)()

func = _FuncCreator()

def vector(elems):
    if hasattr(elems, '__next__'):
        elems = list(elems)
    return ast.MakeVector(elems)

def ident(name, size=None):
    return ast.VectorIdentifier(name, size) if size else ast.Identifier(name)

def ref(name, to_obj=None):
    if to_obj is not None:
        try:
            size = len(to_obj)
        except ast.UnableToDecompose:
            size = None
        return ast.TypedReference(name, to_obj._dtype, size)
    else:
        return ast.Reference(name)

def defn(**kw):
    for name, value in kw.items():
        return ast.Definition(name, value)

def let(*steps):
    return ast.Let(list(steps[:-1]), steps[-1])

def iif(test, iftrue, iffalse):
    return ast.IfThenElse(test, iftrue, iffalse)

def repeat(node, n_times):
    return ast.MakeVector([node]*n_times)

def mean(elems, vector_dim=None):
    divisor = len(elems)
    if vector_dim is not None:
        divisor = [divisor] * vector_dim
    return sum_(elems) / const(divisor)

def sum_(elems):
    return ast.LFold(func.Add, elems)
