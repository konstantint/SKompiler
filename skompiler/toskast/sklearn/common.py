"""
Common useful functionality.
"""
from ...ast import UnaryFunc, ArgMax, Log, LFold, NumberConstant,\
                   MakeVector, BinOp, Div, Add, VecSum, Let, Definition, Reference


def classifier(probs, method):
    """Given a probability expression and a method name, returns the classifier output"""

    if method == 'predict_proba':
        return probs
    elif method == 'predict':
        return UnaryFunc(ArgMax(), probs)
    elif method == 'predict_log_proba':
        return UnaryFunc(Log(), probs)
    else:
        raise ValueError("Invalid method: {0}".format(method))

def repeat(value, ntimes):
    return MakeVector([value]*ntimes)

def mean(elems, vector_dim=None):
    divisor = NumberConstant(len(elems))
    if vector_dim is not None:
        divisor = repeat(divisor, vector_dim)
    return BinOp(Div(), sum_(elems), divisor)

def sum_(elems):
    return LFold(Add(), elems)

def ref(name):
    return Reference(name)

def defn(**kw):
    for name, value in kw.items():
        return Definition(name, value)

def let(*steps):
    return Let(list(steps[:-1]), steps[-1])

def vecsumnormalize(node, vector_dim):
    return let(defn(x=node),
               defn(s=UnaryFunc(VecSum(), ref('x'))),
               BinOp(Div(), ref('x'), repeat(ref('s'), vector_dim)))
