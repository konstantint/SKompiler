"""
Common useful functionality.
"""
from ...ast import UnaryFunc, ArgMax, Log, LFold, NumberConstant, MakeVector, BinOp, Div, Add


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

def mean(elems, vector_dim=None):
    divisor = NumberConstant(len(elems))
    if vector_dim is not None:
        divisor = MakeVector([divisor]*vector_dim)
    return BinOp(Div(), sum_(elems), divisor)

def sum_(elems):
    return LFold(Add(), elems)
