"""
SKompiler: Generate Sympy expressions from SKAST.
"""
import warnings
import numpy as np
from ..ast import ASTProcessor
from ._common import is_, LazyLet, VectorsAsLists

def translate(node):
    """Translates SKAST to an Excel formula (or a list of those, if the output should be a vector).
    
    >>> from skompiler.toskast.string import translate as skast
    >>> expr = skast('[2*x[0]/5, 1] if x[1] <= 3 else [12.0+y, -45.5]')
    >>> print(translate(expr))
    ['IF((x2<=3),((2*x1)/5),(12.0+y))', 'IF((x2<=3),1,(-45.5))']
    """
    return ExcelWriter()(node)

def _sum(iterable):
    return "({0})".format("+".join(iterable))

def _iif(cond, iftrue, iffalse):
    return 'IF({0},{1},{2})'.format(cond, iftrue, iffalse)

def _sklearn_softmax(xs):
    x_max = _max(xs)
    return _vecsumnormalize(['EXP({0}-{1})'.format(x, x_max) for x in xs])

def _matvecproduct(M, x):
    return [_sum('{0}*{1}'.format(m_i[j], x[j]) for j in range(len(x))) for m_i in M]

def _dotproduct(xs, ys):
    return _sum('{0}*{1}'.format(x, y) for x, y in zip(xs, ys))

def _vecsumnormalize(xs):
    return ['({0}/{1})'.format(x, _sum(xs)) for x in xs]

def _step(x):
    return _iif('{0}>0'.format(x), 1, 0)

def _max(xs):
    return 'MAX({0})'.format(','.join(xs))

def _argmax(xs):
    maxval = _max(xs)
    expr = str(len(xs)-1)
    n = len(xs)-1
    while n > 0:
        n -= 1
        expr = _iif('{0} = {1}'.format(xs[n], maxval), str(n), expr)
    return expr

def is_fmt(template):
    return is_(template.format)


class ExcelWriter(ASTProcessor, VectorsAsLists, LazyLet):
    """A SK AST processor, producing an Excel expression (or a list of those)"""

    def __init__(self, positive_infinity=float(np.finfo('float64').max), negative_infinity=float(np.finfo('float64').min)):
        self.positive_infinity = positive_infinity
        self.negative_infinity = negative_infinity

    def Identifier(self, id):
        return id.id

    def IndexedIdentifier(self, sub):
        warnings.warn("Excel does not support vector types natively. "
                      "Numbers will be appended to the given feature name, "
                      "it may not be what you intend.", UserWarning)
        return "{0}{1}".format(sub.id, sub.index+1)
    
    def NumberConstant(self, num):
        # Infinities have to be handled separately
        if np.isinf(num.value):
            val = self.positive_infinity if num.value > 0 else self.negative_infinity
        else:
            val = num.value
        return str(val)
    
    _iif = lambda self, test, ift, iff: _iif(test, ift, iff)

    # Implement binary and unary operations
    Mul = is_fmt('({0}*{1})')
    Div = is_fmt('({0}/{1})')
    Add = is_fmt('({0}+{1})')
    Sub = is_fmt('({0}-{1})')
    LtEq = is_fmt('({0}<={1})')
    USub = is_fmt('(-{0})')
    Exp = is_fmt('EXP({0})')
    Log = is_fmt('LOG({0})')
    Step = is_(_step)
    Sigmoid = is_fmt('(1/(1+EXP(-{0}))')
    MatVecProduct = is_(_matvecproduct)
    DotProduct = is_(_dotproduct)
    VecSumNormalize = is_(_vecsumnormalize)
    SKLearnSoftmax = is_(_sklearn_softmax)
    ArgMax = is_(_argmax)
