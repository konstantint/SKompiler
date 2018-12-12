"""
SKompiler: Generate Sympy expressions from SKAST.
"""
#pylint: disable=wildcard-import,unused-wildcard-import
import warnings
import numpy as np
from ..ast import ASTProcessor, IndexedIdentifier, NumberConstant

def translate(node):
    """Translates SKAST to an Excel formula (or a list of those, if the output should be a vector).
    
    >>> from skompiler.toskast.string import translate as skast
    >>> expr = skast('[2*x[0], 1] if x[1] <= 3 else [12.0, 45.5]')
    >>> print(translate(expr))
    ['IF((x2 <= 3), (2*x1), 12.0)', 'IF((x2 <= 3), 1, 45.5)']
    """
    return ExcelWriter()(node)

def _is(val):
    return lambda self, node: val

def _sum(iterable):
    return "({0})".format("+".join(iterable))

def _iif(cond, iftrue, iffalse):
    return 'IF({0}, {1}, {2})'.format(cond, iftrue, iffalse)

def _sklearn_softmax(xs):
    x_max = _max(xs)
    return _vecsumnormalize(['EXP({0} - {1})'.format(x, x_max) for x in xs])

def _matvecproduct(M, x):
    return [_sum('{0} * {1}'.format(m_i[j], x[j]) for j in range(len(x))) for m_i in M]

def _dotproduct(xs, ys):
    return _sum('{0} * {1}'.format(x, y) for x, y in zip(xs, ys))

def _vecsumnormalize(xs):
    return ['({0} / {1})'.format(x, _sum(xs)) for x in xs]

def _step(x):
    return _iif('{0} > 0'.format(x), 1, 0)

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


def _tolist(x):
    if hasattr(x, 'tolist'):
        return x.tolist()
    else:
        return list(x)
    
class ExcelWriter(ASTProcessor):
    """A SK AST processor, producing an Excel expression (or a list of those)"""
    def __init__(self, positive_infinity=float(np.finfo('float64').max), negative_infinity=float(np.finfo('float64').min)):
        self.positive_infinity = positive_infinity
        self.negative_infinity = negative_infinity

    def Identifier(self, id):
        return id.id

    def VectorIdentifier(self, id):
        return [self(IndexedIdentifier(id.id, i, id.size)) for i in range(id.size)]

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

    def VectorConstant(self, vec):
        return [self(NumberConstant(v)) for v in _tolist(vec.value)]

    def MatrixConstant(self, mtx):
        return [[self(NumberConstant(v)) for v in _tolist(row)] for row in mtx.value]

    def MakeVector(self, vec):
        return [self(el) for el in vec.elems]

    def UnaryFunc(self, op):
        return self(op.op)(self(op.arg))

    def ElemwiseUnaryFunc(self, op):
        arg = self(op.arg)
        if not isinstance(arg, list):
            raise ValueError("Elementwise operations are only supported for vectors")
        return list(map(self(op.op), arg))

    def BinOp(self, op):
        return self(op.op)(self(op.left), self(op.right))
    CompareBinOp = BinOp

    def ElemwiseBinOp(self, op):
        left = self(op.left)
        right = self(op.right)
        op = self(op.op)
        if not isinstance(left, list) or not isinstance(right, list):
            raise ValueError("Elementwise operations are only supported for vectors")
        if len(left) != len(right):
            raise ValueError("Sizes of the arguments do not match")
        return [op(l, r) for l, r in zip(left, right)]

    def IfThenElse(self, node):
        test, iftrue, iffalse = self(node.test), self(node.iftrue), self(node.iffalse)
        if isinstance(iftrue, list):
            if not isinstance(iffalse, list) or len(iftrue) != len(iffalse):
                raise ValueError("Mixed types in IfThenElse expressions are not supported")
            return [_iif(test, ift, iff) for ift, iff in zip(iftrue, iffalse)]
        else:
            if isinstance(iffalse, list):
                raise ValueError("Mixed types in IfThenElse expressions are not supported")
            return _iif(test, iftrue, iffalse)

    #pylint: disable=unnecessary-lambda
    Mul = _is(lambda x, y: '({0}*{1})'.format(x, y))
    Div = _is(lambda x, y: '({0}/{1})'.format(x, y))
    Add = _is(lambda x, y: '({0}+{1})'.format(x, y))
    Sub = _is(lambda x, y: '({0}-{1})'.format(x, y))
    LtEq = _is(lambda x, y: '({0} <= {1})'.format(x, y))
    USub = _is(lambda x: '(-{0})'.format(x))
    Exp = _is(lambda x: 'EXP({0})'.format(x))
    Log = _is(lambda x: 'LOG({0})'.format(x))
    Step = _is(_step)
    Sigmoid = _is(lambda x: '(1 / (1 + EXP(-{0}))'.format(x))
    MatVecProduct = _is(_matvecproduct)
    DotProduct = _is(_dotproduct)
    VecSumNormalize = _is(_vecsumnormalize)
    SKLearnSoftmax = _is(_sklearn_softmax)
    ArgMax = _is(_argmax)

    def Let(self, let):
        # In principle we may consider compiling Let expressions into a series of separate statements, which
        # may be used in a sequence of "with" statements.
        raise NotImplementedError("Let expressions are not implemented. Use substitute_variables instead.")

    Reference = Definition = None
