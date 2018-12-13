"""
SKompiler: Generate Sympy expressions from SKAST.
"""
import warnings
from collections import OrderedDict
from itertools import product, chain
import numpy as np
from ..ast import ASTProcessor
from ._common import is_, LazyLet, VectorsAsLists, id_generator

def translate(node, component=None, multistage=False, assign_to=None,
              multistage_subexpression_min_length=3):
    """Translates SKAST to an Excel formula (or a list of those, if the output should be a vector).

    Kwargs:
        assign_to: A list or a generator expression, producing Excel cell names
                   which can be filled in a multi-stage computation.
                   When None, a default sequence ['A1', 'B1', ...] will be used.
                   if you would like such sequence, but beginning at, say 'G3',
                   pass excel_row_generator('G', 3).

        multistage_subexpression_min_length (int):
            Allows to reduce the number of stages in the computation, by preventing the creation
            of an intermediate step whenever the corresponding expression is shorter than the given length.
            I.e. suppose you would like to avoid having a short separate subexpression
                G1 = MAX(A1,B1,C1)
            and would rather have it inlined.
            In this case specify multistage_subexpression_min_length=14 and the expression shorter than 14 characters won't be
            assigned to a separate variable.
            Specifying a very large value is nearly equivalent to setting multistage=False
            (the only difference is that the returned value is still an OrderedDict with a single assignment)

    >>> from skompiler.toskast.string import translate as skast
    >>> expr = skast('[2*x[0]/5, 1] if x[1] <= 3 else [12.0+y, -45.5]')
    >>> print(translate(expr))
    ['IF((x2<=3),((2*x1)/5),(12.0+y))', 'IF((x2<=3),1,(-45.5))']
    """
    writer = ExcelWriter(multistage=multistage, assign_to=assign_to,
                         multistage_subexpression_min_length=multistage_subexpression_min_length)
    result = writer(node)
    if component is not None:
        result = result[component]
    if multistage:
        writer.add_named_subexpression(result)
        return writer.code
    else:
        return result

def _sum(iterable):
    return "({0})".format("+".join(iterable))

def _iif(cond, iftrue, iffalse):
    return 'IF({0},{1},{2})'.format(cond, iftrue, iffalse)

def _matvecproduct(M, x):
    return [_sum('{0}*{1}'.format(m_i[j], x[j]) for j in range(len(x))) for m_i in M]

def _dotproduct(xs, ys):
    return _sum('{0}*{1}'.format(x, y) for x, y in zip(xs, ys))

def _step(x):
    return _iif('{0}>0'.format(x), 1, 0)

def _max(xs):
    return 'MAX({0})'.format(','.join(xs))

def _argmax(xs, maxval=None):
    if not maxval:
        maxval = _max(xs)
    expr = str(len(xs)-1)
    n = len(xs)-1
    while n > 0:
        n -= 1
        expr = _iif('{0}={1}'.format(xs[n], maxval), str(n), expr)
    return expr

def is_fmt(template):
    # Auto-compacting binary operator
    def auto_compacting_operator(self, _):
        def fn(x, y=''):
            result = template.format(x, y)
            if self.multistage and len(result) > self.max_subexpression_length:
                if len(x) > len(y):
                    x, y = self.add_named_subexpression(x), y
                else:
                    x, y = x, self.add_named_subexpression(y)
                result = template.format(x, y)
            return result
        return fn
    return auto_compacting_operator

_letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
def excel_row_generator(start_column='A', row=1):
    """
    >>> gen = excel_row_generator('C', 3)
    >>> [next(gen) for _ in range(4)]
    ['C3', 'D3', 'E3', 'F3']
    >>> gen = excel_row_generator('Y', 3)
    >>> [next(gen) for _ in range(5)]
    ['Y3', 'Z3', 'AA3', 'AB3', 'AC3']
    >>> gen = excel_row_generator('AY', 1)
    >>> [next(gen) for _ in range(5)]
    ['AY1', 'AZ1', 'BA1', 'BB1', 'BC1']
    >>> gen = excel_row_generator('ZY', 1)
    >>> [next(gen) for _ in range(5)]
    ['ZY1', 'ZZ1', 'AAA1', 'AAB1', 'AAC1']
    """
    ns = _letters
    all_cols = map(''.join, chain(ns, product(ns, ns), product(ns, ns, ns)))
    while next(all_cols) != start_column: pass
    all_cols = chain([start_column], all_cols)
    yield from ('{0}{1}'.format(col, row) for col in all_cols)


class ExcelWriter(ASTProcessor, VectorsAsLists, LazyLet):
    """A SK AST processor, producing an Excel expression (or a list of those)"""

    def __init__(self, multistage=False, assign_to=None, positive_infinity=float(np.finfo('float64').max),
                 negative_infinity=float(np.finfo('float64').min),
                 multistage_subexpression_min_length=3):
        self.positive_infinity = positive_infinity
        self.negative_infinity = negative_infinity
        self.multistage = multistage
        self.multistage_subexpression_min_length = multistage_subexpression_min_length
        self.max_subexpression_length = 8100  # In multistage mode we attempt to keep subexpressions shorter than this
                                              # (because Excel does not allow cell values longer than 8196 chars)
                                              # NB: this is rather ad-hoc and may not always work.

        if self.multistage:
            if assign_to is None:
                warnings.warn("Value of the assign_to parameter is not provided. Will use default ['A1', 'B1', ...']", UserWarning)
                assign_to = excel_row_generator()
            self.assign_to = assign_to if hasattr(assign_to, '__next__') else iter(assign_to)
            self.code = OrderedDict()
            self.references = {}
            self.temp_ids = id_generator()

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

    def _iif(self, test, ift, iff):
        # Auto-compacting IIF
        result = _iif(test, ift, iff)
        if self.multistage and len(result) > self.max_subexpression_length:
            if len(ift) > len(iff):
                ift, iff = self.add_named_subexpression(ift), iff
            else:
                ift, iff = ift, self.add_named_subexpression(iff)
            result = _iif(test, ift, iff)
        return result
        
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

    def ArgMax(self, _):
        return self._argmax

    def _argmax(self, xs):
        xs = [self.possibly_add_named_subexpression(x) for x in xs]
        max_var = self.possibly_add_named_subexpression(_max(xs))
        return _argmax(xs, max_var)

    def VecSumNormalize(self, _):
        return self._vecsumnormalize
    
    def _vecsumnormalize(self, xs):
        xs = [self.possibly_add_named_subexpression(x) for x in xs]
        sum_var = self.possibly_add_named_subexpression(_sum(xs))
        return ['({0}/{1})'.format(x, sum_var) for x in xs]

    def SKLearnSoftmax(self, _):
        return self._sklearn_softmax

    def _sklearn_softmax(self, xs):
        xs = [self.possibly_add_named_subexpression(x) for x in xs]
        x_max = self.possibly_add_named_subexpression(_max(xs))
        return self._vecsumnormalize(['EXP({0}-{1})'.format(x, x_max) for x in xs])

    def Let(self, node):
        if not self.multistage:
            return LazyLet.Let(self, node)
        else:
            for defn in node.defs:
                self.add_named_subexpression(self(defn.body), defn.name)
            return self(node.body)

    def Reference(self, node):
        if not self.multistage:
            raise ValueError("Reference nodes are only supported in multi-stage code generation")
        if node.name not in self.references:
            raise ValueError("Undefined reference: {0}".format(node.name))
        return self.references[node.name]
    
    def possibly_add_named_subexpression(self, value):
        if self.multistage and len(value) >= self.multistage_subexpression_min_length:
            return self.add_named_subexpression(value)
        else:
            return value
        
    def add_named_subexpression(self, value, name=None):
        if not isinstance(value, list):
            value = [value]
        if name is None:
            name = next(self.temp_ids)
        try:
            ref = []
            for v in value:
                next_output = next(self.assign_to)
                if next_output in self.code:
                    raise ValueError("Repeated names are not supported in the assign_to parameter")
                self.code[next_output] = v
                ref.append(next_output)
            if len(ref) == 1:
                ref = ref[0]
            self.references[name] = ref
            return ref
        except StopIteration as ex:
            raise ValueError("The number of fields provided in the assign_to parameter"
                             " is not sufficient to complete the computation.") from ex
