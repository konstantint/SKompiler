"""
SKompiler: Generate Sympy expressions from SKAST.
"""
#pylint: disable=protected-access
import warnings
from collections import OrderedDict
from itertools import product, chain, takewhile, count
import re
import numpy as np
from ._common import ASTProcessor, is_, StandardOps, VectorsAsLists, id_generator


def translate(node, component=None, multistage=True, assign_to=None,
              multistage_subexpression_min_length=3, _max_subexpression_length=8100):
    """Translates SKAST to an Excel formula (or a list of those, if the output should be a vector).

    Kwargs:
        component (int or None):
                   If the result is a vector and only one component is required, pass its index here.

        multistage (bool):
                   When False, generates a single string, describing the model as one long expression.
                   For complex models this string will be too long to be used in Excel.
                   When True (default), returns an ExcelCode object, which is an OrderedDict,
                   mapping cell names to expressions which they should correspond to.

        assign_to: A list or a generator expression, producing Excel cell names
                   which can be filled in a multi-stage computation.
                   When None, a default sequence ['A1', 'B1', ...] will be used.
                   if you would like such sequence, but beginning at, say 'G3',
                   pass excel_range('G3:*3')

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

        _max_subexpression_length (int): Max length of a single subexpression in multistage mode.
                                         You should not change it (used for testing internally)

    >>> from skompiler.toskast.string import translate as skast
    >>> expr = skast('[2*x[0]/5, 1] if x[1] <= 3 else [12.0+y, -45.5]')
    >>> print(translate(expr, multistage=False))
    ['IF((x2<=3),((2*x1)/5),(12.0+y))', 'IF((x2<=3),1,(-45.5))']
    >>> print(translate(expr))
    A1=IF((x2<=3),((2*x1)/5),(12.0+y))
    B1=IF((x2<=3),1,(-45.5))
    >>> expr = skast('a=1+x; a+a+a')
    >>> print(translate(expr))
    A1=(1+x)
    B1=((A1+A1)+A1)
    >>> print(translate(expr, multistage=False))
    (((1+x)+(1+x))+(1+x))
    """
    writer = ExcelWriter(multistage=multistage, assign_to=assign_to,
                         multistage_subexpression_min_length=multistage_subexpression_min_length,
                         _max_subexpression_length=_max_subexpression_length)
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
    if iftrue == iffalse:
        return iftrue
    else:
        return 'IF({0},{1},{2})'.format(cond, iftrue, iffalse)

def _dotproduct(xs, ys):
    return _sum('{0}*{1}'.format(x, y) for x, y in zip(xs, ys))

def _step(x):
    return _iif('{0}>0'.format(x), 1, 0)

def _max(xs):
    if len(xs) == 1:
        return xs[0]
    else:
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

def _takeuntil(value, iterable):
    if value is None:
        return iterable
    else:
        return chain(takewhile(lambda x: x != value, iterable), [value])

_letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'

def excel_column_names(start_column='A', end_column=None):
    """
    Generates excel column names starting from a given one.
    end_column is inclusive.

    >>> gen = excel_column_names('C')
    >>> [next(gen) for _ in range(4)]
    ['C', 'D', 'E', 'F']
    >>> list(excel_column_names('Y', 'AC'))
    ['Y', 'Z', 'AA', 'AB', 'AC']
    >>> list(excel_column_names('AY', 'BC'))
    ['AY', 'AZ', 'BA', 'BB', 'BC']
    >>> list(excel_column_names('ZY', 'AAC'))
    ['ZY', 'ZZ', 'AAA', 'AAB', 'AAC']
    """

    ns = _letters
    all_cols = map(''.join, chain(ns, product(ns, ns), product(ns, ns, ns)))
    while next(all_cols) != start_column:
        pass
    return _takeuntil(end_column, chain([start_column], all_cols))

_range_re = re.compile(r'^([\*A-Z]+)(\d+):([\*A-Z]+)(\d+)$')

def excel_range(range_):
    """
    A convenience method for generating lists of excel cells in a row or column.
    The range_ argument is an expression of the form A3:G3
    The second endpoint of the range may contain * instead of the
    row or column coordinate (as in A3:*3 or A3:A*), in this case
    the return value is a generator, enumerating cells in the given
    row or column indefinitely.
    """
    matches = _range_re.match(range_.upper())
    if not matches:
        raise ValueError("Range must be of the form A1:B1")
    lcol, lrow, rcol, rrow = matches.groups()
    if lcol == '*' or lrow == '*':
        raise ValueError("Only the right side of the interval may include wildcard")
    if lcol == rcol:  # Fixed column
        if rrow == '*':
            rows = count(int(lrow))
        else:
            rows = range(int(lrow), int(rrow)+1)
        yield from ('{0}{1}'.format(lcol, i) for i in rows)
    elif lrow == rrow: # Fixed row
        cols = excel_column_names(lcol, None if rcol == '*' else rcol)
        yield from ('{0}{1}'.format(col, lrow) for col in cols)
    else:
        raise ValueError("Only single-column or single-row ranges are supported")

def _compact_string(s, max_len=70):
    if len(s) > max_len:
        part = (max_len - 30) // 2
        return s[:part] + ' ...{0} chars skipped... '.format(len(s)-2*part) + s[-part:]
    else:
        return s

_builtins = {
    'IF': lambda t, a, b: a if t else b,
    'MAX': max,
    'EXP': np.exp,
    'LOG': np.log,
    'SQRT': np.sqrt,
    'ABS': np.abs,
}
_single_comparison = re.compile(r'(?<!\<)=')

class ExcelCode(OrderedDict):
    """A version of OrderedDict which prints itself nicer."""
    def __str__(self):
        lines = ['{0}={1}'.format(k, _compact_string(v))
                 for k, v in self.items()]
        if len(lines) > 10:
            lines = lines[:4] + ['      ... {0} lines skipped ...'.format(len(lines)-8)] +\
                    lines[-4:]
        return '\n'.join(lines)

    def to_dataframe(self):
        """Converts code to a pandas dataframe,
        suitable for pasting into Excel.

        The main usecase for this method is:
            
            code.to_dataframe().to_clipboard()

        """
        import pandas as pd
        return pd.DataFrame([['={0}'.format(v) for v in self.values()]],
                            columns=self.keys())

    def evaluate(self, **kwargs):
        """Evaluates the excel code using Python's eval.
        Will probably fail with MemoryError for longer strings
        (because Python's ast.parse can't handle them)."""
        env = OrderedDict()
        env.update(_builtins)
        for k, v in self.items():
            expand = _single_comparison.sub('==', v) # Excel uses '=' for comparisons
            env[k] = eval(expand, kwargs, env) #pylint: disable=eval-used
        for k in _builtins:
            del env[k]
        return env


class ExcelWriter(ASTProcessor, StandardOps, VectorsAsLists):
    """A SK AST processor, producing an Excel expression (or a list of those)"""

    def __init__(self, multistage=False, assign_to=None, positive_infinity=float(np.finfo('float64').max),
                 negative_infinity=float(np.finfo('float64').min),
                 multistage_subexpression_min_length=3,
                 _max_subexpression_length=8100):
        self.positive_infinity = positive_infinity
        self.negative_infinity = negative_infinity
        self.multistage = multistage
        self.multistage_subexpression_min_length = multistage_subexpression_min_length
        self.max_subexpression_length = _max_subexpression_length  # In multistage mode we attempt to keep subexpressions shorter than this
                                                                   # (because Excel does not allow cell values longer than 8196 chars)
                                                                   # NB: this is rather ad-hoc and may not always work.

        if self.multistage:
            if assign_to is None:
                warnings.warn("Value of the assign_to parameter is not provided. Will use default ['A1', 'B1', ...']", UserWarning)
                assign_to = excel_range('A1:*1')
            self.assign_to = assign_to if hasattr(assign_to, '__next__') else iter(assign_to)
            self.code = ExcelCode()
            self.references = [{}]
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
    Eq = is_fmt('({0}={1})')
    USub = is_fmt('(-{0})')
    Exp = is_fmt('EXP({0})')
    Sqrt = is_fmt('SQRT({0})')
    Log = is_fmt('LOG({0})')
    Max = is_fmt('MAX({0},{1})')
    Abs = is_fmt('ABS({0})')
    Step = is_(_step)
    Sigmoid = is_fmt('(1/(1+EXP(-{0})))')
    DotProduct = is_(_dotproduct)
    VecSum = is_(_sum)
    VecMax = is_(_max)
    MatVecProduct = lambda self, _: self._matvecproduct
    ArgMax = lambda self, _: self._argmax
    Softmax = lambda self, _: self._softmax

    def _argmax(self, xs):
        xs = [self.possibly_add_named_subexpression(x) for x in xs]
        max_var = self.possibly_add_named_subexpression(_max(xs))
        return _argmax(xs, max_var)

    def _matvecproduct(self, M, xs):
        xs = [self.possibly_add_named_subexpression(x) for x in xs]
        return [_sum('{0}*{1}'.format(m_i[j], xs[j]) for j in range(len(xs))) for m_i in M]
    
    def _vecsumnormalize(self, xs):
        xs = [self.possibly_add_named_subexpression(x) for x in xs]
        sum_var = self.possibly_add_named_subexpression(_sum(xs))
        return ['({0}/{1})'.format(x, sum_var) for x in xs]

    def _softmax(self, xs):
        return self._vecsumnormalize(['EXP({0})'.format(x) for x in xs])

    def Let(self, node, **kw):
        if not self.multistage:
            return StandardOps.Let(self, node)
        else:
            self.references.append({})
            for defn in node.defs:
                self.add_named_subexpression(self(defn.body), defn.name)
            result = self(node.body)
            self.references.pop()
            return result

    def Reference(self, node):
        if not self.multistage:
            raise ValueError("Reference nodes are only supported in multi-stage code generation")
        if node.name not in self.references[-1]:
            raise ValueError("Undefined reference: {0}".format(node.name))
        return self.references[-1][node.name]
    
    def possibly_add_named_subexpression(self, value):
        if self.multistage and len(value) >= self.multistage_subexpression_min_length:
            return self.add_named_subexpression(value)
        else:
            return value
        
    def add_named_subexpression(self, value, name=None):
        is_list = isinstance(value, list)
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
            if len(ref) == 1 and not is_list:
                ref = ref[0]
            self.references[-1][name] = ref
            return ref
        except StopIteration as ex:
            raise ValueError("The number of fields provided in the assign_to parameter"
                             " is not sufficient to complete the computation.") from ex
