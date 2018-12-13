"""
SKompiler: Generate Sympy expressions from SKAST.
"""
from functools import reduce
from collections import namedtuple
import numpy as np
import sqlalchemy as sa
from ..ast import ASTProcessor, ArgMax, VecSumNormalize, SKLearnSoftmax, MatVecProduct, DotProduct
from ._common import StandardArithmetics, LazyLet, is_, tolist, not_implemented, process_assign_to


def translate(expr, assign_to=None, from_obj='data', key_column='id', component=None):
    """Multistage translation logic.
       See explanation in .sqlalchemy.translate().
    """

    result = SQLAlchemyMultistageWriter(from_obj=from_obj, key_column=key_column)(expr)

    if component is not None:
        result = result._replace(cols=[result.cols[component]])
    
    assign_to = process_assign_to(assign_to, len(result.cols))
    if assign_to is not None:
        result = result._replace(cols=[col.label(lbl) for col, lbl in zip(result.cols, assign_to)])
    
    return sa.select(result.cols, from_obj=result.from_obj)


def _max(xs):
    return reduce(greatest, xs)

def _sum(iterable):
    "The built-in 'sum' does not work for us as we need."
    return reduce(lambda x, y: x+y, iterable)

def _iif(cond, iftrue, iffalse):
    return sa.case([(cond, iftrue)], else_=iffalse)

def _matvecproduct(M, x):
    return [_sum(m_i[j] * x[j] for j in range(len(x))) for m_i in M]

def _dotproduct(xs, ys):
    return [_sum(x * y for x, y in zip(xs, ys))]

def _step(x):
    return _iif(x > 0, 1, 0)

def _merge(tbl1, tbl2):
    if tbl1 is None:
        return tbl2
    elif tbl2 is None:
        return tbl1
    if tbl1 is tbl2:
        return tbl1
    joined = tbl1.join(tbl2, tbl1.key_ == tbl2.key_)
    joined.key_ = tbl1.key_
    return joined


Result = namedtuple('Result', 'cols from_obj')

class SQLAlchemyMultistageWriter(ASTProcessor, StandardArithmetics, LazyLet):
    """A SK AST processor, producing a SQLAlchemy "multistage" expression.
       The interpretation of each node is a tuple, containing a list of column expressions and a from_obj,
       where these columns must be queried from."""
    
    def __init__(self, from_obj='data', key_column='id',
                 positive_infinity=float(np.finfo('float64').max),
                 negative_infinity=float(np.finfo('float64').min)):
        self.positive_infinity = positive_infinity
        self.negative_infinity = negative_infinity
        if isinstance(from_obj, str):
            from_obj = sa.table(from_obj, sa.column(key_column))
            # This is a bit hackish, but quite convenient.
            # This way we do not have to carry around an extra "key" field in our results all the time
            from_obj.key_ = from_obj.columns[key_column]
        else:
            from_obj.key_ = from_obj.columns[key_column]
        self.from_obj = from_obj
        self.last_used_id = 0
    
    def Identifier(self, id):
        return Result([sa.column(id.id)], self.from_obj)

    def _indexed_identifier(self, id, idx):
        return sa.column("{0}{1}".format(id, idx+1))
    
    def IndexedIdentifier(self, sub):
        return Result([self._indexed_identifier(sub.id, sub.index)], self.from_obj)

    def _number_constant(self, value):
        # Infinities have to be handled separately
        if np.isinf(value):
            return self.positive_infinity if value > 0 else self.negative_infinity
        else:
            return value

    def NumberConstant(self, num):
        return Result([self._number_constant(num.value)], None)

    def VectorIdentifier(self, id):
        return Result([self._indexed_identifier(id.id, i) for i in range(id.size)], self.from_obj)

    def VectorConstant(self, vec):
        return Result([self._number_constant(v) for v in tolist(vec.value)], None)

    def MatrixConstant(self, mtx):
        return Result([[self._number_constant(v) for v in tolist(row)] for row in mtx.value], None)

    def UnaryFunc(self, node):
        arg = self(node.arg)
        if isinstance(node.op, ArgMax):
            return self._argmax(arg)
        elif isinstance(node.op, VecSumNormalize):
            return self._vecsumnormalize(arg)
        elif isinstance(node.op, SKLearnSoftmax):
            return self._sklearn_softmax(arg)
        else:
            op = self(node.op)
            return Result([op(el) for el in arg.cols], arg.from_obj)

    ElemwiseUnaryFunc = UnaryFunc
    ArgMax = VecSumNormalize = SKLearnSoftmax = not_implemented

    def BinOp(self, node):
        left, right, op = self(node.left), self(node.right), self(node.op)
        if node.op.__class__ in [MatVecProduct, DotProduct]:
            return Result(op(left.cols, right.cols), _merge(left.from_obj, right.from_obj))
        elif len(left.cols) != len(right.cols):
            raise ValueError("Mismatching operand dimensions in {0}".format(repr(node.op)))
        return Result([op(lc, rc) for lc, rc in zip(left.cols, right.cols)], _merge(left.from_obj, right.from_obj))
    
    CompareBinOp = ElemwiseBinOp = BinOp

    def MakeVector(self, vec):
        result = []
        tbls = set()
        for el in vec.elems:
            el = self(el)
            tbls.add(el.from_obj)
            if len(el.cols) != 1:
                raise ValueError("MakeVector expects a list of scalars")
            result.append(el.cols[0])
        if len(tbls) != 1:
            raise NotImplementedError("MakeVector currently only supports concatenation of values from the same source")
        return Result(result, list(tbls)[0])

    def IfThenElse(self, node):
        test, iftrue, iffalse = self(node.test), self(node.iftrue), self(node.iffalse)

        return Result([_iif(test.cols[0], ift, iff) for ift, iff in zip(iftrue.cols, iffalse.cols)],
                      reduce(_merge, [test.from_obj, iftrue.from_obj, iffalse.from_obj]))

    MatVecProduct = is_(_matvecproduct)
    DotProduct = is_(_dotproduct)
    Exp = is_(sa.func.exp)
    Log = is_(sa.func.log)
    Step = is_(_step)

    # ------ The actual "multi-stage" logic -----
    def _next_id(self, stem='temp'):
        self.last_used_id += 1
        return '_{0}{1}'.format(stem, self.last_used_id)
    
    def _make_cte(self, result, col_names=None, key_label='__id__'):
        if col_names is None:
            col_names = ['f{0}'.format(i+1) for i in range(len(result.cols))]
        labeled_cols = [c.label(n) for c, n in zip(result.cols, col_names)]
        new_tbl = sa.select([result.from_obj.key_.label(key_label)] + labeled_cols, from_obj=result.from_obj).cte(self._next_id('tmp'))
        new_tbl.key_ = new_tbl.columns[key_label]
        new_cols = [new_tbl.columns[n] for n in col_names]
        return Result(new_cols, new_tbl)

    def _argmax(self, result):
        features = self._make_cte(result)
        max_val = Result([_max(features.cols)], features.from_obj)
        max_val = self._make_cte(max_val, ['_max'])

        argmax = sa.case([(col == max_val.cols[0], i)
                          for i, col in enumerate(features.cols[:-1])],
                         else_=len(features.cols)-1)
        return Result([argmax], _merge(features.from_obj, max_val.from_obj))
    
    def _sklearn_softmax(self, result):
        features = self._make_cte(result)
        max_val = Result([_max(features.cols)], features.from_obj)
        max_val = self._make_cte(max_val, ['_max'])

        sub_max = Result([sa.func.exp(col - max_val.cols[0]) for col in features.cols],
                         _merge(features.from_obj, max_val.from_obj))
        return self._vecsumnormalize(sub_max)

    def _vecsumnormalize(self, result):
        features = self._make_cte(result)
        sum_val = Result([_sum(features.cols)], features.from_obj)
        sum_val = self._make_cte(sum_val, ['_sum'])
        return Result([col/sum_val.cols[0] for col in features.cols],
                      _merge(features.from_obj, sum_val.from_obj))


# ------- SQLAlchemy "greatest" function
# See https://docs.sqlalchemy.org/en/latest/core/compiler.html
#pylint: disable=wrong-import-position,wrong-import-order
from sqlalchemy.sql import expression
from sqlalchemy.ext.compiler import compiles
from sqlalchemy.types import Numeric

class greatest(expression.FunctionElement):
    type = Numeric()
    name = 'greatest'

@compiles(greatest)
def default_greatest(element, compiler, **kw):
    res = compiler.visit_function(element, **kw)
    return res

@compiles(greatest, 'sqlite')
@compiles(greatest, 'mssql')
@compiles(greatest, 'oracle')
def case_greatest(element, compiler, **kw):
    arg1, arg2 = list(element.clauses)
    return compiler.process(sa.case([(arg1 > arg2, arg1)], else_=arg2), **kw)

# Utilities ----------------------------------
import sqlalchemy.dialects
#pylint: disable=wildcard-import,unused-wildcard-import
from sqlalchemy.dialects import *   # Must do it in order to getattr(sqlalchemy.dialects, ...)
def to_sql(sa_exprs, dialect_name='sqlite', assign_to='y'):
    """
    Helper function. Given a SQLAlchemy expression (or a list of those), returns the corresponding
    SQL string in a given dialect.

    If assign_to is None, the returned value is a list
    of strings (one for each output component).

    If assign_to is a list of column names,
    the returned value is a single string of the form
    "(first sql expr) as <first target name>,
     (second sql expr) as <second target name>,
     ...
     (last sql expr) as <last target name>"
    
     If assign_to is a single string <y>, the target columns are named '<y>1, <y>2, ..., <y>n',
     if there are several, or just '<y>', if there is only one..
     """
    
    dialect_module = getattr(sqlalchemy.dialects, dialect_name)
    if not isinstance(sa_exprs, list):
        sa_exprs = [sa_exprs]
    qs = [q.compile(dialect=dialect_module.dialect(),
                    compile_kwargs={'literal_binds': True}) for q in sa_exprs]

    if assign_to is None:
        return [str(q) for q in qs]

    if isinstance(assign_to, str):
        if len(qs) > 1:
            assign_to = ['{0}{1}'.format(assign_to, i+1) for i in range(len(qs))]
        else:
            assign_to = [assign_to]
    
    if len(assign_to) != len(qs):
        raise ValueError("The number of resulting SQL expressions does not match the number of "
                         "target column names.")

    return ',\n'.join(['{0} as {1}'.format(q, tgt) for q, tgt in zip(qs, assign_to)])
