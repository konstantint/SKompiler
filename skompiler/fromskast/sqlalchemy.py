"""
SKompiler: Generate Sympy expressions from SKAST.
"""
from functools import reduce
import warnings
import numpy as np
import sqlalchemy as sa
from ._common import ASTProcessor, StandardOps, StandardArithmetics, VectorsAsLists, is_, prepare_assign_to
from ._sqla_multistage import translate as translate_multistage


def translate(node, dialect=None, assign_to='y', component=None,
              multistage=False, multistage_key_column='id', multistage_from_obj='data'):
    """Translates SKAST to an SQLAlchemy expression (or a list of those, if the output should be a vector).

    If dialect is not None, further compiles the expression(s) to a given dialect via to_sql.
    
    Kwargs:
       assign_to (None/string/list of str): See to_sql

       multistage (bool):  Translate in "multistage mode". In this mode the returned value is not an expression
                               for a single column (or a list of such expressions), which must then be manually inserted
                               into an appropriate SELECT clause. Insted, it is a complete SELECT query, which may include CTEs.
                               This makes it possible to compute complex functions (such as ArgMax) without having to repeat the
                               same computation over and over again.
                               
                               As it is a full query, you must provide the name of the source object to be queried from (which can
                               be a table name or a SQLAlchemy SELECT object). In addition, you must specify the name of the
                               key column in the source data, as this will be used to join the CTEs in the query.

                               assign_to may not be None in multistage mode.

       multistage_from_obj:    A string or a SQLAlchemy select object - the source table for the data.
       multistage_key_column:  A string or a sa.column object, naming the key column in the source table.

    
    >>> from skompiler.toskast.string import translate as skast
    >>> expr = skast('[2*x[0], 1] if x[1] <= 3 else [12.0, 45.5]')
    >>> print(translate(expr, 'sqlite'))
    CASE WHEN (x2 <= 3) THEN 2 * x1 ELSE 12.0 END as y1,
    CASE WHEN (x2 <= 3) THEN 1 ELSE 45.5 END as y2
    """
    if multistage:
        saexprs = translate_multistage(node, assign_to, multistage_from_obj, multistage_key_column)
        if dialect is None:
            return saexprs
        else:
            return to_sql(saexprs, dialect, assign_to=None)[0]
    else:
        saexprs = SQLAlchemyWriter()(node)
        if component is not None:
            saexprs = saexprs[component]
        if dialect is None:
            return saexprs
        else:
            return to_sql(saexprs, dialect, assign_to=assign_to)

def _sum(iterable):
    "The built-in 'sum' does not work for us as we need."
    return reduce(lambda x, y: x+y, iterable)

def _iif(cond, iftrue, iffalse):
    return sa.case([(cond, iftrue)], else_=iffalse)

def _sklearn_softmax(xs):
    x_max = _max(xs)
    return _vecsumnormalize([sa.func.exp(x - x_max) for x in xs])

def _matvecproduct(M, x):
    return [_sum(m_i[j] * x[j] for j in range(len(x))) for m_i in M]

def _dotproduct(xs, ys):
    return _sum(x * y for x, y in zip(xs, ys))

def _vecsumnormalize(xs):
    return [x / _sum(xs) for x in xs]

def _step(x):
    return _iif(x > 0, 1, 0)

def _max(xs):
    return reduce(greatest, xs)

def _argmax(xs):
    return sa.case([(x == _max(xs), i)
                    for i, x in enumerate(xs[:-1])],
                   else_=len(xs)-1)


class SQLAlchemyWriter(ASTProcessor, StandardOps, StandardArithmetics, VectorsAsLists):
    """A SK AST processor, producing a SQLAlchemy expression (or a list of those)"""
    def __init__(self, positive_infinity=float(np.finfo('float64').max), negative_infinity=float(np.finfo('float64').min)):
        self.positive_infinity = positive_infinity
        self.negative_infinity = negative_infinity
    
    def Identifier(self, id):
        return sa.column(id.id)

    def IndexedIdentifier(self, sub):
        warnings.warn("SQL does not support vector types natively. "
                      "Numbers will be appended to the given feature name, "
                      "it may not be what you intend.", UserWarning)
        return sa.column("{0}{1}".format(sub.id, sub.index+1))

    def NumberConstant(self, num):
        # Infinities have to be handled separately
        if np.isinf(num.value):
            return self.positive_infinity if num.value > 0 else self.negative_infinity
        else:
            return num.value

    _iif = lambda self, test, ift, iff: _iif(test, ift, iff)
    MatVecProduct = is_(_matvecproduct)
    DotProduct = is_(_dotproduct)
    Exp = is_(sa.func.exp)
    Log = is_(sa.func.log)
    Step = is_(_step)
    VecSum = is_(_sum)
    SKLearnSoftmax = is_(_sklearn_softmax)
    ArgMax = is_(_argmax)


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

    assign_to = prepare_assign_to(assign_to, len(qs))
    if assign_to is None:
        return [str(q) for q in qs]
    else:
        return ',\n'.join(['{0} as {1}'.format(q, tgt)
                           for q, tgt in zip(qs, assign_to)])
