"""
SKompiler: Generate SQLAlchemy expressions from SKAST.
"""
from collections import namedtuple
from functools import reduce

import numpy as np
import sqlalchemy as sa
from sqlalchemy import and_
from sqlalchemy.sql.selectable import Join, FromGrouping

from ._common import ASTProcessor, StandardOps, StandardArithmetics, is_, tolist, \
    not_implemented, prepare_assign_to, id_generator, denumpyfy
from ..ast import ArgMax, VecMax, Softmax, IsElemwise, VecSum, Max, IsAtom

DEFAULT_KEY_LABEL_PREFIX = "__id__"


# pylint: disable=trailing-whitespace
def translate(node, dialect=None, assign_to='y', component=None,
              multistage=True, key_column='id', from_obj='data',
              include_key_col_in_query=False):
    """Translates SKAST to an SQLAlchemy expression (or a list of those, if the output should be a vector).

    If dialect is not None, further compiles the expression(s) to a given dialect via to_sql.
    
    Kwargs:
       assign_to (None/string/list of str): See to_sql
       
       component (int):    If the result is a vector and you only need one component of it, specify its index (0-based) here.

       multistage (bool):  When multistage=False, the returned value is a single expression which can be selected directly from the
                           source data table. This, however, may make the resulting query rather long, as some functions (e.g. argmax)
                           require repeated computaion of the same parts over and over.
                           The problem is solved by splitting the computation in a sequence of CTE subqueries - the "multistage" mode.
                           The resulting query may then look like
                           
                           with _tmp1 as (select [probability computations] from data),
                                _tmp2 as (select [argmax computation] from _tmp1),
                                ... 
                           select [final values] from _tmpX
                        
                           Default - True

       from_obj:    A string or a SQLAlchemy selectable object - the source table for the data.
                    In non-multistage mode this may be None - in this case the returned value is 
                    simply 'SELECT cols'.

       key_column:  A string or a sa.column object, naming the key column in the source table.
                    Compulsory for multistage mode.

    
    >>> from skompiler.toskast.string import translate as skast
    >>> expr = skast('[2*x[0], 1] if x[1] <= 3 else [12.0, 45.5]')
    >>> print(translate(expr, 'sqlite', multistage=False, from_obj=None))
    SELECT CASE WHEN (x2 <= 3) THEN 2 * x1 ELSE 12.0 END AS y1, CASE WHEN (x2 <= 3) THEN 1 ELSE 45.5 END AS y2


    >>> expr = skast('x=1; y=2; x+y')
    >>> print(translate(expr, 'sqlite', multistage=True))
    WITH _tmp1 AS 
    (SELECT data.id AS __id__, 1 AS f1 
    FROM data), 
    _tmp2 AS 
    (SELECT data.id AS __id__, 2 AS f1 
    FROM data)
     SELECT _tmp1.f1 + _tmp2.f1 AS y 
    FROM _tmp1 JOIN _tmp2 ON _tmp1.__id__ = _tmp2.__id__
    >>> expr = skast('x+y')
    >>> stbl = sa.select([sa.column('id'), sa.column('x'), sa.column('y')], from_obj=sa.table('test')).cte('_data')
    >>> print(translate(expr, 'sqlite', multistage=False, from_obj=stbl))
    WITH _data AS 
    (SELECT id, x, y 
    FROM test)
     SELECT x + y AS y 
    FROM _data
    """
    if multistage and from_obj is None:
        raise ValueError("from_obj must be specified in multistage mode")
    key_column = [key_column] if not isinstance(key_column, list) else key_column
    result = SQLAlchemyWriter(from_obj=from_obj, key_columns=key_column, multistage=multistage)(node)

    if component is not None:
        result = result._replace(cols=[result.cols[component]])

    assign_to = prepare_assign_to(assign_to, len(result.cols))
    if assign_to is not None:
        result = result._replace(cols=[col.label(lbl) for col, lbl in zip(result.cols, assign_to)])

    if include_key_col_in_query:
        id_cols = _try_get_key_columns_in_from_obj(from_obj, result.from_obj, key_column)
        if id_cols:
            result.cols.extend(id_cols)
        else:
            raise NotImplementedError("Key column cannot be added to sql expression for this node")
    result = sa.select(result.cols, from_obj=result.from_obj)

    if dialect is not None:
        result = to_sql(result, dialect)
    return result

def _get_key_col_name(index, prefix=None):
    prefix = prefix if prefix is not None else DEFAULT_KEY_LABEL_PREFIX
    return f"{prefix}{index}"

def _try_get_key_columns_in_from_obj(from_obj, result_from_obj, key_columns):
    actual_col_names_map = None
    if isinstance(result_from_obj, TableClause) and result_from_obj.name == from_obj:
        id_col_names = key_columns.copy()
    else:
        actual_col_names_map = {}
        id_col_names = []
        for i, col in enumerate(key_columns):
            key_col_name = _get_key_col_name(i)
            id_col_names.append(key_col_name)
            actual_col_names_map[key_col_name] = col
    try:
        added_col_names = set()
        cols = []
        for col in result_from_obj.columns:
            # in clauses like 'join' a key column might exist in multiple tables
            # "not in added_col_names" prevents multiple addition in this case.
            if col.name in id_col_names and col.name not in added_col_names:
                if actual_col_names_map:
                    cols.append(col.label(actual_col_names_map[col.name]))
                else:
                    cols.append(col)
                added_col_names.add(col.name)
        if cols:
            return cols
    except:
        pass
    return None


def _max(xs):
    if len(xs) == 1:
        return xs[0]
    return reduce(greatest, xs)


def _sum(iterable):
    "The built-in 'sum' does not work for us as we need."
    return reduce(lambda x, y: x + y, iterable)


def _iif(cond, iftrue, iffalse):
    # Optimize if (...) then X else X for literal X
    # A lot of these occur when compiling trees
    if isinstance(iftrue, sa.sql.elements.BindParameter) and \
            isinstance(iffalse, sa.sql.elements.BindParameter) and \
            iftrue.value == iffalse.value:
        return iftrue
    return sa.case([(cond, iftrue)], else_=iffalse)


def _matvecproduct(M, x):
    return [_sum(m_i[j] * x[j] for j in range(len(x))) for m_i in M]


def _dotproduct(xs, ys):
    return [_sum(x * y for x, y in zip(xs, ys))]


def _step(x):
    return _iif(x > 0, 1, 0)


def extract_tables(from_obj):
    if isinstance(from_obj, FromGrouping):
        return extract_tables(from_obj.element)
    elif isinstance(from_obj, Join):
        return extract_tables(from_obj.left) + extract_tables(from_obj.right)
    else:
        return [from_obj]


def _merge(tbl1, tbl2):
    if tbl1 is None:
        return tbl2
    elif tbl2 is None:
        return tbl1
    if tbl1 is tbl2:
        return tbl1
    # Either of the arguments may be a join clause and these
    # may include repeated elements. If so, we have to extract them and recombine.
    all_tables = list(sorted(set(extract_tables(tbl1) + extract_tables(tbl2)), key=lambda x: x.name))
    tbl1 = all_tables[0]
    joined = tbl1
    for tbl_next in all_tables[1:]:
        joined = joined.join(tbl_next, onclause=_get_onclause(tbl1, tbl_next))
        joined.keys_ = tbl1.keys_
    return joined


def _get_onclause(tbl_1, tbl_2):
    if len(tbl_1.keys_) > 1:
        return and_(*[tbl_1_key==tbl_2_key for tbl_1_key, tbl_2_key in zip(tbl_1.keys_, tbl_2.keys_)])
    key_col = tbl_1.keys_[0].name
    return tbl_1.columns[key_col] == tbl_2.columns[key_col]


Result = namedtuple('Result', 'cols from_obj')


class SQLAlchemyWriter(ASTProcessor, StandardOps, StandardArithmetics):
    """A SK AST processor, producing a SQLAlchemy "multistage" expression.
       The interpretation of each node is a tuple, containing a list of column expressions and a from_obj,
       where these columns must be queried from."""

    def __init__(self, from_obj='data', key_columns=['id'],
                 positive_infinity=float(np.finfo('float64').max),
                 negative_infinity=float(np.finfo('float64').min),
                 multistage=True):
        self.positive_infinity = positive_infinity
        self.negative_infinity = negative_infinity
        if multistage:
            if isinstance(from_obj, str):
                from_obj = sa.table(from_obj, *[sa.column(key_column) for key_column in key_columns])
                # This is a bit hackish, but quite convenient.
                # This way we do not have to carry around an extra "key" field in our results all the time
                from_obj.keys_ = [from_obj.columns[key_column] for key_column in key_columns]
            else:
                for key_column in key_columns:
                    if key_column not in from_obj.columns:
                        raise ValueError(
                            "The provided selectable does not contain the key column {0}".format(key_column))
                from_obj.keys_ = [from_obj.columns[key_column] for key_column in key_columns]
        elif isinstance(from_obj, str):
            from_obj = sa.table(from_obj)
        self.from_obj = from_obj
        self.temp_ids = id_generator()
        self.references = [{}]
        self.multistage = multistage

    def Identifier(self, id):
        return Result([sa.column(id.id)], self.from_obj)

    def _indexed_identifier(self, id, idx):
        return sa.column("{0}{1}".format(id, idx + 1))

    def IndexedIdentifier(self, sub):
        return Result([self._indexed_identifier(sub.id, sub.index)], self.from_obj)

    def _number_constant(self, value):
        # Infinities have to be handled separately
        if np.isinf(value):
            value = self.positive_infinity if value > 0 else self.negative_infinity
        else:
            value = denumpyfy(value)
        return sa.literal(value)

    def NumberConstant(self, num):
        return Result([self._number_constant(num.value)], self.from_obj)

    def VectorIdentifier(self, id):
        return Result([self._indexed_identifier(id.id, i) for i in range(id.size)], self.from_obj)

    def VectorConstant(self, vec):
        return Result([self._number_constant(v) for v in tolist(vec.value)], self.from_obj)

    def MatrixConstant(self, mtx):
        return Result([[self._number_constant(v) for v in tolist(row)] for row in mtx.value], self.from_obj)

    def UnaryFunc(self, node, **kw):
        arg = self(node.arg)
        if isinstance(node.op, ArgMax):
            return self._argmax(arg)
        elif isinstance(node.op, VecMax):
            return self._vecmax(arg)
        elif isinstance(node.op, VecSum):
            return self._vecsum(arg)
        elif isinstance(node.op, Softmax):
            return self._softmax(arg)
        else:
            op = self(node.op)
            return Result([op(el) for el in arg.cols], arg.from_obj)

    ArgMax = VecSumNormalize = VecSum = VecMax = Softmax = not_implemented

    def BinOp(self, node, **kw):
        left, right, op = self(node.left), self(node.right), self(node.op)
        if not isinstance(node.op, IsElemwise):
            # MatVecProduct requires atomizing the argument, otherwise it will be repeated multiple times in the output
            if not isinstance(node.right, IsAtom):
                right = self._make_cte(right)
            return Result(op(left.cols, right.cols), _merge(left.from_obj, right.from_obj))
        elif len(left.cols) != len(right.cols):
            raise ValueError("Mismatching operand dimensions in {0}".format(repr(node.op)))
        elif isinstance(node.op, Max):
            # Max is implemented as (if x > y then x else y), hence to avoid double-computation,
            # we save x and y in separate CTE's
            if not isinstance(node.left, IsAtom):
                left = self._make_cte(left)
            if not isinstance(node.right, IsAtom):
                right = self._make_cte(right)
            return Result([op(lc, rc) for lc, rc in zip(left.cols, right.cols)], _merge(left.from_obj, right.from_obj))
        else:
            return Result([op(lc, rc) for lc, rc in zip(left.cols, right.cols)], _merge(left.from_obj, right.from_obj))

    def MakeVector(self, vec):
        result = []
        tbls = set()
        for el in vec.elems:
            el = self(el)
            tbls.add(el.from_obj)
            if len(el.cols) != 1:
                raise ValueError("MakeVector expects a list of scalars")
            result.append(el.cols[0])
        tbls = list(tbls)
        target_table = tbls[0]
        for tbl in tbls[1:]:
            new_joined = target_table.join(tbl, onclause=_get_onclause(target_table, tbl))
            new_joined.keys_ = target_table.keys_
            target_table = new_joined
        return Result(result, target_table)

    def IfThenElse(self, node):
        test, iftrue, iffalse = self(node.test), self(node.iftrue), self(node.iffalse)

        return Result([_iif(test.cols[0], ift, iff) for ift, iff in zip(iftrue.cols, iffalse.cols)],
                      reduce(_merge, [test.from_obj, iftrue.from_obj, iffalse.from_obj]))

    MatVecProduct = is_(_matvecproduct)
    DotProduct = is_(_dotproduct)
    Exp = is_(sa.func.exp)
    Log = is_(sa.func.log)
    Sqrt = is_(sa.func.sqrt)
    Abs = is_(sa.func.abs)
    Step = is_(_step)
    Max = is_(lambda x, y: _max([x, y]))

    # ------ The actual "multi-stage" logic -----
    def Let(self, node, **kw):
        if not self.multistage:
            return StandardOps.Let(self, node, **kw)
        self.references.append({})
        for defn in node.defs:
            self.references[-1][defn.name] = self._make_cte(self(defn.body))
        result = self(node.body)
        self.references.pop()
        return result

    def Reference(self, node):
        if not self.multistage:
            raise ValueError("References are not supported in non-multistage mode")
        if node.name not in self.references[-1]:
            raise ValueError("Undefined reference: {0}".format(node.name))
        return self.references[-1][node.name]

    def _make_cte(self, result, col_names=None, key_label_prefix=DEFAULT_KEY_LABEL_PREFIX):
        if not self.multistage:
            return result
        if col_names is None:
            col_names = ['f{0}'.format(i + 1) for i in range(len(result.cols))]
        labeled_cols = [c.label(n) for c, n in zip(result.cols, col_names)]
        key_col_labels = [_get_key_col_name(i, key_label_prefix) for i in range(len(result.from_obj.keys_))]
        labeled_key_cols = [key_col.label(label) for label, key_col in zip(key_col_labels, result.from_obj.keys_)]
        new_tbl = sa.select(labeled_key_cols + labeled_cols, from_obj=result.from_obj).cte(
            next(self.temp_ids))
        new_tbl.keys_ = [new_tbl.columns[label] for label in key_col_labels]
        new_cols = [new_tbl.columns[n] for n in col_names]
        return Result(new_cols, new_tbl)

    def _argmax(self, result):
        if len(result.cols) == 1:
            return Result([sa.literal(0)], self.from_obj)
        features = self._make_cte(result)
        max_val = Result([_max(features.cols)], features.from_obj)
        max_val = self._make_cte(max_val, ['_max'])

        argmax = sa.case([(col == max_val.cols[0], i)
                          for i, col in enumerate(features.cols[:-1])],
                         else_=len(features.cols) - 1)
        return Result([argmax], _merge(features.from_obj, max_val.from_obj))

    def _vecmax(self, result):
        return Result([_max(result.cols)], result.from_obj)

    def _softmax(self, result):
        return self._vecsumnormalize(Result([sa.func.exp(col) for col in result.cols], result.from_obj))

    def _vecsumnormalize(self, result):
        features = self._make_cte(result)
        sum_val = Result([_sum(features.cols)], features.from_obj)
        sum_val = self._make_cte(sum_val, ['_sum'])
        return Result([col / sum_val.cols[0] for col in features.cols],
                      _merge(features.from_obj, sum_val.from_obj))

    def _vecsum(self, result):
        return Result([_sum(result.cols)], result.from_obj)


# ------- SQLAlchemy "greatest" function
# See https://docs.sqlalchemy.org/en/latest/core/compiler.html
# pylint: disable=wrong-import-position,wrong-import-order
from sqlalchemy.sql import expression, TableClause
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


# pylint: disable=wildcard-import,unused-wildcard-import
from sqlalchemy.dialects import *    # Must do it in order to getattr(sqlalchemy.dialects, ...)
def to_sql(sa_expr, dialect_name='sqlite'):
    """
    Helper function. Given a SQLAlchemy expression, returns the corresponding
    SQL string in a given dialect.
    """

    dialect_module = getattr(sqlalchemy.dialects, dialect_name)
    return str(sa_expr.compile(dialect=dialect_module.dialect(),
                               compile_kwargs={'literal_binds': True}))
