"""
Functions, useful within multiple ASTProcessor implementations
"""
from itertools import count
from ..ast import inline_definitions, IndexedIdentifier, NumberConstant, Exp

def is_(val):
    return lambda self, node: val

def tolist(x):
    if hasattr(x, 'tolist'):
        return x.tolist()
    else:
        return list(x)

def not_implemented(self, node, *args, **kw):
    raise NotImplementedError("Processing of node {0} is not implemented.".format(node.__class__.__name__))

def bin_op(self, op, **kw):
    "Most common implementation for BinOp or CompareBinOp"
    return self(op.op, **kw)(self(op.left, **kw), self(op.right, **kw))

def unary_func(self, op, **kw):
    "Common implementation for UnaryFunc"
    return self(op.op, **kw)(self(op.arg, **kw))

#pylint: disable=not-callable
class LazyLet:
    """
    A partial implementation of an AST processor,
    providing a "lazy" handling of Let expressions, where
    we simply expand the definitions and then proceed as normal.
    """

    Reference = Definition = not_implemented

    def Let(self, node):
        "Lazy implementation of the 'Let' node. Simply substitutes variables and proceeds as normal."
        return self(inline_definitions(node))


class VectorsAsLists:
    """A partial implementation of an AST processor,
    which assumes that:
        - all vectors are implemented as lists,
        - all element-wise operations operate element-wise on the lists,
        - all binary and unary operations are interpreted to lambda functions.

    The impementation of IfThenElse requires the class to have an
    _iif function, which corresponds to a unary IfThenElse.
    """

    _iif = not_implemented   # Must implement this method in subclasses

    UnaryFunc = unary_func
    BinOp = CompareBinOp = bin_op

    def ElemwiseBinOp(self, op):
        left = self(op.left)
        right = self(op.right)
        op = self(op.op)
        if not isinstance(left, list) or not isinstance(right, list):
            raise ValueError("Elementwise operations are only supported for vectors")
        if len(left) != len(right):
            raise ValueError("Sizes of the arguments do not match")
        return [op(l, r) for l, r in zip(left, right)]

    def ElemwiseUnaryFunc(self, op):
        arg = self(op.arg)
        if not isinstance(arg, list):
            raise ValueError("Elementwise operations are only supported for vectors")
        return list(map(self(op.op), arg))

    def VectorIdentifier(self, id):
        return [self(IndexedIdentifier(id.id, i, id.size)) for i in range(id.size)]

    def VectorConstant(self, vec):
        return [self(NumberConstant(v)) for v in tolist(vec.value)]

    def MatrixConstant(self, mtx):
        return [[self(NumberConstant(v)) for v in tolist(row)] for row in mtx.value]

    def MakeVector(self, vec):
        return [self(el) for el in vec.elems]

    def IfThenElse(self, node):
        """Implementation for IfThenElse for 'listwise' translators.
        Relies on the existence of self._iif function."""
        
        test, iftrue, iffalse = self(node.test), self(node.iftrue), self(node.iffalse)
        if isinstance(iftrue, list):
            if not isinstance(iffalse, list) or len(iftrue) != len(iffalse):
                raise ValueError("Mixed types in IfThenElse expressions are not supported")
            return [self._iif(test, ift, iff) for ift, iff in zip(iftrue, iffalse)]
        else:
            if isinstance(iffalse, list):
                raise ValueError("Mixed types in IfThenElse expressions are not supported")
            return self._iif(test, iftrue, iffalse)

class StandardArithmetics:
    """A partial implementation of an AST processor,
    which assimes that:
        - all binary and unary operations are interpreted as lambda functions.
        - basic arithmetics and comparisons map to Python's basic arithmetics.
        - sigmoid can be expressed in terms of Exp as usual.
    """

    UnaryFunc = unary_func
    BinOp = CompareBinOp = bin_op
    Mul = is_(lambda x, y: x * y)
    Div = is_(lambda x, y: x / y)
    Add = is_(lambda x, y: x + y)
    Sub = is_(lambda x, y: x - y)
    USub = is_(lambda x: -x)
    LtEq = is_(lambda x, y: x <= y)

    def Sigmoid(self, _):
        return lambda x: 1/(1 + self(Exp())(-x))


def prepare_assign_to(assign_to, n_actual_targets):
    """Converts the value of the assign_to parameter to a list of strings, as needed.
    
    >>> prepare_assign_to('x', 1)
    ['x']
    >>> prepare_assign_to('x', 2)
    ['x1', 'x2']
    >>> prepare_assign_to(['x'], 1)
    ['x']
    >>> prepare_assign_to(['a','b'], 2)
    ['a', 'b']
    >>> prepare_assign_to(None, 3)
    >>> prepare_assign_to(['a'], 2)
    Traceback (most recent call last):
    ...
    ValueError: The number of outputs (2) does not match the number of assign_to values (1)
    """

    if assign_to is None:
        return None

    if isinstance(assign_to, str):
        if n_actual_targets == 1:
            return [assign_to]
        else:
            return ['{0}{1}'.format(assign_to, i+1) for i in range(n_actual_targets)]
    
    if len(assign_to) != n_actual_targets:
        raise ValueError(("The number of outputs ({0}) does not match the number"
                          " of assign_to values ({1})").format(n_actual_targets, len(assign_to)))
    
    return assign_to


def id_generator(template='_tmp{0}', start=1):
    return map(template.format, count(start))  # NB: Py3-specific
