"""
Base class for AST processors and functions, useful within multiple implementations
"""
#pylint: disable=not-callable
from itertools import count
from functools import reduce
from ..ast import AST_NODES, inline_definitions, IndexedIdentifier, NumberConstant, Exp, BinOp, IsElemwise, Reference


class ASTProcessorMeta(type):
    """A metaclass, which checks that the class defines methods for all known AST nodes.
       This is useful to verify SKAST processor implementations for completeness."""
    
    def __new__(mcs, name, bases, dct):
        if name != 'ASTProcessor':
            # This way the verification applies to all subclasses of ASTProcessor
            unimplemented = AST_NODES.difference(dct.keys())
            # Maybe the methods are implemented in one of the base classes?
            for base_cls in bases:
                unimplemented.difference_update(dir(base_cls))
            if unimplemented:
                raise ValueError(("Class {0} does not implement all the required ASTParser methods. "
                                  "Unimplemented methods: {1}").format(name, ', '.join(unimplemented)))
        return super().__new__(mcs, name, bases, dct)


class ASTProcessor(object, metaclass=ASTProcessorMeta):
    """
    The class hides the need to specify ASTProcessorMeta metaclass
    """

    def __call__(self, node, **kw):
        return getattr(self, node.__class__.__name__)(node, **kw)


def is_(val):
    return lambda self, node: val

def tolist(x):
    if hasattr(x, 'tolist'):
        return x.tolist()
    else:
        return list(x)

def not_implemented(self, node, *args, **kw):
    raise NotImplementedError("Processing of node {0} is not implemented.".format(node.__class__.__name__))

def _apply_bin_op(op_node, op, left, right):
    if (not isinstance(left, list) and not isinstance(right, list)) or not isinstance(op_node, IsElemwise):
        return op(left, right)
    if not isinstance(left, list) or not isinstance(right, list):
        raise ValueError("Elementwise operations requires both operands to be lists")
    if len(left) != len(right):
        raise ValueError("Sizes of the arguments do not match")
    return [op(l, r) for l, r in zip(left, right)]


class StandardOps:
    """Common implementation for BinOp, UnaryFunc, LFold, Let and ArgMin"""

    def BinOp(self, node, **kw):
        """Most common implementation for BinOp,
        If the arguments are lists and the op is elemwise, applies
        the operation elementwise and returns a list."""

        left = self(node.left, **kw)
        right = self(node.right, **kw)
        op = self(node.op, **kw)
        return _apply_bin_op(node.op, op, left, right)

    def UnaryFunc(self, node, **kw):
        op, arg = self(node.op, **kw), self(node.arg, **kw)
        if not isinstance(node.op, IsElemwise) or not isinstance(arg, list):
            return op(arg)
        else:
            return [op(a) for a in arg]
    
    def LFold(self, node, **kw):
        # Standard implementation simply expands LFold into a sequence of BinOps and then calls itself
        if not node.elems:
            raise ValueError("LFold expects at least one element")
        return self(reduce(lambda x, y: BinOp(node.op, x, y), node.elems), **kw)
    
    def Let(self, node, **kw):
        "Lazy implementation of the 'Let' node. Simply substitutes variables and proceeds as normal."
        return self(inline_definitions(node), **kw)

    Reference = Definition = not_implemented

    def TypedReference(self, node, **_):
        return self(Reference(node.name))


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
    
    def LFold(self, node, **kw):
        "If we know vectors are lists, we can improve LFold to avoid deep recursions"

        if not node.elems:
            raise ValueError("LFold expects at least one element")
        op = self(node.op, **kw)
        return reduce(lambda x, y: _apply_bin_op(node.op, op, x, y),
                      [self(el, **kw) for el in node.elems])

    VecSum = is_(lambda vec: reduce(lambda x, y: x + y, vec))


class StandardArithmetics:
    """A partial implementation of an AST processor,
    which assimes that:
        - all binary and unary operations are interpreted as lambda functions.
        - basic arithmetics and comparisons map to Python's basic arithmetics.
        - sigmoid can be expressed in terms of Exp as usual.
    """

    Mul = is_(lambda x, y: x * y)
    Div = is_(lambda x, y: x / y)
    Add = is_(lambda x, y: x + y)
    Sub = is_(lambda x, y: x - y)
    USub = is_(lambda x: -x)
    LtEq = is_(lambda x, y: x <= y)
    Eq = is_(lambda x, y: x == y)

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
    return map(template.format, count(start))


def denumpyfy(value):
    if hasattr(value, 'dtype'):
        return value.item()
    else:
        return value
