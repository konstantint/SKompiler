"""
SKompiler: AST nodes.

The classes here describe the AST nodes of the expressions produced
by SKompiler.

Notes:
    - We might have relied on Python's ast.* classes, but this
      would introduce unnecesary complexity and limit possibilities for
      adding custom nodes for special cases.
    - @dataclass would be a nice tech to use here, but it would prevent
      compatibility with Python older than 3.6

>>> expr = BinOp(Mul(), Identifier('x'), IndexedIdentifier('y', -5, 10))
>>> expr = BinOp(Add(), NumberConstant(12.2), expr)
>>> print(str(expr))
(12.2 + (x * y[-5]))
>>> expr = Let([Definition('z', expr)], BinOp(Add(), Reference('z'), NumberConstant(2)))
>>> print(str(expr))
$z = (12.2 + (x * y[-5]))
($z + 2)
"""
#pylint: disable=protected-access,multiple-statements,too-few-public-methods
#pylint: disable=redefined-builtin,no-member

# Each ASTNode registers its class name in this set
AST_NODES = set([])

class ASTNodeCreator(type):
    """A metaclass, which allows us to implement our AST nodes like mutable namedtuples
       with a nicer declaration syntax."""
    def __new__(mcs, name, bases, dct, fields='', repr=None):
        if fields is None:
            return super().__new__(mcs, name, bases, dct)
        else:
            cls = super().__new__(mcs, name, bases, dct)
            cls._fields = fields.split()
            cls._template = repr
            AST_NODES.add(name)
            return cls

class ASTNode(object, metaclass=ASTNodeCreator, fields=None):
    """Base class for all AST nodes. You may not instantiate it."""
    
    def __init__(self, *args, **kw):
        """Make sure the constructor arguments correspond to the _fields proprty"""
        
        if len(args) > len(self._fields): raise Exception("Too many arguments")
        args = dict(zip(self._fields, args))
        for k in args:
            if k in kw:
                raise Exception("Argument %s defined multiple times" % k)
        args.update(kw)
        if len(args) != len(self._fields):
            raise Exception("Not enough arguments")
        for k in self._fields:
            if k not in args:
                raise Exception("Argument %s not provided" % k)
            setattr(self, k, args[k])
    
    def __str__(self):
        dct = {k: str(v) for k, v in vars(self).items()}
        return self._template.format(**dct)

    def __repr__(self):
        return self.__class__.__name__ + '(' + ', '.join(k + '=' + repr(v) for k, v in self.__dict__.items()) + ')'

    def __setattr__(self, field, value):
        if field not in self._fields: raise Exception("Invalid attribute name: " + field)
        self.__dict__[field] = value

    def __iter__(self):
        for k in self._fields:
            yield k, getattr(self, k)
    
    def lambdify(self):
        """
        Converts the SKAST expression to an executable Python function.

        >>> from .toskast.string import translate as skast
        >>> skast("12.4 * (X[1] + Y)").lambdify()(**{'X': [10, 20, 30], 'Y': 1.2})
        262.88
        >>> skast("122.45 + 1").lambdify()()
        123.45
        >>> skast("x[0]").lambdify()(x=[[1]])
        [1]
        >>> skast("a=1; a").lambdify()()
        1
        >>> skast("a=b; b=b+1; c=b+b; a+b+c").lambdify()(b=1.0)
        7.0
        """
        from .fromskast import python

        return python.lambdify(python.translate(self))

    def evaluate(self, **inputs):
        "Convenience routine for evaluating expressions"
        return self.lambdify()(**inputs)

    def to(self, target, *args, **kw):
        """Convenience routine for converting expressions to any
           supported output form"""
        module, *dialect = target.split('/')
        try:
            from importlib import import_module
            mod = import_module('skompiler.fromskast.{0}'.format(module))
        except ModuleNotFoundError:
            raise ValueError("Invalid target: {0}".format(target))
        return mod.translate(self, *dialect, *args, **kw)

# Unary operators and functions
class UnaryFunc(ASTNode, fields='op arg', repr='{op}({arg})'): pass
class USub(ASTNode, repr='-'): pass     # Unary minus operator
#class Exp(ASTNode, repr='exp'): pass
class Log(ASTNode, repr='log'): pass
class Step(ASTNode, repr='step'): pass  # Heaviside step (x >= 0)

# Special functions
class VecSumNormalize(ASTNode, repr='sum_normalize'): pass
class ArgMax(ASTNode, repr='argmax'): pass
class SKLearnSoftmax(ASTNode, repr='sklearn_softmax'): pass
class Sigmoid(ASTNode, repr='sigmoid'): pass

class ElemwiseUnaryFunc(ASTNode, fields='op arg', repr='{op}({arg})'): pass

# Binary operators
class BinOp(ASTNode, fields='op left right', repr='({left} {op} {right})'): pass
class Mul(ASTNode, repr='*'): pass
class Add(ASTNode, repr='+'): pass
class Sub(ASTNode, repr='-'): pass
class Div(ASTNode, repr='/'): pass
class DotProduct(ASTNode, repr='v@v'): pass
class MatVecProduct(ASTNode, repr='m@v'): pass

class ElemwiseBinOp(ASTNode, fields='op left right', repr='({left} {op} {right})'): pass

# Comparison predicate
class CompareBinOp(ASTNode, fields='op left right', repr='({left} {op} {right})'): pass
class LtEq(ASTNode, repr='<='): pass

# IfThenElse
class IfThenElse(ASTNode, fields='test iftrue iffalse', repr='({iftrue} if {test} else {iffalse})'): pass

# Special function
class MakeVector(ASTNode, fields='elems'):
    def __str__(self):
        elems = ', '.join(str(e) for e in self.elems)
        return '[{0}]'.format(elems)

# Leaf nodes
class VectorIdentifier(ASTNode, fields='id size', repr='{id}'): pass
class Identifier(ASTNode, fields='id', repr='{id}'): pass

# Note that IndexedIdentifier is not a generic subscript operator. Its field must contain a string id and an integer index as well as the
# total size of the vector being indexed.
# This lets us "fake" vector input variables in contexts like SQL, where we interpret IndexedIdentifier("x", 1, 10) as a concatenated name "x1"
class IndexedIdentifier(ASTNode, fields='id index size', repr='{id}[{index}]'): pass
class NumberConstant(ASTNode, fields='value', repr='{value}'): pass
class VectorConstant(ASTNode, fields='value', repr='{value}'): pass
class MatrixConstant(ASTNode, fields='value', repr='{value}'): pass

# Variable definition
class Let(ASTNode, fields='defs body'):
    def __str__(self):
        defs = '\n'.join(str(d) for d in self.defs)
        return defs + '\n' + str(self.body)

class Definition(ASTNode, fields='name body', repr='${name} = {body}'): pass
class Reference(ASTNode, fields='name', repr='${name}'): pass


# --------------------- Utility functions -----------------------

def walk_tree(node, visitor_fn, parent=None, parent_field_name=None, preorder=True):
    """
    Walks through the AST, invoking visitor_fn(node, parent, parent_field_name) for each AST node.

    >>> expr = Definition('y', BinOp(Add(), Identifier('x'), NumberConstant(2)))
    >>> walk_tree(expr, print)
    $y = (x + 2) None None
    (x + 2) $y = (x + 2) body
    + (x + 2) op
    x (x + 2) left
    2 (x + 2) right
    """
    if preorder:
        visitor_fn(node, parent, parent_field_name)
    for fname, fval in node:
        if isinstance(fval, ASTNode):
            walk_tree(fval, visitor_fn, node, fname)
    if not preorder:
        visitor_fn(node, parent, parent_field_name)
    


def _substitute_refs(expr, definitions):
    """Walks through a given expression body and replaces all Reference nodes
       with the corresponding values from definitions in-place.

    Args:
       definitions (dict): a dictionary (name -> expr)
    
    >>> defs = {'x': NumberConstant(2), 'y': BinOp(Add(), Identifier('x'), Reference('z'))}
    >>> expr = BinOp(Mul(), BinOp(Add(), Reference('x'), Reference('y')), Reference('x'))
    >>> expr = _substitute_refs(expr, defs)
    >>> print(expr)
    ((2 + (x + $z)) * 2)
    """
    def substitute_ref(node, parent, parent_field_name):
        if isinstance(node, Reference):
            if node.name not in definitions:
                raise ValueError("Unknown variable reference: " + node.name)
            else:
                setattr(parent, parent_field_name, definitions[node.name])

    wrapper = UnaryFunc(arg=expr, op=None)  # Hackish way to handle the case when expr is a reference itself
    walk_tree(wrapper, substitute_ref)
    return wrapper.arg

def inline_definitions(let_expr):
    """Given a Let expression, substitutes all the definitions and returns a single
       evaluatable expression.

    >>> from .toskast.string import translate as skast
    >>> expr = inline_definitions(skast('a=1; a'))
    >>> print(str(expr))
    1
    >>> expr = inline_definitions(skast('a=1; b=a+a+2; c=b+b+3; a+b+c+X'))
    >>> print(str(expr))
    (((1 + ((1 + 1) + 2)) + ((((1 + 1) + 2) + ((1 + 1) + 2)) + 3)) + X)
    >>> expr = inline_definitions(skast('X+Y'))
    Traceback (most recent call last):
    ...
    ValueError: Let expression expected
    """

    if not isinstance(let_expr, Let):
        raise ValueError("Let expression expected")
    
    defs = {}
    for defn in let_expr.defs:
        defs[defn.name] = _substitute_refs(defn.body, defs)
    
    return _substitute_refs(let_expr.body, defs)


class ASTProcessorMeta(type):
    """A metaclass, which checks that the class defines methods for all known AST nodes.
       This is useful to verify SKAST processor implementations for completeness."""
    def __new__(mcs, name, bases, dct):
        if name != 'ASTProcessor':
            # This way the verification applies to all subclasses of ASTProcessor
            unimplemented = AST_NODES.difference(dct.keys())
            if unimplemented:
                raise ValueError(("Class {0} does not implement all the required ASTParser methods. "
                                  "Unimplemented methods: {1}").format(name, ', '.join(unimplemented)))
        return super().__new__(mcs, name, bases, dct)

class ASTProcessor(object, metaclass=ASTProcessorMeta):
    "The class hides the need to specify ASTProcessorMeta metaclass."

    def __call__(self, node, **kw):
        return getattr(self, node.__class__.__name__)(node, **kw)
