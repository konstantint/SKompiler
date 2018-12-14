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
{
$z = (12.2 + (x * y[-5]));
($z + 2)
}
"""
#pylint: disable=protected-access,multiple-statements,too-few-public-methods,no-member
from itertools import count


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
            raise Exception("Not enough arguments ({0}) given to {1}".format(len(args), self.__class__.__name__))
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
           supported output form.
           See project documentation for detailed explanation.

           Equivalent to skompiler.fromskast.<dialect[0]>.translate(self, dialect[1], *args, **kw)
           (where dialect[0] and dialect[1] denote the parts to the left and right of the '/' in the
            dialect parameter)

         Args:
         
            target (str): The target value. Possible values are:
            
               - 'sqlalchemy',
               - 'sqlalchemy/<dialect>', where <dialect> is on of the supported
                 SQLAlchemy dialects ('firebird', 'mssql', 'mysql', 'oracle',
                 'postgresql', 'sqlite', 'sybase')
               - 'sympy'
               - 'sympy/<lang>', where <lang> is either of:
                 'c', 'cxx', 'rust', 'fortran', 'js', 'r', 'julia',
                 'mathematica', 'octave'
               - 'excel'
               - 'python'
               - 'python/code'
               - 'python/lambda'
               - 'string'
            
        Kwargs:

            *args, **kw: Extra arguments are passed to the translation function.
                         
                The most important for `sqlalchemy` and `sympy/<lang>` dialects are:
                
                   assign_to:  When not None, the generated code outputs the result to a variable
                          (or variables, or columns) with given name(s).
                   component:  When not None and the expression produces a vector, only the
                            specified component of the vector is output (0-indexed)
                            
                For more info, see documentation for
                    skompiler.fromskast.<dialect[0]>.translate.

        '"""
        module, *dialect = target.split('/')
        try:
            from importlib import import_module
            mod = import_module('skompiler.fromskast.{0}'.format(module))
        except ModuleNotFoundError:
            raise ValueError("Invalid target: {0}".format(target))
        return mod.translate(self, *dialect, *args, **kw)

# Unary operators and functions
class UnaryFunc(ASTNode, fields='op arg', repr='{op}({arg})'): pass

# Some unary functions distribute over vectors. We mark them as such
class IsElemwise: pass
class USub(ASTNode, IsElemwise, repr='-'): pass     # Unary minus operator
class Exp(ASTNode, IsElemwise, repr='exp'): pass
class Log(ASTNode, IsElemwise, repr='log'): pass
class Step(ASTNode, IsElemwise, repr='step'): pass  # Heaviside step (x > 0)
class Sigmoid(ASTNode, IsElemwise, repr='sigmoid'): pass

# Some functions take vector arguments but do not distribute elementwise
class VecSumNormalize(ASTNode, repr='sum_normalize'): pass
class ArgMax(ASTNode, repr='argmax'): pass
class SKLearnSoftmax(ASTNode, repr='sklearn_softmax'): pass

# Binary operators
class BinOp(ASTNode, fields='op left right', repr='({left} {op} {right})'): pass
class Mul(ASTNode, IsElemwise, repr='*'): pass
class Add(ASTNode, IsElemwise, repr='+'): pass
class Sub(ASTNode, IsElemwise, repr='-'): pass
class Div(ASTNode, IsElemwise, repr='/'): pass
class DotProduct(ASTNode, repr='v@v'): pass
class MatVecProduct(ASTNode, repr='m@v'): pass

# Left-associative fold of an operator over a list of arguments
# E.g. sum(xs) := LFold(Add(), xs)
class LFold(ASTNode, fields='op elems'):
    def __str__(self):
        return str(self.op).join(str(e) for e in self.elems)

# Boolean binary ops
class IsBoolean: pass
class LtEq(ASTNode, IsElemwise, IsBoolean, repr='<='): pass

# IfThenElse
class IfThenElse(ASTNode, fields='test iftrue iffalse', repr='(if {test} then {iftrue} else {iffalse})'): pass

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
        defs = ';\n'.join(str(d) for d in self.defs)
        return '{\n' + defs + ';\n' + str(self.body) + '\n}'

class Definition(ASTNode, fields='name body', repr='${name} = {body}'): pass
class Reference(ASTNode, fields='name', repr='${name}'): pass


# --------------------- Utility functions -----------------------
def map_list(node_list, fn, call_on_enter=False, call_on_exit=True):
    new_list = []
    changed = False
    for node in node_list:
        new_node = map_tree(node, fn, call_on_enter, call_on_exit)
        if new_node is not node:
            changed = True
        new_list.append(new_node)
    return new_list if changed else node_list


def map_tree(node, fn, call_on_enter=False, call_on_exit=True):
    """Applies a function to each node in the tree.

    Args:
        fn: function, which must accept (node, is_entering) and return a node.
    
    Kwargs:
        call_on_enter:  Invoke the function (potentially replacing the node) every time before entering the node.
        call_on_exit:   Invoke the function (potentially replacing the node) every time before leavin the node.
    """
    if call_on_enter: node = fn(node, True)

    updates = {}
    for fname, fval in node:
        if isinstance(fval, ASTNode):
            new_val = map_tree(fval, fn, call_on_enter, call_on_exit)
        elif fname in ['elems', 'defs']:  # Special case for MakeVector and Let nodes
            new_val = map_list(fval, fn, call_on_enter, call_on_exit)
        else:
            new_val = fval
        if new_val is not fval:
            updates[fname] = new_val
    node = replace(node, **updates)

    if call_on_exit: node = fn(node, False)
    return node

def substitute_references(node, definitions):
    """Substitutes all references in the given expression with ones from the `definitions` dictionary.
       If any substitutions were made, returns a new node. Otherwise returns the same node.

       If references with no matches are found, raises a ValueError.

    Args:
       definitions (dict): a dictionary (name -> expr)

    >>> expr = BinOp(Add(), Identifier('x'), NumberConstant(2))
    >>> substitute_references(expr, {}) is expr
    True
    >>> expr.left = Reference('x')
    >>> substitute_references(expr, {})
    Traceback (most recent call last):
    ...
    ValueError: Unknown variable reference: x
    >>> new_expr = substitute_references(expr, {'x': expr})
    >>> print(new_expr)  # Only one level of references is expanded
    (($x + 2) + 2)
    >>> assert new_expr is not expr
    >>> assert new_expr.right is expr.right
    """
    def fn(node, _):
        if isinstance(node, Reference):
            if node.name not in definitions:
                raise ValueError("Unknown variable reference: " + node.name)
            else:
                return definitions[node.name]
        else:
            return node

    return map_tree(node, fn, call_on_enter=False, call_on_exit=True)


def inline_definitions(let_expr):
    """Given a Let expression, substitutes all the definitions and returns a single
       evaluatable non-let expression.

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
        defs[defn.name] = substitute_references(defn.body, defs)
    
    return substitute_references(let_expr.body, defs)


# --------- Let expression extraction ---------- #
def _scope_id_gen():
    yield ''  # Do not mangle the scope of the first let we find
    yield from map('_{0}'.format, count())

class LetCollector:
    def __init__(self):
        self.definitions = []
        self.scopes = []
        self.temp_ids = _scope_id_gen()

    def __call__(self, node, entering):
        if isinstance(node, Let):
            if entering:
                name = next(self.temp_ids)
                self.scopes.append(name)
            else:
                self.scopes.pop()
                return node.body
        elif isinstance(node, Definition) and not entering:
            self.definitions.append(replace(node, name='{0}{1}'.format(self.scopes[-1], node.name)))
        elif isinstance(node, Reference) and not entering:
            if not self.scopes:
                raise ValueError("Undefined reference: {0}".format(node.name))
            return replace(node, name='{0}{1}'.format(self.scopes[-1], node.name))
        return node

def merge_let_scopes(expr):
    """
    Given an expression which may potentially contain Let subexpressions inside,
    bubbles them all up and returns a single Let expression.
    If the original expression contained no Let subexpressions, returns it as-is.

    >>> from skompiler.toskast.string import translate as skast
    >>> let1 = skast("x=1; y=2; x+y")   # 3
    >>> let2 = skast("x=3; y=4; x+2*y") # 11
    >>> let3 = skast("x=0; y=0; 3*x+y") # 20
    >>> let3.defs[0].body=let1
    >>> let3.defs[1].body=let2
    >>> merged = merge_let_scopes(let3)
    >>> print(merged)
    {
    $_0x = 1;
    $_0y = 2;
    $x = ($_0x + $_0y);
    $_1x = 3;
    $_1y = 4;
    $y = ($_1x + (2 * $_1y));
    ((3 * $x) + $y)
    }
    >>> print(inline_definitions(merged))
    ((3 * (1 + 2)) + (3 + (2 * 4)))
    """
    lc = LetCollector()
    expr = map_tree(expr, lc, True, True)
    if not lc.definitions:
        return expr
    else:
        return Let(lc.definitions, expr)


# ---------- Helper functions for working with AST nodes
# We don't implement them as methods to avoid potential conflicts with
# field names ("lambdify", "evaluate" and "to" are the only exceptions as they might be used more often)

def copy(node):
    """
    Copies the node.

    >>> o = BinOp('+', 2, 3)
    >>> o2 = copy(o)
    >>> o.op = '-'
    >>> print(o, o2)
    (2 - 3) (2 + 3)
    """
    return node.__class__(**dict(node))

def replace(node, **kw):
    """Equivalent to namedtuple's _replace.
    When **kw is empty simply returns node itself.
    
    >>> o = BinOp('+', 2, 3)
    >>> o2 = replace(o, op='-', left=3)
    >>> print(o, o2)
    (2 + 3) (3 - 3)
    """
    if kw:
        new_node = copy(node)
        for k, v in kw.items():
            setattr(new_node, k, v)
        return new_node
    else:
        return node
