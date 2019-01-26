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
from importlib import import_module
import numpy as np


# Each ASTNode registers its class name in this set
AST_NODES = set([])

# This is the set of conversions supported in the node.to(...) method
TRANSLATORS = {
    'excel': 'skompiler.fromskast.excel:translate',
    'python': 'skompiler.fromskast.python:translate',
    'sqlalchemy': 'skompiler.fromskast.sqlalchemy:translate',
    'sympy': 'skompiler.fromskast.sympy:translate',
    'pfa': 'skompiler.fromskast.pfa:translate',
    'string': str
}


#region ASTNode base -------------------------------------------------

# Basic type inference (NB: not an enum to keep compatibility with Python 3.3. Maybe it is lost already, though)
DTYPE_SCALAR = 1
DTYPE_VECTOR = 2
DTYPE_MATRIX = 3
DTYPE_OTHER = 42
DTYPE_UNKNOWN = None
class ASTTypeError(ValueError): pass
class UnableToDecompose(ValueError): pass


class ASTNodeCreator(type):
    """A metaclass, which allows us to implement our AST nodes like mutable namedtuples
       with a nicer declaration syntax."""
    def __new__(mcs, name, bases, dct, fields='', repr=None, dtype=DTYPE_OTHER):
        if fields is None:
            return super().__new__(mcs, name, bases, dct)
        else:
            cls = super().__new__(mcs, name, bases, dct)
            cls._fields = fields.split()
            cls._template = repr
            cls._default_dtype = dtype
            AST_NODES.add(name)
            return cls

    # For Python 3.5, see https://stackoverflow.com/a/25191150/318964
    def __init__(cls, name, bases, dct, **_):
        super().__init__(name, bases, dct)


_singletons = {}

class ASTNode(object, metaclass=ASTNodeCreator, fields=None):
    """Base class for all AST nodes. You may not instantiate it."""

    def __new__(cls, *_args, **_kw):
        # Save some memory on singletons
        if hasattr(cls, '_fields') and not cls._fields:
            # Singleton node
            name = cls.__name__
            if _singletons.get(name, None) is None:
                _singletons[name] = super().__new__(cls)
            return _singletons[name]
        return super().__new__(cls)

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
            self.__dict__[k] = args[k]
        self.__dict__['_dtype'] = self._compute_dtype()
    
    def __str__(self):
        dct = {k: str(v) for k, v in vars(self).items()}
        return self._template.format(**dct)

    def __repr__(self):
        return self.__class__.__name__ + '(' + ', '.join(k + '=' + repr(getattr(self, k)) for k in self._fields) + ')'

    def __setattr__(self, field, value):
        raise Exception("AST nodes are immutable")

    def __iter__(self):
        "Iteration over node fields. Note that the node works like a dict.items(), not like a tuple"
        for k in self._fields:
            yield k, getattr(self, k)

    def __bool__(self):
        return True
    
    # Convenience routines for combining AST nodes
    def __len__(self):
        raise UnableToDecompose()
    
    def __getitem__(self, idx):
        raise UnableToDecompose()

    def __add__(self, other):
        other = self._align_scalar(other)
        return BinOp(Add(), self, other)
    
    def __mul__(self, other):
        other = self._align_scalar(other)
        return BinOp(Mul(), self, other)
    
    def __truediv__(self, other):
        other = self._align_scalar(other)
        return BinOp(Div(), self, other)
    
    def __sub__(self, other):
        other = self._align_scalar(other)
        return BinOp(Sub(), self, other)
    
    def __matmul__(self, other):
        if self._dtype == DTYPE_MATRIX:
            return BinOp(MatVecProduct(), self, other)
        else:
            return BinOp(DotProduct(), self, other)

    def __le__(self, other):
        other = self._align_scalar(other)
        return BinOp(LtEq(), self, other)
    
    def __eq__(self, other):
        other = self._align_scalar(other)
        return BinOp(Eq(), self, other)
    
    def __call__(self, arg, arg2=None):
        if arg2 is not None:
            return BinOp(self, arg, arg2)
        else:
            return UnaryFunc(self, arg)
    
    def _align_scalar(self, other):
        # If a scalar is given in a binary operation,
        # we try to align it in length with us
        # This is hackish and should be used with caution
        if np.isscalar(other):
            self_type = self._dtype
            if self_type == DTYPE_SCALAR:
                return NumberConstant(other)
            elif self_type == DTYPE_VECTOR:
                return VectorConstant([other]*len(self))
            else:
                raise ASTTypeError("Unable to align scalar with a node of type {0}".format(self.__class__.__name__))
        else:
            return other

    # Type inference
    def _compute_dtype(self):
        return self._default_dtype

    # Convenience routines for evaluating AST nodes
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

    # The main convenience function, which converts the node to any of the
    # supported formats
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
               - 'pfa'
               - 'pfa/json'
               - 'pfa/yaml'
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
        translator, *dialect = target.split('/')
        if translator not in TRANSLATORS:
            raise ValueError("Invalid translator: {0}".format(translator))
        translator = TRANSLATORS[translator]
        if not hasattr(translator, '__call__'):
            module, callable = translator.split(':')
            mod = import_module(module)
            translator = getattr(mod, callable)
        return translator(self, *dialect, *args, **kw)
#endregion


#region AST node types ---------------------------------------------

# Unary operators and functions
class UnaryFunc(ASTNode, fields='op arg', repr='{op}({arg})'):
    def __len__(self):
        if isinstance(self.op, IsElemwise):
            return len(self.arg)
        elif getattr(self.op, '_out_dtype', None) == DTYPE_SCALAR:
            return 1
        elif isinstance(self.op, Softmax):
            return len(self.arg)
        else:
            raise UnableToDecompose()
    
    def __getitem__(self, index):
        if isinstance(self.op, IsElemwise):
            return UnaryFunc(self.op, self.arg[index])
        else:
            raise UnableToDecompose()

    def _compute_dtype(self):
        if self.arg._dtype not in [DTYPE_UNKNOWN, DTYPE_SCALAR, DTYPE_VECTOR]:
            raise ASTTypeError()
        if isinstance(self.op, IsElemwise):
            return self.arg._dtype
        else:
            return self.op._out_dtype

# Some unary functions distribute over vectors. We mark them as such
class IsElemwise: pass
class USub(ASTNode, IsElemwise, repr='-'): pass     # Unary minus operator
class Exp(ASTNode, IsElemwise, repr='exp'): pass
class Log(ASTNode, IsElemwise, repr='log'): pass
class Abs(ASTNode, IsElemwise, repr='abs'): pass
class Sqrt(ASTNode, IsElemwise, repr='sqrt'): pass
class Step(ASTNode, IsElemwise, repr='step'): pass  # Heaviside step (x > 0)
class Sigmoid(ASTNode, IsElemwise, repr='sigmoid'): pass

# Some functions take vector arguments but do not distribute elementwise
class VecSum(ASTNode, repr='sum'):
    _out_dtype = DTYPE_SCALAR
class VecMax(ASTNode, repr='max'):
    _out_dtype = DTYPE_SCALAR
class ArgMax(ASTNode, repr='argmax'):
    _out_dtype = DTYPE_SCALAR
class Softmax(ASTNode, repr='softmax'):
    _out_dtype = DTYPE_VECTOR

# Binary operators
def _common_dtype(nodes):
    dtypes = {n._dtype for n in nodes if n._dtype is not DTYPE_UNKNOWN}
    if not dtypes:
        return DTYPE_UNKNOWN
    elif len(dtypes) == 2:
        raise ASTTypeError("Mismatching operand types")
    else:
        result = next(iter(dtypes))
        if result not in [DTYPE_SCALAR, DTYPE_VECTOR]:
            raise ASTTypeError("Arguments must be scalars or vectors")
        return result


class BinOp(ASTNode, fields='op left right', repr='({left} {op} {right})'):
    def __len__(self):
        if isinstance(self.op, IsElemwise) or isinstance(self.op, MatVecProduct):
            return len(self.left)
        else:
            raise UnableToDecompose()
    
    def __getitem__(self, index):
        if isinstance(self.op, IsElemwise):
            return BinOp(self.op, self.left[index], self.right[index])
        elif isinstance(self.op, MatVecProduct):
            return BinOp(DotProduct(), self.left[index], self.right)
        else:
            raise UnableToDecompose()

    def _compute_dtype(self):
        if isinstance(self.op, IsElemwise):
            return _common_dtype([self.left, self.right])
        else:
            return self.op._out_dtype

class Mul(ASTNode, IsElemwise, repr='*'): pass
class Add(ASTNode, IsElemwise, repr='+'): pass
class Sub(ASTNode, IsElemwise, repr='-'): pass
class Div(ASTNode, IsElemwise, repr='/'): pass
class Max(ASTNode, IsElemwise, repr='max'): pass
class DotProduct(ASTNode, repr='v@v'):
    _out_dtype = DTYPE_SCALAR
class MatVecProduct(ASTNode, repr='m@v'):
    _out_dtype = DTYPE_VECTOR

class LFold(ASTNode, fields='op elems'):
    """
    Left-associative fold of an operator over a list of arguments
    E.g. sum(xs) := LFold(Add(), xs)
    This could be represented as a sequence of binary ops, but having a
    dedicated operator may significantly reduce the depth of AST trees with long sums
    which you may find in ensemble classifiers.
    Deep trees are bad because you run the risk of hitting system recursion limit when processing
    them via recursive parsers (as is the case currently)
    """

    def __str__(self):
        return str(self.op).join(str(e) for e in self.elems)

    def __getitem__(self, idx):
        return LFold(self.op, [el[idx] for el in self.elems])

    def __len__(self):
        if not self.elems:
            raise ASTTypeError("Empty LFolds are not allowed")
        return len(self.elems[0])

    def _compute_dtype(self):
        return _common_dtype(self.elems)

# Boolean binary ops
class IsBoolean: pass
class LtEq(ASTNode, IsElemwise, IsBoolean, repr='<='): pass
class Eq(ASTNode, IsElemwise, IsBoolean, repr='=='): pass


# IfThenElse
class IfThenElse(ASTNode, fields='test iftrue iffalse', repr='(if {test} then {iftrue} else {iffalse})'):
    def __len__(self):
        return len(self.iftrue)
    
    def __getitem__(self, index):
        return IfThenElse(self.test, self.iftrue[index], self.iffalse[index])

    def _compute_dtype(self):
        return _common_dtype([self.iftrue, self.iffalse])

# Special function
class MakeVector(ASTNode, fields='elems', dtype=DTYPE_VECTOR):
    def __str__(self):
        elems = ', '.join(str(e) for e in self.elems)
        return '[{0}]'.format(elems)

    def __getitem__(self, idx):
        return self.elems[idx]

    def __len__(self):
        return len(self.elems)

# Leaf nodes
class IsAtom: pass
class IsInput: pass
class VectorIdentifier(ASTNode, IsAtom, IsInput, fields='id size', repr='{id}', dtype=DTYPE_VECTOR):
    def __getitem__(self, index):
        return IndexedIdentifier(self.id, index, self.size)
    def __len__(self):
        return self.size

class Identifier(ASTNode, IsAtom, IsInput, fields='id', repr='{id}', dtype=DTYPE_SCALAR): pass

# Note that IndexedIdentifier is not a generic subscript operator. Its field must contain a string id and an integer index as well as the
# total size of the vector being indexed.
# This lets us "fake" vector input variables in contexts like SQL, where we interpret IndexedIdentifier("x", 1, 10) as a concatenated name "x1"
class IndexedIdentifier(ASTNode, IsAtom, IsInput, fields='id index size', repr='{id}[{index}]', dtype=DTYPE_SCALAR): pass
class NumberConstant(ASTNode, IsAtom, fields='value', repr='{value}', dtype=DTYPE_SCALAR): pass
class VectorConstant(ASTNode, IsAtom, fields='value', repr='{value}', dtype=DTYPE_VECTOR):
    def __getitem__(self, index):
        return NumberConstant(self.value[index])
    def __len__(self):
        return len(self.value)

class MatrixConstant(ASTNode, IsAtom, fields='value', repr='{value}', dtype=DTYPE_MATRIX):
    def __len__(self):
        return len(self.value)
    def __getitem__(self, index):
        return VectorConstant(self.value[index])

# Variable definitions
# NB: at the moment let-scopes do not capture the outside variables.
# I.e. all the references inside the definitions and body of a let expressions are assumed to
# refer to variables defined in this let scope
class Let(ASTNode, fields='defs body'):
    def __str__(self):
        defs = ';\n'.join(str(d) for d in self.defs)
        return '{\n' + defs + ';\n' + str(self.body) + '\n}'
    def __len__(self):
        return len(self.body)
    def __getitem__(self, idx):
        return self.body[idx]
    def _compute_dtype(self):
        return self.body._dtype

class Definition(ASTNode, fields='name body', repr='${name} = {body}'):
    def _compute_dtype(self):
        return self.body._dtype

class Reference(ASTNode, fields='name', repr='${name}', dtype=DTYPE_UNKNOWN): pass

# Sometimes marking the type and dimension of the object referenced to is convenient
class TypedReference(ASTNode, fields='name dtype size', repr='${name}'):
    def _compute_dtype(self):
        return self.dtype
    def __len__(self):
        return self.size

#endregion


#region Node processing functions -----------------------
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
    >>> expr = replace(expr, left = Reference('x'))
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
        if isinstance(node, Reference) or isinstance(node, TypedReference):
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
    
    let_expr = merge_let_scopes(let_expr)
    
    defs = {}
    for defn in let_expr.defs:
        defs[defn.name] = substitute_references(defn.body, defs)
    
    return substitute_references(let_expr.body, defs)


# Let expression extraction
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
        elif (isinstance(node, Reference) or isinstance(node, TypedReference)) and not entering:
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
    >>> let3.defs[0].__dict__['body']=let1
    >>> let3.defs[1].__dict__['body']=let2
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


# Node copying and field replacing methods
# NB: We don't implement them as methods to avoid potential conflicts with
# field names ("lambdify", "evaluate" and "to" are the only exceptions as they might be used more often)

def copy(node):
    """
    Copies the node.

    >>> o = BinOp(Add(), NumberConstant(2), NumberConstant(3))
    >>> o2 = copy(o)
    >>> o.__dict__['op'] = Sub()
    >>> print(o, o2)
    (2 - 3) (2 + 3)
    """
    return node.__class__(**dict(node))

def replace(node, **kw):
    """Equivalent to namedtuple's _replace.
    When **kw is empty simply returns node itself.
    
    >>> o = BinOp(Add(), NumberConstant(2), NumberConstant(3))
    >>> o2 = replace(o, op=Sub(), left=NumberConstant(3))
    >>> print(o, o2)
    (2 + 3) (3 - 3)
    """
    if kw:
        vals = dict(node)
        vals.update(**kw)
        return node.__class__(**vals)
    else:
        return node

def decompose(node):
    return [node[i] for i in range(len(node))]

#endregion
