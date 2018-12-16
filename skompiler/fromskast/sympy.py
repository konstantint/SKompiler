"""
SKompiler: Generate Sympy expressions from SKAST.
"""
import numpy as np
import sympy as sp
from ..ast import IsElemwise, Mul
from ._common import ASTProcessor, is_, StandardOps, StandardArithmetics


def translate(node, dialect=None, true_argmax=True, assign_to='y', component=None, lambdify_inputs_str='x', **kw):
    """Translates SKAST to a Sympy expression and optionally generates code from it.

    KwArgs:
        dialect (string):   If None, returns the Sympy expression. Otherwise translates it further to one of the supported languages.
                            Supported values:
                                'c', 'cxx', 'rust', 'fortran', 'js', 'r', 'julia', 'mathematica', 'octave', 'lambda'.
                            If dialect == 'lambda', the expression is lambdified (the call is then equivalent to
                            lambdify(lambdify_inputs_str, translate(node)))
        
        true_argmax (bool): When True (default), the generated expression will include a "Sympy-executable" definition of argmax(vector)
                            (in the form of a lengthy expression "if v[0] == max(v) then 0, else if v[1] == max(v) then 1, ...")
                            When False, the expression will contain just the name "argmax". This may be sufficient
                            if you only need Sympy as an intermediate representation before compiling into a different language.

        assign_to:          This value is passed further to sympy code printers when dialect is not None.
                            When assign_to is not None, it specifies that the code printer should generate assignment statements,
                            setting values of the given variable. Otherwise, a pure expression is generated (which is not always
                            possible if the expression is a matrix)
        
        component (int):    If the result is a vector, return only the specified component of it.

        lambdify_inputs_str (str): The string specifying lambdified function inputs (when dialect == 'lambda')

        **kw:               Other arguments passed to the code generator (when dialect is not None)
    
    >>> from skompiler.toskast.string import translate as skast
    >>> expr = skast('[2*x[0], 1] if x[1] <= 3 else [12.0, 45.5]')
    >>> print(translate(expr, 'js'))
    if (x_1 <= 3) {
       y[0] = 2*x_0;
    }
    else {
       y[0] = 12.0;
    }
    if (x_1 <= 3) {
       y[1] = 1;
    }
    else {
       y[1] = 45.5;
    }
    >>> print(translate(expr, 'js', assign_to=None, component=0))
    ((x_1 <= 3) ? (
       2*x_0
    )
    : (
       12.0
    ))
    """
    syexpr = SympyWriter(true_argmax=true_argmax)(node)
    if component is not None:
        syexpr = syexpr[component]
    if dialect is None:
        return syexpr
    elif dialect == 'lambda':
        return lambdify(lambdify_inputs_str, syexpr)
    else:
        return to_code(syexpr, dialect, assign_to=assign_to, **kw)

def _argmax(val):
    "A sympy implementation of argmax"

    maxval = sp.Max(*val)
    pieces = [(i, sp.Eq(val[i], maxval)) for i in range(len(val)-1)]
    pieces.append((len(val)-1, True))
    return sp.Piecewise(*pieces)

def _max(val):
    return sp.Max(*val)

def _softmax(vec):
    "A sympy implementation of sklearn's softmax"
    sexp = [sp.exp(vec[i]) for i in range(len(vec))]
    return sp.ImmutableMatrix([sexp[i] / sum(sexp) for i in range(len(sexp))])


class SympyWriter(ASTProcessor, StandardOps, StandardArithmetics):
    """A SK AST processor, producing a Sympy expression"""

    def __init__(self, true_argmax=False):
        self.true_argmax = true_argmax
    
    def Identifier(self, id):
        return sp.symbols(id.id)

    def VectorIdentifier(self, node):
        # This is not the best option, because this way all our vector inputs
        # must be 2D matrices (for purposes of lambdify as well as printingc:w
        x = sp.MatrixSymbol(node.id, node.size, 1)
        return sp.ImmutableMatrix([x[i] for i in range(node.size)])

        # NB: This alone won't work, because Dot operator wants to have an actual matrix as input
        # return sp.MatrixSymbol(node.id, node.size, 1)
        #
        # This (and  the version with x = sp.IndexedBase('x')) is not good because it breaks code printers
        # (apparently they expect all indices to have lower and upper bounds which does not happen if you provide
        #  numeric indices, or smth like that, probably a bug somewhere in sympy)
        # x = sp.MatrixSymbol(node.id, node.size, 1)
        # return sp.ImmutableMatrix([x[i] for i in range(node.size)])

    def IndexedIdentifier(self, sub):
        if sub.size is None:  # Hack to handle Python-written expressions. We do not represent them as "true" indexed values in Sympy
            return sp.symbols('{0}_{1}'.format(sub.id, sub.index))
        else:
            return sp.IndexedBase(sub.id, shape=(sub.size,))[sub.index]

    def NumberConstant(self, num):
        return sp.sympify(num.value)

    def VectorConstant(self, vec):
        return sp.ImmutableMatrix(vec.value)

    MatrixConstant = VectorConstant

    def MakeVector(self, vec):
        return sp.ImmutableMatrix([self(el) for el in vec.elems])

    def UnaryFunc(self, node, **kw):
        arg = self(node.arg)
        op = self(node.op)
        if isinstance(arg, sp.MatrixBase) and isinstance(node.op, IsElemwise):
            if arg.shape[1] != 1:
                raise NotImplementedError("Elementwise operations are only supported for vectors (column matrices)")
            return sp.ImmutableMatrix([op(arg[i]) for i in range(len(arg))])
        else:
            return op(arg)

    def BinOp(self, node, **kw):
        left = self(node.left)
        right = self(node.right)
        op = self(node.op)
        if isinstance(left, sp.MatrixBase) and isinstance(right, sp.MatrixBase) and isinstance(node.op, IsElemwise):
            if left.shape != right.shape:
                raise ValueError("Shapes of the arguments do not match")
            if left.shape[1] != 1:
                raise NotImplementedError("Elementwise operations are only supported for vectors (column matrices)")
            return sp.ImmutableMatrix([op(left[i], right[i]) for i in range(len(left))])
        else:
            return op(left, right)
    
    def IfThenElse(self, node):
        # Piecewise function with matrix output is not a Matrix itself, which breaks some of the logic
        # Hence this won't work in general:
        test, iftrue, iffalse = self(node.test), self(node.iftrue), self(node.iffalse)
        if hasattr(iftrue, 'shape'):
            if iftrue.shape != iffalse.shape:
                raise ValueError("Shapes of the IF branches must match")
            if iftrue.shape[1] != 1:
                raise NotImplementedError("Elementwise operations are only supported for vectors (column matrices)")
            return sp.ImmutableMatrix([sp.Piecewise((ift, test), (iff, True)) for ift, iff in zip(iftrue, iffalse)])
        else:
            return sp.Piecewise((iftrue, test), (iffalse, True))

    def MatVecProduct(self, _):
        return self(Mul())
    
    DotProduct = is_(lambda x, y: x.dot(y))
    Exp = is_(sp.exp)
    Log = is_(sp.log)
    Step = is_(sp.Heaviside)
    Sqrt = is_(sp.sqrt)
    Abs = is_(sp.Abs)
    VecSum = is_(sum) # Yes, we return a Python summation here
    Softmax = is_(_softmax)
    VecMax = is_(_max)
    Max = is_(lambda x, y: _max([x, y]))
    Eq = is_(sp.Eq)

    def ArgMax(self, _):
        return _argmax if self.true_argmax else sp.Function('argmax')

# Utility function

def _softmax(X):
    X = np.exp(X)
    sum_prob = np.sum(X, axis=1).reshape((-1, 1))
    X /= sum_prob
    return X

_lambdify_modules = ["numpy",
                     {"Heaviside": lambda x: int(x > 0),
                      "softmax": lambda x: _softmax([x])[0, :]
                     }]

def lambdify(sympy_inputs_str, sympy_expr):
    return sp.lambdify(sp.symbols(sympy_inputs_str), sympy_expr, modules=_lambdify_modules)



_ufns = {'argmax': 'argmax'}

#pylint: disable=unnecessary-lambda
_code_printers = {
    'c': lambda expr, **kw: sp.ccode(expr, standard='c99', user_functions=_ufns, **kw),
    'cxx': lambda expr, **kw: sp.cxxcode(expr, user_functions=_ufns, **kw),
    'rust': lambda expr, **kw: sp.rust_code(expr, user_functions=_ufns, **kw),
    'fortran': lambda expr, **kw: sp.fcode(expr, standard=95, user_functions=_ufns, **kw),
    'js': lambda expr, **kw: sp.jscode(expr, user_functions=_ufns, **kw),
    'r': lambda expr, **kw: sp.rcode(expr, user_functions=_ufns, **kw),
    'julia': lambda expr, **kw: sp.julia_code(expr, user_functions=_ufns, **kw),
    'mathematica': lambda expr, assign_to=None, **kw: sp.mathematica_code(expr, user_functions=_ufns, **kw),
    'octave': lambda expr, **kw: sp.octave_code(expr, user_functions=_ufns, **kw),
}

def to_code(syexpr, dialect, assign_to='y', **kw):
    '''
    Shorthand for converting the resulting Sympy expression to code in various languages.


    Args:

        dialect (str): The target language. Can be one of:
            'c', 'cxx', 'rust', 'fortran', 'js', 'r', 'julia', 'mathematica', 'octave'.

    Kwargs:

        assign_to (str or sympy expression):
            Passed to the sympy code printers.
            If this is not None, the generated code is a set of assignments to the target variable
            (or, of the result is an array, to the components of the target array).
            Note that if assign_to is None, and the resulting expression is a vector (e.g. predict_proba vector
            of probabilities), sympy may be unable to generate meaningful code for many languages,
            because those lack the "Matrix" datatype (hence it cannot be the result of a one-liner expression)
    '''
    if dialect not in _code_printers:
        raise ValueError("Unknown dialect: ")
    return _code_printers[dialect](syexpr, assign_to=assign_to, **kw)
