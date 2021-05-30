"""
SKompiler: Generate Python AST expressions from SKAST.
Useful for testing and evaluating expressions.

Note that the expressions with a Let condition are compiled differently to
those without.

>>> from ..toskast import string

Bare expressions without Let compile to bare Python expressions.
You need to wrap them in ast.Expression to use and may eval to get a value:

>>> expr = string.translate("12.4 * (b[1] + Y[0])")
>>> pyast = translate(expr)
>>> code = compile(ast.Expression(body=pyast), "__main__", "eval")
>>> eval(code, {'b': [10, 20, 30], 'Y': [1.2]})
262.88

Let-expressions compile to ast.Module with multiple expressions, the last of which
writes the result value to the __result__ global. You may only exec this code:

>>> expr = string.translate("a=X[1]; b=a+1; 12.4 * (b + Y[0])")
>>> pyast = translate(expr)
>>> code = compile(pyast, "__main__", "exec")
>>> vars = {}
>>> eval(code, {'X': [10, 20, 30], 'Y': [1.2]}, vars)
>>> vars['__result__']
275.28

In general, use utils.evaluate to evaluate SKAST expressions.
"""
import ast
import sys
import numpy as np
from ..ast import USub, Identifier, NumberConstant, IsBoolean, merge_let_scopes, Max
from ._common import ASTProcessor, StandardOps, denumpyfy


_linearg = dict(lineno=1, col_offset=0) # Most Python AST nodes require these

def translate(node, dialect=None):
    """
    When dialect is None, translates the given SK AST expression to a Python AST tree.
    Otherwise, further converts the tree depending on the value of dialect, to:
    
      'code':   Python source code (via the astor package)
      'lambda': Executable function (via expr.lambdify)
    
    >>> from skompiler.toskast.string import translate as skast
    >>> expr = skast('[2*x[0], 1] if x[1] <= 3 else [12.0, 45.5]')
    >>> print(translate(expr, 'code'))
    (np.array([2 * x[0], 1]) if x[1] <= 3 else np.array([12.0, 45.5]))
    <BLANKLINE>
    >>> fn = translate(expr, 'lambda')
    >>> fn(x=[1, 2])
    array([2, 1])
    """
    pyast = PythonASTWriter()(merge_let_scopes(node))
    if dialect is None:
        return pyast
    elif dialect == 'lambda':
        return lambdify(pyast)
    elif dialect == 'code':
        import astor
        code = astor.to_source(pyast)
        # Replace some internal identifiers with matching functions
        code = code.replace('__np__', 'np').replace('__exp__', 'np.exp').replace('__log__', 'np.log')
        code = code.replace('__argmax__', 'np.argmax').replace('__sum__', 'np.sum')
        return code
    else:
        raise ValueError("Unknown dialect: {0}".format(dialect))

def _ident(name):
    "Shorthand for defining methods in PythonASTWriter (see code below)"
    return lambda self, x: self(Identifier(name))

def _is(node):
    "Shorthand for defining methods in PythonASTWriter (see code below)"
    return lambda self, x: node

class PythonASTWriter(ASTProcessor, StandardOps):
    """
    An AST processor, which translates a given SKAST node to a Python AST.

    >>> import ast
    >>> topy = PythonASTWriter()
    >>> print(ast.dump(topy(Identifier('x'))))
    Name(id='x', ctx=Load())
    """

    def Identifier(self, name):
        return ast.Name(id=name.id, ctx=ast.Load(), **_linearg)
    VectorIdentifier = Identifier

    def IndexedIdentifier(self, sub):
        return ast.Subscript(value=self(Identifier(sub.id)),
                             slice=ast.Index(value=self(NumberConstant(sub.index))),
                             ctx=ast.Load(), **_linearg)

    def NumberConstant(self, num):
        return ast.Num(n=denumpyfy(num.value), **_linearg)

    def VectorConstant(self, vec):
        result = ast.parse('__np__.array()', mode='eval').body
        result.args = [ast.List(elts=[ast.Num(n=denumpyfy(el), **_linearg) for el in vec.value],
                                ctx=ast.Load(), **_linearg)]
        return result

    def MakeVector(self, mv):
        result = ast.parse('__np__.array()', mode='eval').body
        result.args = [ast.List(elts=[self(el) for el in mv.elems],
                                ctx=ast.Load(), **_linearg)]
        return result

    def MatrixConstant(self, mat):
        result = ast.parse('__np__.array()', mode='eval').body
        result.args = [ast.List(elts=[ast.List(elts=[ast.Num(n=denumpyfy(el), **_linearg) for el in row],
                                               ctx=ast.Load(), **_linearg) for row in mat.value], ctx=ast.Load(), **_linearg)]
        return result

    def UnaryFunc(self, node, **kw):
        if isinstance(node.op, USub):
            return ast.UnaryOp(op=self(node.op), operand=self(node.arg), **_linearg)
        else:
            return ast.Call(func=self(node.op), args=[self(node.arg)], keywords=[], **_linearg)

    def BinOp(self, node, **kw):
        op, left, right = self(node.op), self(node.left), self(node.right)
        if isinstance(node.op, IsBoolean):
            return ast.Compare(left=left, ops=[op], comparators=[right], **_linearg)
        elif isinstance(node.op, Max):
            return ast.Call(func=self(node.op), args=[self(node.left), self(node.right)], keywords=[], **_linearg)
        else:
            return ast.BinOp(op=op, left=left, right=right, **_linearg)
    
    def IfThenElse(self, node):
        return ast.IfExp(test=self(node.test), body=self(node.iftrue), orelse=self(node.iffalse), **_linearg)
    
    def Let(self, node, **kw):
        code = [ast.Assign(targets=[ast.Name(id='_def_' + defn.name, ctx=ast.Store(), **_linearg)],
                           value=self(defn.body), **_linearg) for defn in node.defs]
        # Evaluate the expression body into a "__result__" variable
        code.append(
            ast.Assign(targets=[ast.Name(id='__result__', ctx=ast.Store(), **_linearg)],
                       value=self(node.body), **_linearg))
        return ast.Module(body=code,
                          **({} if sys.version < '3.8' else {"type_ignores": []}))

    def Reference(self, ref):
        return ast.Name(id='_def_' + ref.name, ctx=ast.Load(), **_linearg)
    TypedReference = Reference

    # Functions
    Exp = _ident('__exp__')
    Sqrt = _ident('__sqrt__')
    Log = _ident('__log__')
    Step = _ident('__step__')
    VecSum = _ident('__sum__')
    ArgMax = _ident('__argmax__')
    Sigmoid = _ident('__sigmoid__')
    Softmax = _ident('__softmax__')
    VecMax = _ident('__vecmax__')
    Max = _ident('__max__')
    Abs = _ident('__abs__')

    # Operators
    Mul = _is(ast.Mult())
    Div = _is(ast.Div())
    Add = _is(ast.Add())
    Sub = _is(ast.Sub())
    USub = _is(ast.USub())
    DotProduct = _is(ast.MatMult())
    MatVecProduct = DotProduct

    # Predicates
    LtEq = _is(ast.LtE())
    Eq = _is(ast.Eq())

# ------------- Evaluation of Python AST-s --------------- #

def _softmax(X):
    X = np.exp(X)
    sum_prob = np.sum(X, axis=1).reshape((-1, 1))
    X /= sum_prob
    return X

_eval_vars = {
    '__np__': np,
    '__exp__': np.exp,
    '__sqrt__': np.sqrt,
    '__log__': np.log,
    '__sum__': np.sum,
    '__argmax__': np.argmax,
    '__vecmax__': np.max,
    '__max__': np.maximum,
    '__abs__': np.abs,
    '__sigmoid__': lambda z: 1.0/(1.0 + np.exp(-z)),
    '__sum_normalize__': lambda x: x / np.sum(x),
    '__softmax__': lambda x: _softmax([x])[0, :],
    '__step__': lambda x: 1 if x > 0 else 0  # This is how step is implemented in LogisticRegression
}

def lambdify(pyast):
    """
    Converts a given Python AST, produced by PythonASTWriter to an executable Python function.

    >>> from ..ast import NumberConstant, BinOp, Mul, Identifier
    >>> pyast = translate(BinOp(Mul(), NumberConstant(2), Identifier('x')))
    >>> fn = lambdify(pyast)
    >>> fn(x=3.14)
    6.28
    """
    if isinstance(pyast, ast.Module):
        # Exec the code:
        code = compile(pyast, "__main__", "exec")
        def result(**inputs):
            globals_ = {}
            globals_.update(_eval_vars)
            eval(code, inputs, globals_)  # pylint: disable=eval-used
            return globals_['__result__']
    else:
        # Eval the code:
        code = compile(ast.Expression(body=pyast), "__main__", "eval")
        def result(**inputs):
            globals_ = {}
            globals_.update(_eval_vars)
            return eval(code, inputs, globals_)  # pylint: disable=eval-used
    return result
