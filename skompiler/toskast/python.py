'''
Python AST to SKAST translator.

May only convert nodes of Python AST which have a mapping in SK-AST.
Useful for debugging testing and simplistic parsing mostly:


>>> expr = ast.parse("12.4 * (X1[25.3] + Y)")
>>> print(str(translate(expr)))
(12.4 * (X1[25.3] + Y))
>>> expr = ast.parse("a=12+b; b=2*a; 12.4 * (X[25.3] + Y + 2*a*b)")
>>> print(str(translate(expr)))
{
$a = (12 + b);
$b = (2 * $a);
(12.4 * ((X[25.3] + Y) + ((2 * $a) * $b)))
}
'''
#pylint: disable=wildcard-import,unused-wildcard-import,unused-argument
import ast
from skompiler.ast import *
from ._common import is_

def translate(node):
    return PythonASTProcessor()(node)

_funcmap = {
    'log': Log(),
    'exp': Exp(),
    'step': Step(),
    'sqrt': Sqrt(),
    'abs': Abs(),
}

class PythonASTProcessor:

    def __call__(self, node, **kw):
        cls = node.__class__.__name__
        if not hasattr(self, cls):
            raise NotImplementedError("No translation logic implemented for node type {0}".format(cls))
        return getattr(self, cls)(node, **kw)

    def Module(self, module, local_varnames=None):
        # Module may have more than one statement. We only allow a sequence of Assign statements,
        # followed by an expression
        definitions = []
        local_varnames = local_varnames or set()
        for assn in module.body[:-1]:
            if not isinstance(assn, ast.Assign):
                raise NotImplementedError("Only a sequence of assignments followed by an expression is allowed")
            if len(assn.targets) != 1:
                raise NotImplementedError("Assignment to a single variable allowed only")
            if not isinstance(assn.targets[0], ast.Name):
                raise NotImplementedError("Assignment may only be done to a named variable")
            varname = assn.targets[0].id
            definitions.append(Definition(varname, self(assn.value, local_varnames=local_varnames)))
            local_varnames.add(varname)

        body = self(module.body[-1], local_varnames=local_varnames)
        if local_varnames:
            return Let(definitions, body)
        return body

    def Expr(self, expr, **kw):
        return self(expr.value, **kw)

    def Expression(self, expr, **kw):
        return self(expr.body, **kw)

    def Name(self, name, local_varnames=None):
        if local_varnames and name.id in local_varnames:
            return Reference(name.id)
        else:
            return Identifier(name.id)

    def Subscript(self, sub, local_varnames=None):
        if not isinstance(sub.value, ast.Name):
            raise NotImplementedError("Unsupported form of subscript")
        if local_varnames and sub.value.id in local_varnames:
            raise ValueError("Subscripting named references is not supported")
        if isinstance(sub.slice, ast.Index) and isinstance(sub.slice.value, ast.Num):
            return IndexedIdentifier(id=sub.value.id,
                                    index=sub.slice.value.n,
                                    size=None) # This makes Sympy sad
        elif isinstance(sub.slice, ast.Constant):
            return IndexedIdentifier(id=sub.value.id,
                                    index=sub.slice.value,
                                    size=None)
        else:
            raise NotImplementedError("Unsupported form of subscript")

    def Constant(self, const, **kw):
        # Starting from Py3.8 AST for constants is just Constant, rather than Num/Str/NameConstant
        if type(const.value) not in [int, float]:
            raise ValueError("Only numeric constants are supported")
        return NumberConstant(const.value)

    def Num(self, num, **kw):
        return NumberConstant(num.n)

    def UnaryOp(self, op, **kw):
        return UnaryFunc(op=self(op.op, **kw),
                         arg=self(op.operand, **kw))

    def BinOp(self, op, **kw):
        return BinOp(op=self(op.op, **kw),
                     left=self(op.left, **kw),
                     right=self(op.right, **kw))
    
    def Call(self, call, **kw):
        if not isinstance(call.func, ast.Name) or call.keywords or len(call.args) != 1:
            raise ValueError("Only one-argument functions are supported")
        if call.func.id not in _funcmap:
            raise ValueError("Unsupported unary function: " + call.func.id)
        return UnaryFunc(op=_funcmap[call.func.id], arg=self(call.args[0], **kw))

    def IfExp(self, ifexp, **kw):
        return IfThenElse(self(ifexp.test, **kw), self(ifexp.body, **kw), self(ifexp.orelse, **kw))
    
    def Compare(self, cmp, **kw):
        if len(cmp.comparators) != 1 or len(cmp.ops) != 1:
            raise ValueError("Only one-element comparison expressions are supported")
        return BinOp(self(cmp.ops[0], **kw), self(cmp.left, **kw), self(cmp.comparators[0], **kw))
    
    def List(self, lst, **kw):
        return MakeVector([self(el, **kw) for el in lst.elts])
    
    Mult = is_(Mul())
    Add = is_(Add())
    Sub = is_(Sub())
    USub = is_(USub())
    LtE = is_(LtEq())
    Div = is_(Div())
