"""
Converter from SKAST to Portable Format for Analytics
(http://dmg.org/pfa/)
"""
import json
from functools import reduce
import numpy as np
from skompiler import ast
from ._common import ASTProcessor, tolist, not_implemented, denumpyfy


def translate(expr, dialect=None):
    """
    Translates a given expression to PFA.

    Returns:
      - a Dict (if dialect is None)
      - a JSON string (if dialect is 'json')
      - a YAML string (if dialect is 'yaml'). This will invoke `import yaml`, so make sure PyYaml is installed.
    """

    ic = InputCollector()
    ast.map_tree(expr, ic)
    input_def = {
        "type": "record",
        "name": "Input",
        "fields": [{"name": k, "type": ({"type": "array", "items": "double"} if v else "double")} for k, v in ic.inputs.items()]
    }
    
    if expr._dtype == ast.DTYPE_VECTOR:
        output_def = {"type": "array", "items": "double"}
    else:
        output_def = {"type": "double"}
   
    writer = PFAWriter()
    result = {
        "input": input_def,
        "output": output_def,
        "action": writer(expr)
    }
    fcn_defs = {fn: ufuncs[fn] for fn in writer.fcns}
    if fcn_defs:
        result["fcns"] = fcn_defs
    
    if dialect is None:
        return result
    elif dialect == 'json':
        return json.dumps(result)
    elif dialect == 'yaml':
        import yaml
        return yaml.dump(result)
    else:
        raise ValueError("Unknown dialect: {0}".format(dialect))


ufuncs = {
    "usub": {"params": [{"x": "double"}],
             "ret": "double",
             "do": {"u-": "x"}},
    "mul": {"params": [{"x": "double"}, {"y": "double"}],
            "ret": "double",
            "do": {"*": ["x", "y"]}},
    "div": {"params": [{"x": "double"}, {"y": "double"}],
            "ret": "double",
            "do": {"/": ["x", "y"]}},
    "max": {"params": [{"x": "double"}, {"y": "double"}],
            "ret": "double",
            "do": {"max": ["x", "y"]}},
    "sigmoid": {"params": [{"x": "double"}],
                "ret": "double",
                "do": {"/": [1.0, {"+": [1, {"m.exp": {"u-": "x"}}]}]}},
    "step": {"params": [{"x": "double"}],
             "ret": "double",
             "do": {"if": {"<=": ["x", 0.0]}, "then": 0.0, "else": 1.0}},
    "vdot": {"params": [{"x": {"type": "array", "items": "double"}},
                        {"y": {"type": "array", "items": "double"}}],
             "ret": "double",
             "do": {"attr":
                    {"la.dot": [{"type": {"type": "array", "items": {"type": "array", "items": "double"}}, "new": ["x"]}, "y"]},
                    "path": [0]}
            },
    # In theory this should not be needed, however Python's PFA implementation somewhy crashes if I use m.abs without this wrapper.
    "abs": {"params": [{"x": "double"}],  
            "ret": "double",
            "do": {"m.abs": "x"}},
}


def is_fn(name, elemwise_name=None, scalar_name=None):
    def _fn(self, _, is_elemwise=False):
        fname = name
        if elemwise_name is not None and is_elemwise:
            fname = elemwise_name
        elif scalar_name is not None and not is_elemwise:
            fname = scalar_name
        if fname.startswith("u."):
            self.fcns.add(fname[2:])
        if is_elemwise and elemwise_name is None:
            def result(*args):
                fcn = [{"fcn": fname}]
                args = list(args)
                if len(args) > 1:
                    return {"a.zipmap": args + fcn}
                else:
                    return {"a.map": args + fcn}
            return result
        else:
            return lambda *args: {fname: list(args) if len(args) > 1 else args[0]}
    return _fn


class InputCollector:
    """Collects all input variable names from the expression"""
    
    def __init__(self):
        self.inputs = {}
        
    def __call__(self, node, _):
        if isinstance(node, ast.IsInput):
            self.inputs[node.id] = isinstance(node, ast.VectorIdentifier) or isinstance(node, ast.IndexedIdentifier)
        return node

class PFAWriter(ASTProcessor):
    def __init__(self):
        self.fcns = set()
    
    def Identifier(self, id):
        return "input.{0}".format(id.id)
        
    def IndexedIdentifier(self, sub):
        return "input.{0}.{1}".format(sub.id, sub.index)
    
    def _number_constant(self, value):
        # Infinities have to be handled separately
        if np.isinf(value):
            value = float('inf') if value > 0 else -float('inf')
        else:
            value = denumpyfy(value)
        return value

    def NumberConstant(self, num):
        return self._number_constant(num.value)

    def VectorIdentifier(self, id):
        return "input.{0}".format(id.id)

    def VectorConstant(self, vec):
        return {'type': {'type': 'array', 'items': 'double'}, 'value': [self._number_constant(v) for v in tolist(vec.value)]}

    def MakeVector(self, vec):
        return {'type': {'type': 'array', 'items': 'double'}, 'new': [self(el) for el in vec.elems]}
        
    def MatrixConstant(self, mtx):
        return {'type': {'type': 'array', 'items': {'type': 'array', 'items': 'double'}},
                'value': [[self._number_constant(v) for v in tolist(row)] for row in mtx.value]}

    def BinOp(self, node):
        left = self(node.left)
        right = self(node.right)
        is_elemwise = (isinstance(node.op, ast.IsElemwise) and
                       node.left._dtype == ast.DTYPE_VECTOR and node.right._dtype == ast.DTYPE_VECTOR)
        op = self(node.op, is_elemwise=is_elemwise)
        return op(left, right)

    def UnaryFunc(self, node):
        arg = self(node.arg)
        is_elemwise = isinstance(node.op, ast.IsElemwise) and node.arg._dtype == ast.DTYPE_VECTOR
        op = self(node.op, is_elemwise=is_elemwise)
        return op(arg)
    
    def IfThenElse(self, node):
        test, iftrue, iffalse = self(node.test), self(node.iftrue), self(node.iffalse)
        return {'if': test, 'then': iftrue, 'else': iffalse}
    
    VecMax = is_fn("a.max")
    ArgMax = is_fn("a.argmax")
    VecSum = is_fn("a.sum")
    Softmax = is_fn("m.link.softmax")
    
    MatVecProduct = is_fn("la.dot")
    DotProduct = is_fn("u.vdot")
    Exp = is_fn("m.exp")
    Log = is_fn("m.ln")
    Sqrt = is_fn("m.sqrt")
    Abs = is_fn("u.abs")
    Max = is_fn("u.max")

    Sigmoid = is_fn("u.sigmoid")
    Step = is_fn("u.step")
    Mul = is_fn("u.mul", scalar_name="*")
    Div = is_fn("u.div", scalar_name="/")
    Add = is_fn("+", elemwise_name="la.add")
    Sub = is_fn("-", elemwise_name="la.sub")
    USub = is_fn("u.usub", scalar_name="u-")
    LtEq = is_fn("<=")
    Eq = is_fn("==")
    
    def Let(self, node):
        result = [{'let': {defn.name: self(defn.body)}} for defn in node.defs]
        result.append(self(node.body))
        return {"do": result}

    def Reference(self, node):
        return node.name

    def LFold(self, node, **kw):
        # Standard implementation simply expands LFold into a sequence of BinOps and then calls itself
        if not node.elems:
            raise ValueError("LFold expects at least one element")
        return self(reduce(lambda x, y: ast.BinOp(node.op, x, y), node.elems), **kw)

    def TypedReference(self, node, **_):
        return self(ast.Reference(node.name))

    Definition = not_implemented
