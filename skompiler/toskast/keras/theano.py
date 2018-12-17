"""
Translation for Theano graphs (limited to those generatable via Keras default operations)
"""
#pylint: disable=protected-access
from collections import OrderedDict
from skompiler import ast
from skompiler.dsl import const, func, vector, ref
from .._common import is_


def _dot(x, y):
    xtype = x._dtype
    ytype = y._dtype
    if xtype == ast.DTYPE_VECTOR and ytype == ast.DTYPE_VECTOR:
        return x @ y
    elif xtype == ast.DTYPE_VECTOR and ytype == ast.DTYPE_MATRIX:
        # Keras multiplies matrices on the right, we reverse this
        return y @ x
    else:
        raise NotImplementedError("Dot product not implemented for given arguments")

def _auto_broadcasting(elemwise_func):
    # The theano graph actually contains explicit instructions
    # on how the values must be broadcast. However, at the moment we do not carry
    # this information with us during translation and simply assume that the only
    # kind of broadcasting possible is turning a constant into a vector.
    def fn(x, y=None):
        if y is not None:
            xtype, ytype = x._dtype, y._dtype
            if xtype == ast.DTYPE_VECTOR and ytype == ast.DTYPE_SCALAR:
                y = vector([y]*len(x))
            elif xtype == ast.DTYPE_SCALAR and ytype == ast.DTYPE_VECTOR:
                x = vector([x]*len(y))
            return elemwise_func(x, y)
        else:
            return elemwise_func(x)
    return fn

_elemwise_ops = {
    'Add': func.Add,
    'Mul': func.Mul,
    'Abs': func.Abs,
    'Tanh': lambda x: func.Sigmoid(x*2)*2 - 1,
    'ScalarSigmoid': func.Sigmoid,
    'TrueDiv': func.Div,
    'Cast': lambda x: x, # Ignore casts for now
    'LT': func.LtEq,     # TODO: Support for LessThan
}


class TheanoTranslator:
    def __init__(self, input_var_name, inputs):
        self.input_var_name = input_var_name
        self.inputs = inputs
        self.definitions = OrderedDict()
    
    def __call__(self, node, **kw):
        cname = node.__class__.__name__
        return getattr(self, cname)(node, **kw)
    
    def TensorVariable(self, node):
        if node.name == self.input_var_name:
            return self.inputs
        elif node.name == 'keras_learning_phase':
            return False
        elif node.owner is None:
            raise NotImplementedError("Do not know how to handle tensor variable {0}".format(node.name))
        else:
            return self(node.owner)
    
    def TensorSharedVariable(self, node):
        if node.name is None:
            return const(node.container.value)
        refname = node.name.replace('/', '_')
        if refname not in self.definitions:
            res = const(node.container.value)
            if isinstance(res, ast.MatrixConstant):
                # Keras always multiplies matrices on the right, we prefer the other way
                res = const(res.value.T)
            self.definitions[refname] = res
        return ref(refname, self.definitions[refname])

    def TensorConstant(self, node):
        return const(node.data)

    def Apply(self, node):
        # Special case to support Dropout without the need to parse the training phase branch
        if node.op.__class__.__name__ == 'IfElse':
            test = self(node.inputs[0])
            if isinstance(test, bool):
                return self(node.inputs[1]) if test else self(node.inputs[2])
            else:
                return ast.IfThenElse(test, self(node.inputs[1]), self(node.inputs[2]))
        op = self(node.op)
        inputs = [self(inp) for inp in node.inputs]
        return op(*inputs)

    def DimShuffle(self, node):
        o = node.new_order
        if o not in [('x', 'x'), ('x', 0)]:
            raise NotImplementedError("Unusual DimShuffle not implemented")
        # If o is ('x', 'x'), the argument is a constant that must be broadcasted
        # If o is ('x', 0) the argument is a vector that is taken as-i
        return lambda x: x

    def Elemwise(self, node):
        opname = node.scalar_op.__class__.__name__
        op = _elemwise_ops.get(opname, None)
        if not op:
            raise NotImplementedError("Elemwise operation {0} not implemented".format(opname))
        return _auto_broadcasting(op)

    Dot = is_(_dot)
    Softmax = is_(func.Softmax)
