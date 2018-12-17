"""
Commonly useful functions.
"""
from skompiler.dsl import ident, vector


def is_(x):
    return lambda self, node, **kw: x

def prepare_inputs(inputs, n_features=None):
    if hasattr(inputs, '__next__'):
        # Unroll iterators
        inputs = [next(inputs) for i in range(n_features)]
    if isinstance(inputs, str):
        if not n_features:
            raise ValueError("Impossible to determine number of input variables")
        return ident(inputs, size=n_features)
    elif isinstance(inputs, list):
        if n_features is not None and len(inputs) != n_features:
            raise ValueError("The number of inputs must match the number of features in the tree")
        if isinstance(inputs[0], str):
            inputs = [ident(el) for el in inputs]
        return vector(inputs)
    else:
        return inputs
