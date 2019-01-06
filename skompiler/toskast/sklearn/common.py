"""
Common useful functionality.
"""
from skompiler.dsl import func, let, defn, ref, repeat


def classifier(probs, method):
    """Given a probability expression and a method name, returns the classifier output"""

    if method == 'predict_proba':
        return probs
    elif method == 'predict':
        return func.ArgMax(probs)
    elif method == 'predict_log_proba':
        return func.Log(probs)
    else:
        raise ValueError("Invalid method: {0}".format(method))

def vecsumnormalize(node, vector_dim):
    x = node
    s = func.VecSum(ref('x', x))
    return let(defn(x=x),
               defn(s=s),
               ref('x', x) / repeat(ref('s', s), vector_dim))

def sklearn_softmax(node, vector_dim):
    x = node
    xmax = func.VecMax(ref('x', x))
    xfix = ref('x', x) - repeat(ref('xmax', xmax), vector_dim)
    return let(defn(x=x),
               defn(xmax=xmax),
               defn(xfix=xfix),
               func.Softmax(ref('xfix', xfix)))
