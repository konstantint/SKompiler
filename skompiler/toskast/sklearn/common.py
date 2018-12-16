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
    return let(defn(x=node),
               defn(s=func.VecSum(ref('x'))),
               ref('x') / repeat(ref('s'), vector_dim))

def sklearn_softmax(node, vector_dim):
    return let(defn(x=node),
               defn(xmax=func.VecMax(ref('x'))),
               defn(xfix=ref('x') - repeat(ref('xmax'), vector_dim)),
               func.Softmax(ref('xfix')))
