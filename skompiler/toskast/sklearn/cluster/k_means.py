"""
K-means implementation.
"""
from skompiler.dsl import const, func, let, defn, ref, vector


def k_means(cluster_centers, inputs, method):
    res = []
    for c in cluster_centers:
        dx = inputs - const(c)
        res.append(let(defn(dx=dx), ref('dx', dx) @ ref('dx', dx)))
    
    sq_dists = vector(res)
    if method == 'transform':
        return func.Sqrt(sq_dists)
    elif method == 'predict':
        return func.ArgMax(sq_dists * -1)
    else:
        raise ValueError("Unsupported methods for KMeans: {0}".format(method))
