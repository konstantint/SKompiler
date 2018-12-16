"""
K-means implementation.
"""
from skompiler.dsl import const, func, let, defn, ref, vector


def k_means(cluster_centers, inputs, method):
    sq_dists = vector(let(defn(dx=inputs - const(c)),
                          ref('dx') @ ref('dx'))
                      for c in cluster_centers)
    if method == 'transform':
        return func.Sqrt(sq_dists)
    elif method == 'predict':
        return func.ArgMax(sq_dists * -1)
    else:
        raise ValueError("Unsupported methods for KMeans: {0}".format(method))
