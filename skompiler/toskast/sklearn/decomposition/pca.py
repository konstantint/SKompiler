"""
PCA implementation
"""
import numpy as np
from skompiler.dsl import const

def pca(model, inputs):
    matrix = np.array(model.components_)
    if model.whiten:
        matrix /= np.sqrt(model.explained_variance_)[:, np.newaxis]
    if model.mean_ is not None:
        inputs = inputs - const(model.mean_)
    return const(matrix) @ inputs
