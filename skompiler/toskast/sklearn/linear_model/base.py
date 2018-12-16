"""
SKLearn linear model to SKAST.
"""
from skompiler.dsl import const

def linear_model(coef, intercept, inputs):
    """
    Linear regression.
    Depending on the shape of the coef and intercept, produces either a single-valued
    linear model (w @ x + b) or a multi-valued one (M @ x + b_vec)

    Args:

        coef (np.array): A vector (1D array, for single-valued model) or a matrix (2D array, for multi-valued one) for the model.
        intercept:  a number (for single-valued) or a 1D array (for multi-valued regression).
        inputs:  a list of AST nodes to be used as the input vector to the model or a single node, corresponding to a vector.
    """

    single_valued = (coef.ndim == 1)
    if single_valued and hasattr(intercept, '__iter__'):
        raise ValueError("Single-valued linear model must have a single value for the intercept")
    elif not single_valued and (coef.ndim != 2 or intercept.ndim != 1):
        raise ValueError("Multi-valued linear model must have a 2D coefficient matrix and a 1D intercept vector")

    return const(coef) @ inputs + const(intercept)
