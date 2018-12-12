"""
A convenience interface to SKompiler's functionality.
Wraps around the intricacies of the various toskast/fromskast pieces.
"""
from .toskast.sklearn import translate as from_sklearn

def skompile(method, inputs='x'):
    """
    Creates a SKAST expression from a given bound method of a fitted SKLearn model.
    A shorthand notation for SKompiledModel(method, inputs)

    Args:

        method:  A bound method of a trained model which you need to compile.
                 For example "model.predict_proba"
        
        inputs:  A string or a list of strings, denoting the input variables to your model.
                 A single string corresponds to a vector variable (which will be indexed to access
                 the components). A list of strings corresponds to a vector with separately named components.
    
    Returns:
        An instance of SKompiledModel.

    Examples:
    """
    return from_sklearn(method.__self__, inputs=inputs, method=method.__name__)
