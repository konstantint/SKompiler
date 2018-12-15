"""
A convenience interface to SKompiler's functionality.
Wraps around the intricacies of the various toskast/fromskast pieces.
"""
from .toskast.sklearn import translate as from_sklearn


def skompile(model_or_method, inputs='x', method_name=None):
    """
    Creates a SKAST expression from a given bound method of a fitted SKLearn model.
    A shorthand notation for SKompiledModel(method, inputs)

    Args:

        model_or_method:  Either a bound method of a trained model which you need to compile.
                          (e.g. model.predict_proba), or the model object (in which case you need to
                          provide the method name in a separate parameter)
        
        inputs:  A string or a list of strings, denoting the input variables to your model.
                 A single string corresponds to a vector variable (which will be indexed to access
                 the components). A list of strings corresponds to a vector with separately named components.
        
        method_name:     If the first argument is a model object, specify the method name to generate code for.
    
    Returns:
        An instance of SKompiledModel.

    Examples:
    """
    if method_name is None:
        method_name = model_or_method.__name__
        model_or_method = model_or_method.__self__
    return from_sklearn(model_or_method, inputs=inputs, method=method_name)
