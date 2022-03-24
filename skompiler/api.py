"""
A convenience interface to SKompiler's functionality.
Wraps around the intricacies of the various toskast/fromskast pieces.
"""


def skompile(*args, inputs=None):
    """
    Creates a SKAST expression from a given bound method of a fitted SKLearn model.
    A shorthand notation for SKompiledModel(method, inputs)

    Args:

        args:  Either a bound method of a trained model (e.g. skompile(model.predict_proba)),
               OR two arguments - a model and a method name (e.g. skompile(model, 'predict_proba')
               (which may be necessary for some models where the first option cannot be used due to metaclasses)
        
        inputs:  A string or a list of strings, or a SKAST node or a list of SKAST nodes,
                 denoting the input variable(s) to your model.
                 A single string corresponds to a vector variable (which will be indexed to access
                 the components). A list of strings corresponds to a vector with separately named components.
               You may pass the inputs as a non-keyword argument as well (the last one in *args)
               If not specified, the default value of 'x' is used.
    
    Returns:
        An instance of SKompiledModel.

    Examples:
        >>> from sklearn.datasets import load_iris
        >>> from sklearn.linear_model import LogisticRegression
        >>> X, y = load_iris(return_X_y=True)
        >>> m = LogisticRegression().fit(X, y)
        >>> print(skompile(m.predict))
        argmax((([[...]] m@v x) + [ ...]))
        >>> print(skompile(m, 'predict'))
        argmax((([[...]] m@v x) + [ ...]))
        >>> print(skompile(m.predict, 'y'))
        argmax((([[...]] m@v y) + [ ...]))
        >>> print(skompile(m.predict, ['x','y','z','w']))
        argmax((([[...]] m@v [x, y, z, w]) + [ ...]))
        >>> print(skompile(m, 'predict', 'y'))
        argmax((([[...]] m@v y) + [ ...]))
        >>> print(skompile(m, 'predict', ['x','y','z','w']))
        argmax((([[...]] m@v [x, y, z, w]) + [ ...]))
        >>> from skompiler.ast import VectorIdentifier, Identifier
        >>> print(skompile(m, 'predict', VectorIdentifier('y', 4)))
        argmax((([[...]] m@v y) + [ ...]))
        >>> print(skompile(m, 'predict', map(Identifier, ['x','y','z','w'])))
        argmax((([[...]] m@v [x, y, z, w]) + [ ...]))
        >>> from sklearn.pipeline import Pipeline
        >>> p = Pipeline([('1', m)])
        >>> skompile(p.predict)
        Traceback (most recent call last):
        ...
        ValueError: The bound method ... Please, use the skompile(m, 'predict') syntax instead.
    """

    if len(args) > 3:
        raise ValueError("Too many arguments")
    elif not args:
        raise ValueError("Invalid arguments")
    elif len(args) == 3:
        if inputs is not None:
            raise ValueError("Too many arguments")
        model, method, inputs = args
    elif len(args) == 2:
        if hasattr(args[0], '__call__'):
            model, method = _get_model_and_method(args[0])
            inputs = args[1]
        else:
            model, method = args
    else:
        model, method = _get_model_and_method(args[0])
    if not inputs:
        inputs = 'x'
    return _translate(model, inputs, method)

def _translate(model, inputs, method):
    if model.__class__.__module__.startswith('keras.'):
        if method != 'predict':
            raise ValueError("Only the 'predict' method is supported for Keras models")
        # Import here, this way we do not force everyone to install everything
        from .toskast.keras import translate as from_keras
        return from_keras(model, inputs)
    else:
        from .toskast.sklearn import translate as from_sklearn
        return from_sklearn(model, inputs=inputs, method=method)

def _get_model_and_method(obj):
    if not hasattr(obj, '__call__'):
        raise ValueError("Please, provide a method to compile.")
    if not hasattr(obj, '__self__'):
        raise ValueError("The bound method object was probably mangled by "
                         "SKLearn's metaclasses and cannot be passed to skompile as skompile(m.predict). "
                         "Please, use the skompile(m, 'predict') syntax instead.")
    return obj.__self__, obj.__name__
