"""
SKLearn model transformation to SKompiler's AST.
"""
#pylint: disable=unused-argument
# NB: Python 3.4+ has this in stdlib: from functools import singledispatch
from singledispatch import singledispatch
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model.base import LinearModel
from sklearn.svm import SVC, SVR
from sklearn.tree.tree import BaseDecisionTree, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor,\
                             GradientBoostingClassifier, GradientBoostingRegressor

from sklearn.preprocessing import Binarizer
from sklearn.pipeline import Pipeline

from .linear_model.logistic import logreg_binary, logreg_multiclass
from .linear_model.base import linear_model
from .tree.base import decision_tree
from .ensemble.forest import random_forest_classifier, random_forest_regressor
from .ensemble.gradient_boosting import gradient_boosting_classifier, gradient_boosting_regressor
from .preprocessing.data import binarize
from ...ast import VectorIdentifier, Identifier


@singledispatch
def translate(model, inputs='x', method='predict'):
    """
    Translate a given SKLearn model to a SK AST expression.

    Kwargs:

      inputs (string or ASTNode):
            The name of the variable that will be used to represent
            the input in the resulting expression. It can be given either as a
            single string (in this case the variable denotes an input vector),
            a list of strings (in this case each element denotes the name of one input component of a vector),
            or an ASTNode (e.g. VectorIdentifier, or MakeVector([Identifier('x'), Identifier('y')]))
        
      method (string):  Method to be expressed. Possible options:
            'predict', for all supported models.
            'decision_function', for all supported models.
            'predict_proba', 'predict_log_proba': for classifiers.
    """
    raise NotImplementedError("Conversion not implemented for {0}".format(model.__class__.__name__))

_supported_methods = {}

# An improved version of @translate.register decorator
def register(cls, methods):
    def decorator(fn):
        @translate.register(cls)
        def new_fn(model, inputs='x', method='predict'):
            if method not in methods:
                raise ValueError("Method {0} is not supported (or not implemented yet) for {1}".format(method, cls.__name__))
            return fn(model, inputs, method)
        return new_fn
    _supported_methods[cls] = methods
    return decorator


@register(LogisticRegression, ['decision_function', 'predict', 'predict_proba', 'predict_log_proba'])
def _(model, inputs, method):
    ovr = (model.multi_class in ["ovr", "warn"] or
           (model.multi_class == 'auto' and (model.classes_.size <= 2 or
                                             model.solver == 'liblinear')))
    if model.coef_.shape[0] == 1: # Binary logreg
        if not ovr:
            raise NotImplementedError("Logistic regression with binary outcomes and multinomial outputs is not implemented")
            # ... It's not too hard, actually, just need to find the 15 minutes needed to implement it.
        return logreg_binary(model.coef_.ravel(), model.intercept_[0], inputs=_prepare_inputs(inputs, model.coef_.shape[-1]), method=method)
    else: # Multiclass logreg
        return logreg_multiclass(model.coef_, model.intercept_, method=method,
                                 inputs=_prepare_inputs(inputs, model.coef_.shape[-1]), multi_class='ovr' if ovr else 'multinomial')

@register(SVC, ['decision_function', 'predict'])
def _(model, inputs, method):
    # For linear SVC the predict and decision function logic is the same as for logreg
    if model.kernel != 'linear':
        raise NotImplementedError("Translation for nonlinear SVC not implemented")
    if model.decision_function_shape != 'ovr':
        raise NotImplementedError("Translation not implemented for one-vs-one SVC")
    if len(model.classes_) > 2 and method != 'predict':
        raise NotImplementedError("Translation not implemented for non-binary SVC") # See sklearn.utils.multiclass._ovr_decision_function
    if model.coef_.shape[0] == 1: # Binary
        return logreg_binary(model.coef_.ravel(), model.intercept_[0], inputs=_prepare_inputs(inputs, model.coef_.shape[-1]), method=method)
    else: # Multiclass logreg
        return logreg_multiclass(model.coef_, model.intercept_, method=method,
                                 inputs=_prepare_inputs(inputs, model.coef_.shape[-1]), multi_class='ovr')

@register(SVR, ['predict'])
def _(model, inputs, method):
    if isinstance(model, SVR) and model.kernel != 'linear':
        raise NotImplementedError("Nonlinear SVR not implemented")
    return linear_model(model.coef_.ravel(), model.intercept_.item(), _prepare_inputs(inputs, model.coef_.shape[-1]))

@register(LinearModel, ['predict'])
def _(model, inputs, method):
    return linear_model(model.coef_.ravel(), model.intercept_, _prepare_inputs(inputs, model.coef_.shape[-1]))
    
@register(BaseDecisionTree, ['predict', 'predict_proba', 'predict_log_proba'])
def _(model, inputs, method):
    if isinstance(model, DecisionTreeRegressor) and method != 'predict':
        raise ValueError("Method {0} is not supported for DecisionTreeRegressor".format(method))
    return decision_tree(model.tree_, _prepare_inputs(inputs, model.n_features_), method)

@register(RandomForestClassifier, ['predict', 'predict_proba', 'predict_log_proba'])
def _(model, inputs, method):
    return random_forest_classifier(model, _prepare_inputs(inputs, model.n_features_), method)

@register(RandomForestRegressor, ['predict'])
def _(model, inputs, method):
    return random_forest_regressor(model, _prepare_inputs(inputs, model.n_features_))

@register(GradientBoostingClassifier, ['decision_function'])
def _(model, inputs, method):
    return gradient_boosting_classifier(model, _prepare_inputs(inputs, model.n_features_))

@register(GradientBoostingRegressor, ['predict'])
def _(model, inputs, method):
    return gradient_boosting_regressor(model, _prepare_inputs(inputs, model.n_features_))


@register(Binarizer, ['transform'])
def _(model, inputs, method):
    return binarize(model.threshold, _prepare_inputs(inputs))


@register(Pipeline, ['predict', 'predict_proba', 'decision_function', 'predict_log_proba', 'transform'])
def _(model, inputs, method):
    if not model.steps:
        raise ValueError("Empty pipeline provided")
    # The first step in the pipeline is responsible for preparing the inputs
    first_method = 'transform' if len(model.steps) > 1 else method
    expr = translate(model.steps[0][1], inputs, first_method)
    for i in range(1, len(model.steps)-1):
        expr = translate(model.steps[i][1], expr, 'transform')
    if len(model.steps) > 1:
        expr = translate(model.steps[-1][1], expr, method)
    return expr


def _prepare_inputs(inputs, n_features=None):
    if hasattr(inputs, '__next__'):
        # Unroll iterators
        inputs = [next(inputs) for i in range(n_features)]
    if isinstance(inputs, str):
        if not n_features:
            raise ValueError("Impossible to determine number of input variables")
        return VectorIdentifier(inputs, size=n_features)
    elif isinstance(inputs, list) and isinstance(inputs[0], str):
        if n_features is not None and len(inputs) != n_features:
            raise ValueError("The number of inputs must match the number of features in the tree")
        features = [Identifier(el) for el in inputs]
    else:
        features = inputs  # Assume we pass a list of input columns directly
    return features
