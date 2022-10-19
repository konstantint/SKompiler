"""
SKLearn model transformation to SKompiler's AST.
"""
#pylint: disable=unused-argument
from functools import singledispatch
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model._base import LinearModel
from sklearn.svm import SVC, SVR
from sklearn.tree._classes import BaseDecisionTree, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor,\
                             GradientBoostingClassifier, GradientBoostingRegressor,\
                             AdaBoostClassifier
from sklearn.cluster import KMeans
from sklearn.decomposition._pca import _BasePCA
from sklearn.neural_network import MLPClassifier, MLPRegressor

from sklearn.preprocessing import Binarizer, MinMaxScaler, MaxAbsScaler, StandardScaler,\
                                  Normalizer
from sklearn.pipeline import Pipeline

from skompiler.dsl import ident, vector

from .._common import prepare_inputs
from .linear_model.logistic import logreg_binary, logreg_multiclass
from .linear_model.base import linear_model
from .tree.base import decision_tree
from .ensemble.forest import random_forest_classifier, random_forest_regressor
from .ensemble.gradient_boosting import gradient_boosting_classifier, gradient_boosting_regressor
from .ensemble.weight_boosting import adaboost_classifier
from .cluster.k_means import k_means
from .decomposition.pca import pca
from .neural_network.multilayer_perceptron import mlp, mlp_classifier
from .preprocessing.data import binarize, scale, unscale, standard_scaler, normalizer


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
            or an ASTNode (e.g. VectorIdentifier, or vector([ident('x'), ident('y')]))
        
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
        return logreg_binary(model.coef_.ravel(), model.intercept_[0], inputs=prepare_inputs(inputs, model.coef_.shape[-1]), method=method)
    else: # Multiclass logreg
        return logreg_multiclass(model.coef_, model.intercept_, method=method,
                                 inputs=prepare_inputs(inputs, model.coef_.shape[-1]), multi_class='ovr' if ovr else 'multinomial')

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
        return logreg_binary(model.coef_.ravel(), model.intercept_[0], inputs=prepare_inputs(inputs, model.coef_.shape[-1]), method=method)
    else: # Multiclass logreg
        return logreg_multiclass(model.coef_, model.intercept_, method=method,
                                 inputs=prepare_inputs(inputs, model.coef_.shape[-1]), multi_class='ovr')

@register(SVR, ['predict'])
def _(model, inputs, method):
    if isinstance(model, SVR) and model.kernel != 'linear':
        raise NotImplementedError("Nonlinear SVR not implemented")
    return linear_model(model.coef_.ravel(), model.intercept_.item(), prepare_inputs(inputs, model.coef_.shape[-1]))

@register(LinearModel, ['predict'])
def _(model, inputs, method):
    return linear_model(model.coef_.ravel(), model.intercept_, prepare_inputs(inputs, model.coef_.shape[-1]))
    
@register(BaseDecisionTree, ['predict', 'predict_proba', 'predict_log_proba'])
def _(model, inputs, method):
    if isinstance(model, DecisionTreeRegressor) and method != 'predict':
        raise ValueError("Method {0} is not supported for DecisionTreeRegressor".format(method))
    return decision_tree(model.tree_, prepare_inputs(inputs, model.n_features_in_), method)

@register(RandomForestClassifier, ['predict', 'predict_proba', 'predict_log_proba'])
def _(model, inputs, method):
    return random_forest_classifier(model, prepare_inputs(inputs, model.n_features_in_), method)

@register(RandomForestRegressor, ['predict'])
def _(model, inputs, method):
    return random_forest_regressor(model, prepare_inputs(inputs, model.n_features_in_))

@register(GradientBoostingClassifier, ['decision_function'])
def _(model, inputs, method):
    return gradient_boosting_classifier(model, prepare_inputs(inputs, model.n_features_in_))

@register(GradientBoostingRegressor, ['predict'])
def _(model, inputs, method):
    return gradient_boosting_regressor(model, prepare_inputs(inputs, model.n_features_in_))

@register(AdaBoostClassifier, ['decision_function', 'predict', 'predict_proba', 'predict_log_proba'])
def _(model, inputs, method):
    return adaboost_classifier(model, prepare_inputs(inputs, model.estimators_[0].n_features_in_), method)

@register(KMeans, ['transform', 'predict'])
def _(model, inputs, method):
    return k_means(model.cluster_centers_, prepare_inputs(inputs, model.cluster_centers_.shape[1]), method)

@register(_BasePCA, ['transform'])
def _(model, inputs, method):
    return pca(model, prepare_inputs(inputs, model.components_.shape[1]))

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

@register(MLPRegressor, ['predict'])
def _(model, inputs, method):
    return mlp(model, prepare_inputs(inputs, len(model.coefs_[0])))

@register(MLPClassifier, ['predict', 'predict_proba', 'predict_log_proba'])
def _(model, inputs, method):
    return mlp_classifier(model, prepare_inputs(inputs, len(model.coefs_[0])), method)

@register(Binarizer, ['transform'])
def _(model, inputs, method):
    return binarize(model.threshold, prepare_inputs(inputs))

@register(MinMaxScaler, ['transform'])
def _(model, inputs, method):
    return scale(model.scale_, model.min_, prepare_inputs(inputs, len(model.scale_)))

@register(MaxAbsScaler, ['transform'])
def _(model, inputs, method):
    return unscale(model.scale_, prepare_inputs(inputs, len(model.scale_)))

@register(StandardScaler, ['transform'])
def _(model, inputs, method):
    n = None
    if model.with_mean:
        n = len(model.mean_)
    elif model.with_std:
        n = len(model.scale_)
    return standard_scaler(model, prepare_inputs(inputs, n))

@register(Normalizer, ['transform'])
def _(model, inputs, method):
    return normalizer(model.norm, prepare_inputs(inputs))
