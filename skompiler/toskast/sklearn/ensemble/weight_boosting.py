"""
AdaBoost
"""
import numpy as np
from skompiler.dsl import const, func, sum_
from ..common import classifier, vecsumnormalize
from ..tree.base import decision_tree

# NB: AdaboostRegressor is annoying to implement as it requires
# finding a weighted median among the all estimator predictions,
# which is not meaningfully implementable without adding extra special functions
# Thus we only support AdaboostClassifier for now
def adaboost_classifier(model, inputs, method="predict_proba"):
    """
    Creates a SKAST expression corresponding to a given adaboost classifier.
    """
    divisor = model.estimator_weights_.sum()
    if method == 'decision_function':
        divisor /= (model.n_classes_ - 1)
    tree_exprs = [decision_tree(e.tree_,
                                method='predict_proba' if (model.algorithm == 'SAMME.R' or
                                                           method.startswith('predict_')) else 'predict',
                                inputs=inputs,
                                value_transform=adaboost_values(model, w/divisor, method))
                  for e, w in zip(model.estimators_, model.estimator_weights_)]
    decision = sum_(tree_exprs)

    if method == 'decision_function':
        if model.n_classes_ == 2:
            decision = decision @ const([-1, 1])
        return decision
    elif method == 'predict':
        return func.ArgMax(decision)
    else:
        return classifier(vecsumnormalize(func.Exp(sum_(tree_exprs)), model.n_classes_), method)


def adaboost_values(m, weight=1.0, method='predict_proba'):
    def _samme(proba):
        proba = np.array(proba)
        proba[proba < np.finfo(proba.dtype).eps] = np.finfo(proba.dtype).eps
        log_proba = np.log(proba)
        return (log_proba - (1. / m.n_classes_) * log_proba.sum(axis=1)[:, np.newaxis])*weight
    
    def _ada_predict_proba(probs):
        return probs * weight / (m.n_classes_ - 1)
    
    def _ada_predict(preds):
        probs = np.zeros((len(preds), m.n_classes_))
        probs[np.arange(len(preds)), preds] = 1
        probs *= weight/(m.n_classes_-1)
        return probs

    if m.algorithm == 'SAMME.R':
        return _samme
    elif method == 'predict' or method == 'decision_function':
        return _ada_predict
    else:
        return _ada_predict_proba
