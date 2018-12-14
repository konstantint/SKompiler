"""
Decision trees to SKAST
"""
from ..common import classifier, sum_
from ..tree.base import decision_tree


def random_forest_classifier(model, inputs, method="predict_proba"):
    """
    Creates a SKAST expression corresponding to a given random forest classifier
    """
    trees = [decision_tree(estimator.tree_, inputs, method="predict_proba", value_transform=lambda v: v/len(model.estimators_))
             for estimator in model.estimators_]
    return classifier(sum_(trees), method)


def random_forest_regressor(model, inputs):
    """
    Creates a SKAST expression corresponding to a given random forest regressor
    """

    return sum_([decision_tree(estimator.tree_, inputs=inputs, method="predict", value_transform=lambda v: v/len(model.estimators_))
                 for estimator in model.estimators_])
