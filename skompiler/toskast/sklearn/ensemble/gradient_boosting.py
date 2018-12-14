"""
Decision trees to SKAST
"""
from ..tree.base import decision_tree
from ..common import sum_
from ....ast import MakeVector, NumberConstant


def gradient_boosting_classifier(model, inputs, method="decision_function"):
    """
    Creates a SKAST expression corresponding to a given gradient boosting classifier

    At the moment we only support model's decision_function method.
    FYI: Conversion to probabilities and a prediction depends on the loss and by default
          is done as np.exp(score - (logsumexp(score, axis=1)[:, np.newaxis])))
    """

    if method != "decision_function":
        raise NotImplementedError("Only decision_function is implemented for gradient boosting models so far")

    tree_exprs = [MakeVector([decision_tree(estimator.tree_, inputs, method="predict", value_transform=lambda v: v * model.learning_rate)
                              for estimator in iteration])
                  for iteration in model.estimators_]
    prior = MakeVector([NumberConstant(prior) for prior in model.init_.priors])
    return sum_(tree_exprs + [prior])


def gradient_boosting_regressor(model, inputs, method="decision_function"):
    """
    Creates a SKAST expression corresponding to a given GB regressor.
    
    The logic is mostly the same as for the classifier, except we work with scalars rather than vectors.
    """

    if method != "decision_function":
        raise NotImplementedError("Only decision_function is implemented for gradient boosting models so far")
    
    tree_exprs = [decision_tree(iteration[0].tree_, inputs, method="predict", value_transform=lambda v: v * model.learning_rate)
                  for iteration in model.estimators_]
    prior = NumberConstant(model.init_.mean)
    return sum_(tree_exprs + [prior])
