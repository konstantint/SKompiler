"""
Decision trees to SKAST
"""
from functools import reduce
from ..tree.base import decision_tree
from ....ast import BinOp, ElemwiseBinOp, MakeVector, NumberConstant, Add, Mul

def gradient_boosting_classifier(model, inputs, method="decision_function"):
    """
    Creates a SKAST expression corresponding to a given gradient boosting classifier

    At the moment we only support model's decision_function method.
    FYI: Conversion to probabilities and a prediction depends on the loss and by default
          is done as np.exp(score - (logsumexp(score, axis=1)[:, np.newaxis])))
    """

    if method != "decision_function":
        raise NotImplementedError("Only decision_function is implemented for gradient boosting models so far")
    
    tree_exprs = [MakeVector([decision_tree(estimator.tree_, inputs, method="predict") for estimator in iteration])
                  for iteration in model.estimators_]
    tree_sum = reduce(lambda x, y: ElemwiseBinOp(Add(), x, y), tree_exprs)
    tree_sum = ElemwiseBinOp(Mul(), tree_sum, MakeVector([NumberConstant(model.learning_rate) for _ in range(model.n_classes_)]))
    return ElemwiseBinOp(Add(), tree_sum, MakeVector([NumberConstant(prior) for prior in model.init_.priors]))


def gradient_boosting_regressor(model, inputs, method="decision_function"):
    """
    Creates a SKAST expression corresponding to a given GB regressor.
    
    The logic is mostly the same as for the classifier, except we work with scalars rather than vectors.
    """

    if method != "decision_function":
        raise NotImplementedError("Only decision_function is implemented for gradient boosting models so far")
    
    tree_exprs = [decision_tree(iteration[0].tree_, inputs, method="predict") for iteration in model.estimators_]
    tree_sum = reduce(lambda x, y: BinOp(Add(), x, y), tree_exprs)
    tree_sum = BinOp(Mul(), tree_sum, NumberConstant(model.learning_rate))
    return BinOp(Add(), tree_sum, NumberConstant(model.init_.mean))
