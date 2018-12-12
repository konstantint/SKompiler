"""
Decision trees to SKAST
"""
from functools import reduce
from ..tree.base import decision_tree
from ....ast import BinOp, ElemwiseBinOp, Div, MakeVector, NumberConstant,\
                    UnaryFunc, ArgMax, ElemwiseUnaryFunc, Log, Add

def random_forest_classifier(model, inputs, method="predict_proba"):
    """
    Creates a SKAST expression corresponding to a given random forest classifier
    """

    tree_exprs = [decision_tree(estimator.tree_, inputs, method="predict_proba") for estimator in model.estimators_]
    tree_sum = reduce(lambda x, y: ElemwiseBinOp(Add(), x, y), tree_exprs)
    probs = ElemwiseBinOp(Div(), tree_sum, MakeVector([NumberConstant(model.n_estimators) for _ in range(model.n_classes_)]))

    if method == 'predict_proba':
        return probs
    elif method == 'predict':
        return UnaryFunc(ArgMax(), probs)
    elif method == 'predict_log_proba':
        return ElemwiseUnaryFunc(Log(), probs)
    else:
        raise ValueError("Invalid method: {0}".format(method))

def random_forest_regressor(model, inputs):
    """
    Creates a SKAST expression corresponding to a given random forest classifier
    """

    tree_exprs = [decision_tree(estimator.tree_, inputs=inputs, method="predict") for estimator in model.estimators_]
    tree_sum = reduce(lambda x, y: BinOp(Add(), x, y), tree_exprs)
    return BinOp(Div(), tree_sum, NumberConstant(model.n_estimators))
