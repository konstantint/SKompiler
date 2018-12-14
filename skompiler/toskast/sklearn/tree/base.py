"""
Decision trees to SKAST
"""
import numpy as np
from ....ast import IfThenElse, NumberConstant, VectorConstant, IndexedIdentifier, BinOp, LtEq, VectorIdentifier


def decision_tree(tree, inputs, method="predict"):
    """
    Creates a SKAST expression corresponding to a given SKLearn Tree object.

    Kwargs:
      
       inputs:  a list of AST nodes to be used as inputs to the model.
       method:  'predict' (for classifier and regressor models),
                'predict_proba' or 'predict_log_proba' (for classifier models)
    
    >>> from sklearn.datasets import load_iris
    >>> from sklearn.tree import DecisionTreeClassifier
    >>> m = DecisionTreeClassifier(max_depth=2, random_state=1).fit(*load_iris(True))
    >>> print(decision_tree(m.tree_, VectorIdentifier('x', m.n_features_)))
    (if (x[3] <= 0.80...) then 0 else (if (x[3] <= 1.75) then 1 else 2))
    >>> print(decision_tree(m.tree_, ['a','b','c','d'], method='predict_proba'))
    (if (d <= 0.80...) then [1. 0. 0.] else (if (d <= 1.75) then [0... 0.90... 0.09...] else [0... 0.02... 0.97...]))
    """
    features = inputs

    v = tree.value[:, 0, :]
    if v.shape[1] == 1:
        # Regression model
        if method != 'predict':
            raise ValueError("Only predict method is supported for regression trees")
        v = v[:, 0]
    else:
        # Classifier
        if method == "predict":
            v = np.argmax(v, axis=1)
        else:
            v = v / v.sum(axis=1)[:, np.newaxis]
            if method == "predict_log_proba":
                v = np.log(v)
            elif method != "predict_proba":
                raise ValueError("Invalid method: {0}".format(method))
    
    return TreeWalker(tree, features, v).walk()


class TreeWalker:
    """Converts a SKLearn Tree object to a SKAST expression.
    
    >>> from sklearn.datasets import load_iris
    >>> from sklearn.tree import DecisionTreeRegressor
    >>> m =  DecisionTreeRegressor(max_depth=2, random_state=1).fit(*load_iris(True))
    >>> tr = TreeWalker(m.tree_, [IndexedIdentifier('x', i, 4) for i in range(4)])
    >>> print(tr.walk())
    (if (x[3] <= 0.80...) then 0.0 else (if (x[3] <= 1.75) then 1.09... else 1.97...))
    >>> tr = TreeWalker(m.tree_, [IndexedIdentifier('x', i, 4) for i in range(4)], np.arange(m.tree_.node_count))
    >>> print(tr.walk())
    (if (x[3] <= 0.80...) then 1 else (if (x[3] <= 1.75) then 3 else 4))
    """
    
    def __init__(self, tree, features, node_values=None):
        """
        Kwargs:
           node_values (list/array): A way to override the tree.value array.
                                     Must be a 1D or 2D array of values.
        """
        if isinstance(features, VectorIdentifier):
            features = [IndexedIdentifier(features.id, i, features.size) for i in range(features.size)]
        
        self.tree = tree
        self.features = features
        if tree.n_features != len(features):
            raise ValueError("Incorrect number of features provided")
        if node_values is None:
            self.values = tree.value[:, 0]
            if self.values.shape[1] == 1:
                self.values = self.values[:, 0]
        else:
            self.values = node_values
        
    def walk(self, node_id=0):
        if node_id >= self.tree.node_count or node_id < 0:
            raise ValueError("Invalid node id")
        if self.tree.children_left[node_id] == -1:
            if self.tree.children_right[node_id] != -1:
                raise ValueError("Invalid tree structure. Children must either be both present or absent")
            
            if self.values.ndim == 1:
                return NumberConstant(self.values[node_id].item())
            else:
                return VectorConstant(self.values[node_id])
        else:
            ft = self.tree.feature[node_id]
            if ft < 0 or ft >= self.tree.n_features:
                raise ValueError("Invalid feature value for node {0}".format(node_id))
            return IfThenElse(BinOp(LtEq(), self.features[ft], NumberConstant(self.tree.threshold[node_id].item())),
                              self.walk(self.tree.children_left[node_id]),
                              self.walk(self.tree.children_right[node_id]))
