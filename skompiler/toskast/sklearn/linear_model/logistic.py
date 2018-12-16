"""
SKLearn logistic regression to SKAST.
"""
from skompiler.dsl import func
from .base import linear_model
from ..common import classifier, vecsumnormalize, sklearn_softmax


def logreg_binary(coef, intercept, inputs, method="predict_proba"):
    """
    Binary logistic regression.

    Args:

       inputs:  a list of AST nodes to be used as inputs to the model.

    Kwargs:

        method (string): The sklearn method's output to emulate.
            'decision_function' - The logistic regression decision function only.
            'predict_proba' - The output will be the probability of class 1
                    (note that it is NOT two probabilities, as in case of SKLearn's
                    actual predict_proba output)
            'predict_log_proba' - The log probability of class 1
            'predict'  - The output will be an integer 0/1, predicting the class.
    """
    decision = linear_model(coef, intercept, inputs)

    if method == "decision_function":
        return decision
    if method == "predict":
        return func.Step(decision)

    return classifier(func.Sigmoid(decision), method)

    
def logreg_multiclass(coef_matrix, intercept_vector, inputs='x', method="predict_proba", multi_class='ovr'):
    """
    Multiclass logistic regression.

    Kwargs:

      output (string): The sklearn method's output to emulate.
          'decision_function' - The logistic regression decision function only.
          'predict_proba' - The output will be the probability of class 1
                (note that it is NOT two probabilities, as in case of SKLearn's
                 actual predict_proba output)
          'predict_log_proba' - The log probability of class 1
          'predict'  - The output will be an integer 0/1, predicting the class.
      
      inputs (string or ASTNode): The name of the inputs variable to use in the formula.
                    Any SKAST expression could be used instead, assuming it encodes an
                    input vector.
    
      multi_class (string):  The value of the "multi_class" setting used in the model. Either 'ovr' or 'multinomial'
    """
    decision = linear_model(coef_matrix, intercept_vector, inputs)

    if method == "decision_function":
        return decision
    if method == "predict":
        return func.ArgMax(decision)

    if multi_class == 'ovr':
        probs = vecsumnormalize(func.Sigmoid(decision), coef_matrix.shape[0])
    elif multi_class == 'multinomial':
        probs = sklearn_softmax(decision, coef_matrix.shape[0])
    else:
        raise ValueError("Invalid value of the multi_class argument: " + multi_class)

    return classifier(probs, method)
    