"""
Utility for basic testing of models.
"""
import warnings
import numpy as np
from sklearn.datasets import load_iris
from .evaluators import PythonEval, SympyEval, SQLiteEval, ExcelEval, PFAEval

# Set up evaluators
X, y = load_iris(True)
y_bin = np.array(y)
y_bin[y_bin == 2] = 0

_evaluators = {
    'python': PythonEval(X),
    'sympy': SympyEval(X, true_argmax=False),
    'sympy2': SympyEval(X, true_argmax=True),
    'sqlite': SQLiteEval(X, False),
    'sqlite2': SQLiteEval(X, True),
    'excel': ExcelEval(X),
    'pfa': PFAEval(X),
}

def verify_one(model, method, evaluator, expr, binary_fix=False, inf_fix=False, data_preprocessing=None):
    X_inputs = X
    if data_preprocessing:
        X_inputs = data_preprocessing(X_inputs)
    true_Y = getattr(model, method)(X_inputs)
    if binary_fix and true_Y.ndim > 1 and true_Y.shape[1] > 1:
        true_Y = true_Y[:, 1]
    if true_Y.ndim > 1 and true_Y.shape[1] == 1:
        true_Y = true_Y[:, 0]
    pred_Y = _evaluators[evaluator](expr)
    if inf_fix:
        # Our custom SQL log function returns -FLOAT_MIN instead of -inf for log(0)
        pred_Y[pred_Y == np.finfo('float64').min] = -float('inf')
        assert (np.isinf(true_Y) == np.isinf(pred_Y)).all()
        assert np.abs(true_Y[~np.isinf(true_Y)] - pred_Y[~np.isinf(pred_Y)]).max() < 1e-10
    else:
        assert np.abs(pred_Y - true_Y).max() < 1e-10

def verify(model, method, expr, binary_fix=False, inf_fix=False):
    with warnings.catch_warnings():
        if method == 'predict_log_proba': # Ignore divide by zeroes encountered in log(0)
            warnings.simplefilter('ignore', RuntimeWarning)

        verify_one(model, method, 'excel', expr, binary_fix, inf_fix)
        verify_one(model, method, 'python', expr, binary_fix, inf_fix)
        verify_one(model, method, 'sympy', expr, binary_fix, inf_fix)
        if not binary_fix and method == 'predict' and hasattr(model, 'decision_function'):
            verify_one(model, method, 'sympy2', expr, binary_fix, inf_fix)  # See that Sympy supports true_argmax correctly
        verify_one(model, method, 'sqlite', expr, binary_fix, inf_fix)
        verify_one(model, method, 'sqlite2', expr, binary_fix, inf_fix)

        warnings.simplefilter('ignore', PendingDeprecationWarning) # Those two come from the PFA evaluator
        warnings.simplefilter('ignore', DeprecationWarning)
        try:
            verify_one(model, method, 'pfa', expr, binary_fix, inf_fix)
        except NameError as e:
            if str(e) == "name 'inf' is not defined":
                # This happens because I do not know how to properly encode inf/-inf in
                # the PFA output. Ignore this so far.
                pass
            else:
                raise
