"""
Smoke test for all supported models and translation types.
"""
import warnings
from sklearn.tree import DecisionTreeRegressor
from skompiler.ast import BinOp, Mul, NumberConstant, IndexedIdentifier
from skompiler.dsl import ident
from skompiler.toskast.sklearn import _supported_methods
from skompiler import skompile
from .verification import verify_one

# Sympy targets not included because these take a long time to evaluate (and fail on some models, e.g. 'sympy/c', 'sympy/js')
_translate_targets = ['string', 'python/code', 'pfa/json']
_eval_targets = ['python', 'excel', 'sqlite2', 'sympy', 'pfa']

def list_supported_methods(model):
    if isinstance(model, DecisionTreeRegressor):
        return ['predict']
    for cls, methods in _supported_methods.items():
        if isinstance(model, cls):
            return methods
    raise ValueError("Unsupported model: {0}".format(model))

#pylint: disable=unsupported-membership-test
_limit = None

def test_skompile(models):
    # TODO: If we use NumberConstant(2), we get a failed test for RandomForestRegressor.
    # Could it be due to float precision issues?
    transformed_features = [BinOp(Mul(), IndexedIdentifier('x', i, 4), NumberConstant(2.1)) for i in range(4)]

    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)  # Ignore divide by zero warning for log(0)
        warnings.simplefilter('ignore', PendingDeprecationWarning) # Those two come from the PFA evaluator
        warnings.simplefilter('ignore', DeprecationWarning)
        for name, model in models.items():
            if _limit and name not in _limit:
                continue
            methods = list_supported_methods(model)
            for method in methods:
                if name in ['bin', 'n1', 'n2', 'n3']: # Binarizer and Normalizer want to know number of features
                    expr = skompile(getattr(model, method), inputs=ident('x', 4))
                else:
                    expr = skompile(getattr(model, method))

                print(name, model, method)
                for evaluator in _eval_targets:
                    print(evaluator)
                    try:
                        verify_one(model, method, evaluator, expr,
                                   binary_fix=name.endswith('_bin'), inf_fix=(method == 'predict_log_proba'))
                    except NameError as e:
                        if evaluator == 'pfa' and str(e) == "name 'inf' is not defined":
                            # This happens because I do not know how to properly encode inf/-inf in
                            # the PFA output. Ignore this so far.
                            pass
                        else:
                            raise
                for target in _translate_targets:
                    expr.to(target)

                # Check that everything will work if we provide expressions instead of raw features
                expr = skompile(getattr(model, method), transformed_features)
                verify_one(model, method, 'python', expr,
                           binary_fix=name.endswith('_bin'), inf_fix=(method == 'predict_log_proba'),
                           data_preprocessing=lambda X: X*2.1)
