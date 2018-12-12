"""
Smoke test for all supported models and translation types.
"""
#pylint: disable=possibly-unused-variable
import warnings
import numpy as np
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.svm import SVR, SVC
from sklearn.datasets import load_iris
from skompiler.toskast.sklearn import _supported_methods
from skompiler import skompile
from .verification import verify_one

X, y = load_iris(True)
y_bin = np.array(y)
y_bin[y_bin == 2] = 0

def list_supported_methods(model):
    if isinstance(model, DecisionTreeRegressor):
        return ['predict']
    for cls, methods in _supported_methods.items():
        if isinstance(model, cls):
            return methods
        
def make_models():
    ols = LinearRegression().fit(X, y)
    lr_bin = LogisticRegression().fit(X, y_bin)
    lr_ovr = LogisticRegression(multi_class='ovr').fit(X, y)
    lr_mn = LogisticRegression(solver='lbfgs', multi_class='multinomial').fit(X, y)
    svc = SVC(kernel='linear').fit(X, y_bin)
    svr = SVR(kernel='linear').fit(X, y)
    dtc = DecisionTreeClassifier(max_depth=4).fit(X, y)
    dtr = DecisionTreeRegressor(max_depth=4).fit(X, y)
    rfc = RandomForestClassifier(n_estimators=3, max_depth=3).fit(X, y)
    rfr = RandomForestRegressor(n_estimators=3, max_depth=3).fit(X, y)
    gbc = GradientBoostingClassifier(n_estimators=3, max_depth=3).fit(X, y)
    gbr = GradientBoostingRegressor(n_estimators=3, max_depth=3).fit(X, y)
    return locals()

_models = make_models()
# Sympy targets not included because these take a long time to evaluate
_targets = ['sqlalchemy', 'python', 'excel', 'string', 'sqlalchemy/sqlite', 'python/code']

def test_skompile():
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)  # Ignore divide by zero warning for log(0)
        for name, model in _models.items():
            methods = list_supported_methods(model)
            for method in methods:
                expr = skompile(getattr(model, method))
                verify_one(model, method, 'python', expr,
                           binary_fix=(name == 'lr_bin'), inf_fix=(method == 'predict_log_proba'))
                for target in _targets:
                    expr.to(target)

def test_sympy_skompile():
    """We run sympy in a separate test, because it takes longer than other generations"""
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)  # Ignore divide by zero warning for log(0)
        for name, model in _models.items():
            methods = list_supported_methods(model)
            for method in methods:
                expr = skompile(getattr(model, method))
                verify_one(model, method, 'sympy', expr,
                           binary_fix=(name == 'lr_bin'), inf_fix=(method == 'predict_log_proba'))
                # This thing is way too slow and falls on some models
                # TODO: Figure out why
                #for target in ['sympy/c', 'sympy/js']:
                #    expr.to(target)
