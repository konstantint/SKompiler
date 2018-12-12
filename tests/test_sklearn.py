"""
Test generic sklearn translate function on supported classifier models.
"""

import warnings
import numpy as np
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV, \
                                 LinearRegression, Ridge, Lars, LarsCV, ElasticNet, ElasticNetCV, \
                                 Lasso, LassoCV, LassoLars, LassoLarsCV, LassoLarsIC
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, \
                             GradientBoostingClassifier, GradientBoostingRegressor
from skompiler.toskast.sklearn import translate
import skompiler.fromskast.sympy as to_sympy
from .verification import X, y, y_bin, verify as _verify
from .evaluators import SQLiteEval


def verify(model, methods=None, binary_fix=False, inf_fix=False):
    methods = methods or ['decision_function', 'predict_proba', 'predict_log_proba', 'predict']
    for method in methods:
        expr = translate(model, method=method)
        _verify(model, method, expr, binary_fix=binary_fix, inf_fix=inf_fix)

def test_logreg():
    m = LogisticRegression(solver='lbfgs', multi_class='ovr')
    verify(m.fit(X, y_bin), binary_fix=True)
    verify(m.fit(X, y))
    verify(m.set_params(multi_class='multinomial').fit(X, y))

def test_logregcv():
    m = LogisticRegressionCV(solver='lbfgs', multi_class='ovr')
    verify(m.fit(X, y_bin), binary_fix=True)
    verify(m.fit(X, y))
    verify(m.set_params(multi_class='multinomial').fit(X, y))

def test_linearsvc():
    m = SVC(kernel='linear')
    verify(m.fit(X, y_bin), ['decision_function', 'predict'], True)
    # Non-binary SVM not implemented so far
    # verify(m.fit(X, y), methods=['decision_function', 'predict'])

def test_linreg():
    for m in [LinearRegression(), Ridge(), Lars(), LarsCV(), ElasticNet(), ElasticNetCV(), \
              Lasso(), LassoCV(), LassoLars(), LassoLarsCV(), LassoLarsIC(), SVR(kernel='linear')]:
        verify(m.fit(X, y), ['predict'])
        if not isinstance(m, SVR):
            verify(m.set_params(fit_intercept=False).fit(X, y), ['predict'])

def test_tree():
    for m_class in [DecisionTreeClassifier, DecisionTreeRegressor]:
        m = m_class(max_depth=3).fit(X, y)
        verify(m, ['predict'])
        if m_class == DecisionTreeClassifier:
            verify(m, ['predict_proba'])
            with warnings.catch_warnings(): # Ignore divide by zeroes encountered in log(0)
                warnings.simplefilter('ignore', RuntimeWarning)
                verify(m, ['predict_log_proba'], inf_fix=True)

def test_rf():
    for m_class in [RandomForestClassifier, RandomForestRegressor]:
        m = m_class(random_state=1, max_depth=3, n_estimators=3).fit(X, y)
        verify(m, ['predict'])
        if m_class == RandomForestClassifier:
            verify(m, ['predict_proba'])
            with warnings.catch_warnings(): # Ignore divide by zeroes encountered in log(0)
                warnings.simplefilter('ignore', RuntimeWarning)
                verify(m, ['predict_log_proba'], inf_fix=True)

def test_gb():
    for m_class in [GradientBoostingClassifier, GradientBoostingRegressor]:
        m = m_class(random_state=1, max_depth=3, n_estimators=3).fit(X, y)
        if m_class == GradientBoostingRegressor:
            verify(m, ['predict'])
        else:
            verify(m, ['decision_function'])

def test_columnlist():
    m = LinearRegression()
    m.fit(X, y)
    true_Y = m.predict(X)
    inputs = ['x{0}'.format(i+1) for i in range(X.shape[1])]
    expr = translate(m, inputs=inputs)
    ev = SQLiteEval(X)
    assert np.abs(ev(expr) - true_Y).max() < 1e-10

    fn = to_sympy.lambdify(' '.join(inputs), to_sympy.translate(expr))
    pred_Y = np.asarray([fn(*x) for x in X])
    assert np.abs(pred_Y - true_Y).max() < 1e-10
