from sklearn.linear_model import LogisticRegression
from skompiler.ast import VectorIdentifier
from skompiler.toskast.sklearn.linear_model.logistic import logreg_binary, logreg_multiclass
from .verification import X, y, y_bin, verify


_inputs = VectorIdentifier('x', 4)  # Iris table has four input columns

def test_logreg_binary():
    m = LogisticRegression(solver='lbfgs')
    m.fit(X, y_bin)

    for method in ['decision_function', 'predict_proba', 'predict_log_proba', 'predict']:
        expr = logreg_binary(m.coef_.ravel(), m.intercept_[0], _inputs, method=method)
        verify(m, method, expr, True)

def test_logreg_multiclass_ovr():
    m = LogisticRegression(solver='lbfgs', multi_class='ovr')
    m.fit(X, y)

    for method in ['decision_function', 'predict_proba', 'predict_log_proba', 'predict']:
        expr = logreg_multiclass(m.coef_, m.intercept_, method=method, inputs=_inputs, multi_class='ovr')
        verify(m, method, expr)

def test_logreg_multiclass_multinomial():
    m = LogisticRegression(solver='lbfgs', multi_class='multinomial')
    m.fit(X, y)

    for method in ['decision_function', 'predict_proba', 'predict_log_proba', 'predict']:
        expr = logreg_multiclass(m.coef_, m.intercept_, method=method, inputs=_inputs, multi_class='multinomial')
        verify(m, method, expr)
