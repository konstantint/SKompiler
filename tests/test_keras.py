"""
Test for keras support.
"""
from keras.layers import InputLayer, Dense, Dropout
from keras.models import Sequential
import numpy as np
from skompiler.toskast.keras import translate as from_keras
from .verification import _evaluators

pyeval = _evaluators['python']

def _test(m, X):
    expr = from_keras(m)
    res = pyeval(expr)
    assert np.abs(res - m.predict(X)).max() < 1e-6


_cargs = dict(optimizer='rmsprop', loss='categorical_crossentropy')
def test_keras(X, y_ohe):
    m = Sequential([InputLayer((4,))])
    _test(m, X)

    models = [
        Sequential([Dense(3, activation='linear', input_shape=(4,)),
                    Dense(3, activation='tanh'),
                    Dense(3, activation='relu'),
                    Dense(3, activation='softmax')]),
        Sequential([Dense(3, activation='linear', input_shape=(4,)),
                    Dropout(0.5),
                    Dense(3, activation='sigmoid')])
    ]

    for m in models:
        m.compile(**_cargs)
        m.fit(X, y_ohe)
        _test(m, X)
