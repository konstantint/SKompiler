import numpy as np

from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPClassifier

from skompiler import skompile


def test_random_pipeline():
    m = Pipeline([('scale', StandardScaler()),
                  ('dim_reduce', PCA(6)),
                  ('cluster', KMeans(10)),
                  ('classify', MLPClassifier([5, 4], 'tanh'))])

    X, y = load_breast_cancer(return_X_y=True)
    m.fit(X, y)
    
    expr = skompile(m, 'predict_proba')

    pred_Y = np.asarray([expr.evaluate(x=X[i]) for i in range(len(X))]).ravel()
    true_Y = m.predict_proba(X)[:, 1]

    assert np.abs(true_Y - pred_Y.ravel()).max() < 1e-10
