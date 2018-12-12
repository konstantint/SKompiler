
from pytest import fixture
from sklearn.datasets import load_iris

@fixture
def iris():
    return load_iris(return_X_y=True)
