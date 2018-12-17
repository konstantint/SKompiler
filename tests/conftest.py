"""
Test fixtures.
"""
#pylint: disable=possibly-unused-variable,redefined-outer-name
import os
import numpy as np
import pandas as pd
from pytest import fixture
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.ensemble import AdaBoostClassifier
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.svm import SVR, SVC
from sklearn.preprocessing.data import Binarizer, MinMaxScaler, MaxAbsScaler, StandardScaler, Normalizer
os.environ['KERAS_BACKEND'] = 'theano'

@fixture('session')
def iris():
    return load_iris(return_X_y=True)

@fixture('session')
def X(iris):
    return iris[0]

@fixture('session')
def y(iris):
    return iris[1]

@fixture('session')
def y_bin(y):
    y_bin = np.array(y)
    y_bin[y_bin == 2] = 0
    return y_bin

@fixture('session')
def y_ohe(y):
    return pd.get_dummies(y)
        
def make_models(X, y, y_bin):
    return dict(
        ols=LinearRegression().fit(X, y),
        lr_bin=LogisticRegression().fit(X, y_bin),
        lr_ovr=LogisticRegression(multi_class='ovr').fit(X, y),
        lr_mn=LogisticRegression(solver='lbfgs', multi_class='multinomial').fit(X, y),
        svc=SVC(kernel='linear').fit(X, y_bin),
        svr=SVR(kernel='linear').fit(X, y),
        dtc=DecisionTreeClassifier(max_depth=4).fit(X, y),
        dtr=DecisionTreeRegressor(max_depth=4).fit(X, y),
        rfc=RandomForestClassifier(n_estimators=3, max_depth=3, random_state=1).fit(X, y),
        rfr=RandomForestRegressor(n_estimators=3, max_depth=3, random_state=1).fit(X, y),
        gbc=GradientBoostingClassifier(n_estimators=3, max_depth=3, random_state=1).fit(X, y),
        gbr=GradientBoostingRegressor(n_estimators=3, max_depth=3, random_state=1).fit(X, y),
        abc=AdaBoostClassifier(algorithm='SAMME', n_estimators=3, random_state=1).fit(X, y),
        abc2=AdaBoostClassifier(algorithm='SAMME.R', n_estimators=3, random_state=1).fit(X, y),
        abc3=AdaBoostClassifier(algorithm='SAMME', n_estimators=3, random_state=1).fit(X, y_bin),
        abc4=AdaBoostClassifier(algorithm='SAMME.R', n_estimators=3, random_state=1).fit(X, y_bin),
        km=KMeans(1).fit(X),
        km2=KMeans(5).fit(X),
        pc1=PCA(1).fit(X),
        pc2=PCA(2).fit(X),
        pc3=PCA(2, whiten=True).fit(X),
        mlr1=MLPRegressor([2], 'relu').fit(X, y),
        mlr2=MLPRegressor([2, 1], 'tanh').fit(X, y),
        mlr3=MLPRegressor([2, 2, 2], 'identity').fit(X, y),
        mlc=MLPClassifier([2, 2], 'tanh').fit(X, y),
        mlc_bin=MLPClassifier([2, 2], 'identity').fit(X, y_bin),
        bin=Binarizer(0.5),
        mms=MinMaxScaler().fit(X),
        mas=MaxAbsScaler().fit(X),
        ss1=StandardScaler().fit(X),
        ss2=StandardScaler(with_mean=False).fit(X),
        ss3=StandardScaler(with_std=False).fit(X),
        n1=Normalizer('l1'),
        n2=Normalizer('l2'),
        n3=Normalizer('max')
    )


@fixture('session')
def models(X, y, y_bin):
    return make_models(X, y, y_bin)
