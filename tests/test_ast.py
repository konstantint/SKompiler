#pylint: disable=wildcard-import,unused-wildcard-import,no-member
import numpy as np
from skompiler.ast import *
from skompiler.dsl import *


def test_dsl():
    assert isinstance(ident('x'), Identifier)
    assert isinstance(ident('x', 1), VectorIdentifier)
    assert isinstance(const(1), NumberConstant)
    assert isinstance(const([1]), VectorConstant)
    assert isinstance(const([[1]]), MatrixConstant)
    assert isinstance(const(np.array([1], dtype='int')[0]), NumberConstant)
    assert isinstance(const(np.array(1)), NumberConstant)
    mtx = const(np.array([[1, 2]]))
    assert isinstance(mtx, MatrixConstant)
    assert len(mtx) == 1
    v = mtx[0]
    assert isinstance(v, VectorConstant)
    assert len(v) == 2
    n = v[1]
    assert isinstance(n, NumberConstant)
    assert n.value == 2
    ids = vector(map(ident, 'abc'))
    assert isinstance(ids, MakeVector)
    assert len(ids.elems) == 3
    assert isinstance(ids.elems[0], Identifier)

def test_singleton():
    assert Add() is Add()
    assert func.Add is Add()
    assert func.Mul is Mul()
