import math
import pandas as pd
import numpy as np
import sqlalchemy as sa

from skompiler.fromskast.sqlalchemy import translate as to_sql
import skompiler.fromskast.sympy as to_sympy

def _sql_log(x):
    if x <= 0:
        return np.finfo('float64').min
    else:
        return np.log(x)

class SQLiteEval:
    def __init__(self, X):
        self.engine = sa.create_engine("sqlite://")
        self.conn = self.engine.connect()
        self.conn.connection.create_function('log', 1, _sql_log) #math.log)
        self.conn.connection.create_function('exp', 1, math.exp)
        df = pd.DataFrame(X, columns=['x{0}'.format(i+1) for i in range(X.shape[1])])
        df.to_sql('data', self.conn)
        
    def __call__(self, expr):
        query = 'select {0} from data'.format(to_sql(expr, 'sqlite', 'y'))
        result = pd.read_sql(query, self.conn).values
        if result.shape[1] == 1:
            result = result[:, 0]
        return result
    
    def __del__(self):
        #self.conn.close()   # <-- This raises an exception somewhy
        pass

class PythonEval:
    def __init__(self, X):
        self.X = X

    def __call__(self, expr):
        fn = expr.lambdify()
        return np.asarray([fn(x=x) for x in self.X])

class SympyEval:
    def __init__(self, X, true_argmax=False):
        self.X = X
        self.true_argmax = true_argmax
    
    def __call__(self, expr):
        fn = to_sympy.lambdify('x', to_sympy.translate(expr, true_argmax=self.true_argmax))
        pred_Y = np.asarray([np.array([fn(x.reshape(-1, 1))]).ravel() for x in self.X])
        if pred_Y.shape[-1] == 1:
            pred_Y = pred_Y[..., 0]
        return pred_Y
