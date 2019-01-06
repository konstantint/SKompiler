import math
import warnings
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
    def __init__(self, X, multistage):
        self.engine = sa.create_engine("sqlite://")
        self.conn = self.engine.connect()
        self.conn.connection.create_function('log', 1, _sql_log)
        self.conn.connection.create_function('exp', 1, math.exp)
        self.conn.connection.create_function('sqrt', 1, math.sqrt)
        df = pd.DataFrame(X, columns=['x{0}'.format(i+1) for i in range(X.shape[1])]).reset_index()
        df.to_sql('data', self.conn)
        self.multistage = multistage
        
    def __call__(self, expr):
        query = to_sql(expr, 'sqlite', multistage=self.multistage, key_column='index')
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning) # divide by zero encountered in log
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
        result = np.asarray([fn(x=x) for x in self.X])
        if result.ndim > 1 and result.shape[-1] == 1:
            result = result[..., 0]
        return result

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

class ExcelEval:
    def __init__(self, X):
        self.X = X

    def _eval(self, code, n_outputs, x):
        inputs = {'x{0}'.format(i+1): x_i for i, x_i in enumerate(x)}
        result = code.evaluate(**inputs)
        keys = list(result.keys())[-n_outputs:]
        return np.asarray([result[k] for k in keys])
    
    def __call__(self, expr):
        code = expr.to('excel', multistage=True, _max_subexpression_length=500)
        # We don't know how many outputs should the expression produce just from the
        # excel's result, so we use a hackish way to determine it via a separate evaluator
        res = expr.lambdify()(x=self.X[0])
        shape = getattr(res, 'shape', None)
        n_outputs = 1 if not shape else shape[0]
        result = np.asarray([self._eval(code, n_outputs, x) for x in self.X])
        if result.shape[-1] == 1:
            result = result[..., 0]
        return result

class PFAEval:
    def __init__(self, X):
        self.X = X

    def __call__(self, expr):
        from titus.prettypfa import PFAEngine
        engine, = PFAEngine.fromJson(expr.to('pfa/json'))
        result = np.asarray([engine.action({'x': x}) for x in self.X])
        if result.ndim > 1 and result.shape[-1] == 1:
            result = result[..., 0]
        return result
