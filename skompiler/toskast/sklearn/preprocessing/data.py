from skompiler.ast import decompose
from skompiler.dsl import const, vector, iif, let, defn, func, ref

def binarize(threshold, inputs):
    if not isinstance(inputs, list):
        inputs = decompose(inputs)
    return vector([iif(inp <= const(threshold), const(0), const(1)) for inp in inputs])

def scale(scale_, min_, inputs):
    return inputs * const(scale_) + const(min_)

def unscale(scale_, inputs):
    return inputs / const(scale_)

def standard_scaler(model, inputs):
    if model.with_mean:
        inputs = inputs - const(model.mean_)
    if model.with_std:
        inputs = inputs / const(model.scale_)
    return inputs

def normalizer(norm, inputs):
    if norm == 'l2':
        norm = func.Sqrt(func.VecSum(inputs * inputs))
    elif norm == 'l1':
        norm = func.VecSum(func.Abs(inputs))
    elif norm == 'max':
        norm = func.VecMax(inputs)
    else:
        raise ValueError("Unknown norm {0}".format(norm))
    norm_fix = iif(ref('norm', norm) == const(0), const(1), ref('norm', norm))
    return let(defn(norm=norm),
               defn(norm_fix=norm_fix),
               inputs / vector([ref('norm_fix', norm_fix)]*len(inputs)))
