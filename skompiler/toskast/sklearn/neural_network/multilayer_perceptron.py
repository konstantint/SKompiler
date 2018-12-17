"""
Multilayer perceptron
"""
from skompiler.dsl import func, const
from ..common import classifier

_activations = {
    'identity': lambda x: x,
    'tanh': lambda x: func.Sigmoid(x*2)*2 - 1,
    'logistic': func.Sigmoid,
    'relu': lambda x: func.Max(x, const([0] * len(x))),
    'softmax': func.Softmax
}

def mlp(model, inputs):
    actns = [model.activation]*(len(model.coefs_)-1) + [model.out_activation_]
    outs = inputs
    for M, b, a in zip(model.coefs_, model.intercepts_, actns):
        outs = _activations[a](const(M.T) @ outs + const(b))
    return outs

def mlp_classifier(model, inputs, method):
    out = mlp(model, inputs)
    if model.n_outputs_ == 1 and method == 'predict':
        # Binary classifier
        return func.Step(out - 0.5)
    else:
        return classifier(out, method)
