"""
Translation from Keras.
So far just a proof of concept, working via Theano.
"""
from keras.models import Sequential
from skompiler.ast import Let, Definition, inline_definitions
from .._common import prepare_inputs
from .theano import TheanoTranslator


def translate(model, inputs='x'):
    if not isinstance(model, Sequential):
        raise NotImplementedError("Support for non-sequential models is not implemented")

    if len(model.inputs) != 1:
        raise NotImplementedError("Support for models with multiple inputs is not implemented")
    
    if len(model.outputs) != 1:
        raise NotImplementedError("Support for models with multiple outputs is not implemented")

    if not model.outputs[0].__class__.__module__.startswith('theano.'):
        raise NotImplementedError("Non-theano backend support is not implemented")

    if len(model.input_shape) != 2:
        raise NotImplementedError("Support for models with non-vector inputs is not implemented")
    
    if model.input_shape[0] is not None:
        raise NotImplementedError("Not sure what to do when model.input_shape[0] is not None")
    
    inputs = prepare_inputs(inputs, model.input_shape[1])
    tt = TheanoTranslator(model.inputs[0].name, inputs)
    result = tt(model.outputs[0])
    if tt.definitions:
        result = Let([Definition(name, value) for name, value in tt.definitions.items()],
                     result)
        # For now excel/sql code generators cannot handle reference variables which point to matrix constants.
        # which TheanoTranslator generates.
        # We overcome the problem by dumb inlining
        result = inline_definitions(result)
    return result
