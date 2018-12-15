from ....ast import MakeVector, IfThenElse, BinOp, LtEq, NumberConstant, decompose

def binarize(threshold, inputs):
    if not isinstance(inputs, list):
        inputs = decompose(inputs)
    return MakeVector([IfThenElse(BinOp(LtEq(), inp, NumberConstant(threshold)),
                                  NumberConstant(0),
                                  NumberConstant(1)) for inp in inputs])
