"""Layout support shared by this package's target implementations."""


def materialized(op, context):
    return all(context.tensors[name].alias_of is None and context.tensors[name].strides is None
               for name in (*op.inputs, *op.outputs))
