"""An external rule expands square into public multiply, retaining two reads.

The rule owns its expression and materialization. Existing multiply adaptation,
component timing and common dependency/storage scheduling are reused unchanged.
"""
import math
from arch_explorer.workload.api import MappingExpansion, Operator, OperatorMapping
from .semantics import KIND, validate


def expand(op, tensors, options):
    validate(op, tensors)
    if options:
        raise ValueError("example.square.materialized has no options")
    product = Operator(op.name + "/materialized_square", "mul", (op.inputs[0], op.inputs[0]),
                       tuple(op.outputs), {"shape": list(op.attrs["shape"])}, "external square")
    return MappingExpansion(operators=(product,))


def access(op, tensors, options):
    if (options or op.op != "mul" or len(op.inputs) != 2 or op.inputs[0] != op.inputs[1]
            or len(op.outputs) != 1):
        raise ValueError("square mapping access requires its exact two-read multiply product")
    source, output = tensors[op.inputs[0]], tensors[op.outputs[0]]
    if source.dtype != "fp32" or output.dtype != "fp32" or source.shape != output.shape:
        raise ValueError("square mapped operands must retain matching FP32 shape")
    return {"operations": math.prod(output.shape), "read_bytes": 2 * source.bytes,
            "write_bytes": output.bytes, "boundary": "component_io",
            "operand_reads": 2, "data_access": "two explicit reads of X, one materialized X*X output"}


def inspect(op, tensors, options):
    access(op, tensors, options)
    return {"expression": "X*X", "dtype": "fp32", "shape": list(tensors[op.outputs[0]].shape),
            "materialized": True, "implicit_alias": False}


MATERIALIZED = OperatorMapping(id="example.square.materialized", version="1", operator=KIND,
                              expand=expand, access=access, inspect=inspect)
