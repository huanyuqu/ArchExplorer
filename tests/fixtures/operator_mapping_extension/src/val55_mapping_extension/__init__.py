"""Unknown semantic square expands to existing materialized mul; no new backend."""
from arch_explorer.workload.api import (
    OperatorDefinition,OperatorMapping,MappingExpansion,
    register_operator,register_operator_mapping)
from arch_explorer.workload.operators import Operator
from arch_explorer.workload.extensions import get_operator

from .access import square_access

KIND = 'val55.square'
RULE_ID = 'val55.square.materialized'
ORPHAN_KIND = 'val55.orphan.semantic'
ORPHAN_RULE = 'val55.orphan.mapping'


def multiplication(operator):
    if len(operator.inputs)!=1 or len(operator.outputs)!=1:
        raise ValueError('square requires exactly one input and output')
    return Operator(operator.name+'.multiply','mul',
                    (operator.inputs[0],operator.inputs[0]),tuple(operator.outputs),
                    attrs=dict(operator.attrs))


def validate(operator,tensors):
    return get_operator('mul').validate(multiplication(operator),tensors)


def reference(operator,values):
    return get_operator('mul').reference(multiplication(operator),[values[0],values[0]])


def expand(operator,tensors,options):
    return MappingExpansion(operators=(multiplication(operator),))


def access(operator,tensors,options):
    return square_access(operator,tensors)


def inspection(operator,tensors,options):
    return dict(fixture='val55-square',output_shape=list(tensors[operator.outputs[0]].shape))


DEFINITION = OperatorDefinition(KIND,validate,version='1',reference=reference)
MAPPING = OperatorMapping(RULE_ID,'1',KIND,expand,access,inspect=inspection)
ORPHAN_DEFINITION = OperatorDefinition(ORPHAN_KIND,validate,version='1',reference=reference)
ORPHAN_MAPPING = OperatorMapping(ORPHAN_RULE,'1',ORPHAN_KIND,expand,access)


def register():
    register_operator(DEFINITION)
    register_operator_mapping(MAPPING)


def register_then_fail():
    register_operator(ORPHAN_DEFINITION)
    register_operator_mapping(ORPHAN_MAPPING)
    raise RuntimeError('VAL55 intentional semantic and mapping registrar failure')
