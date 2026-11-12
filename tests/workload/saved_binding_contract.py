"""Prepare digest-valid mutations; product semantic refusal is a separate check.

Callers supply the exact schema and fields for each changed bound envelope.
This helper never prepares, evaluates or reconstructs a workload.
"""
from copy import deepcopy
from dataclasses import dataclass


@dataclass(frozen=True)
class BindingLayer:
    path: tuple
    schema: str
    fields: frozenset


def _at(document, path):
    value = document
    for key in path:
        value = value[key]
    if type(value) is not dict:
        raise ValueError('binding layer must select an object')
    return value


def check_layers(document, layers):
    """Run the real digest checker for every declared layer, without rebuilding."""
    from arch_explorer.workload import compilation as compiler
    receipts = []
    for layer in layers:
        value = _at(document, layer.path)
        compiler.check(value, set(layer.fields), layer.schema)
        receipts.append(dict(path=list(layer.path), schema=layer.schema,
                             binding=deepcopy(value['binding']), binding_valid=True,
                             scope='digest-only; product semantic check pending'))
    return receipts


def reseal_layers(document, layers):
    """Copy, strip bindings and seal inside out, then check every sealed layer.

The root and all bound ancestors of a selected layer must be declared. An
unchanged bound sibling can retain its original digest. The input stays intact.
"""
    from arch_explorer.workload import compilation as compiler
    layers = tuple(layers)
    paths = {layer.path for layer in layers}
    if not layers or () not in paths or len(paths) != len(layers):
        raise ValueError('binding layers require a unique root and unique paths')
    for layer in layers:
        if 'binding' not in layer.fields or 'schema' not in layer.fields:
            raise ValueError('binding layer fields must include schema and binding')
        for depth in range(len(layer.path)):
            prefix = layer.path[:depth]
            ancestor = _at(document, prefix)
            if 'binding' in ancestor and prefix not in paths:
                raise ValueError('binding layers omit a bound ancestor')
    result = deepcopy(document)
    for layer in sorted(layers, key=lambda item: len(item.path), reverse=True):
        value = _at(result, layer.path)
        value.pop('binding', None)
        sealed = compiler.seal(value)
        if not layer.path:
            result = sealed
        else:
            parent = result
            for key in layer.path[:-1]:
                parent = parent[key]
            parent[layer.path[-1]] = sealed
    return result, check_layers(result, layers)
