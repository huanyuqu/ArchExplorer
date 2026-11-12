"""An unknown installed adapter/implementation; uses an existing backend.

Only semantic-to-Kernel conversion is owned here. Rates and actual time/resource
demand remain in the existing VectorUnitModel; this fixture has no time formula.
"""
from dataclasses import replace

from arch_explorer.components.api import Kernel, register_component_implementation
from arch_explorer.components.sm.implementations.events import DEFINITION as EVENTS
from arch_explorer.workload.api import (
    TaskAdapter, AdaptedGroup, AdapterDomain,
    register_task_adapter, register_adapter_domain,
)


def lower(context):
    by_name = {op.name:op for op in context.program.operators}
    result = {}
    for name in context.tasks:
        op = by_name[name]
        a = op.attrs
        if op.op == 'compiled.matmul':
            kind, shape = 'gemm', [a['batch'], a['rows'], a['k'], a['cols']]
        elif op.op == 'compiled.assemble':
            kind, shape = 'vector', [1, a['batch']*a['m']*a['n']]
        else:
            raise NotImplementedError('VAL54 fixture supports compiled matrix semantics only')
        result[name] = {'work': dict(op=kind, shape=shape, precision='fp32',
                                    simt=0, sfu=0, row_simt=0, row_sfu=0),
                        'traffic': 'external'}
    return AdaptedGroup({'tasks':result}, {'traffic':'external',
                        'basis':'VAL54 test-only conversion; existing lane backend'})


def restore(work):
    return Kernel(**{**work, 'shape':tuple(work['shape'])})


def compose_must_not_run(adf, program, groups):
    raise AssertionError('a cross-domain set must reject before composing either domain')


SEMANTICS = ('compiled.matmul', 'compiled.assemble')
FIRST = TaskAdapter('val54.region', '1', 'events', SEMANTICS, lower, restore=restore)
SECOND = TaskAdapter('val54.alternative', '1', 'events', SEMANTICS, lower, restore=restore)
FOREIGN = TaskAdapter('val54.foreign', '1', 'val54.foreign', SEMANTICS, lower, restore=restore)
FOREIGN_DOMAIN = AdapterDomain('val54.foreign', compose_must_not_run)
COMPONENT = replace(EVENTS, id='val54.sm', adapters=('val54.region',))
AMBIGUOUS = replace(EVENTS, id='val54.ambiguous', adapters=('val54.region', 'val54.alternative'))
ABSENT = replace(EVENTS, id='val54.absent', adapters=())
WRONG_DOMAIN = replace(EVENTS, id='val54.foreign', execution_domain='val54.foreign',
                       adapters=('val54.foreign',))
ORPHAN = replace(EVENTS, id='val54.orphan', adapters=('val54.orphan',))
ORPHAN_ADAPTER = TaskAdapter('val54.orphan', '1', 'events', SEMANTICS, lower, restore=restore)


def register():
    for component in (COMPONENT, AMBIGUOUS, ABSENT, WRONG_DOMAIN):
        register_component_implementation(component)
    register_adapter_domain(FOREIGN_DOMAIN)
    for adapter in (FIRST, SECOND, FOREIGN):
        register_task_adapter(adapter)


def register_then_fail():
    register_component_implementation(ORPHAN)
    register_task_adapter(ORPHAN_ADAPTER)
    raise RuntimeError('VAL54 intentional registrar failure after both registrations')
