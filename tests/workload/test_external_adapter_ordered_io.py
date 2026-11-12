"""CORE25 external adapter guards, using the installed real extension.

Negative tests inject an invalid product at the operator-lowering boundary.
They target the adapter's ordered-I/O guard itself; valid conversion still uses
the real registered lowering. No replacement timing or backend is provided.
"""
from dataclasses import replace
import importlib

import pytest

from test_public_compilation_contract import compiler_api, prohibit_execution


def external_context(w):
    from arch_explorer.adf import find
    from arch_explorer.components.api import resolve_component
    from arch_explorer.mapping.compilation import map_workload
    from arch_explorer.workload.api import workload_from_spec
    scene = importlib.import_module('ae_component_implementations.compilation')
    adapter = importlib.import_module('ae_component_implementations.adapter')
    adf, spec, mapping = scene.compilation_scene(m=3, k=3, n=5)
    program, _ = map_workload(workload_from_spec(spec), mapping)
    task = next(op for op in program.operators if op.op == 'compiled.matmul')
    compute, memory = mapping.params['compute'], mapping.params['memory']
    component = resolve_component(find(adf, compute), execution_domain='events')
    context = w.AdapterContext(adf, program, mapping, (task.name,), compute, memory,
                               component, {}, {})
    return adapter, context, task


def test_external_adapter_preserves_real_ordered_inputs_outputs(compiler_api, monkeypatch):
    w, s = compiler_api
    adapter, context, task = external_context(w)
    with prohibit_execution(monkeypatch, s):
        group = adapter.lower(context)
    assert set(group.payload['tasks']) == {task.name}
    item = group.payload['tasks'][task.name]
    assert item['traffic'] == 'external'
    kernel = adapter.restore(item['work'])
    assert kernel.op == 'gemm'
    assert kernel.shape == (task.attrs['batch'], task.attrs['rows'],
                             task.attrs['k'], task.attrs['cols'])


@pytest.mark.parametrize('mutation', ['reversed-inputs', 'wrong-output',
                                    'missing-input', 'hidden-temporary', 'two-steps'])
def test_external_adapter_rejects_undeclared_or_reordered_step_io(
        compiler_api, monkeypatch, mutation):
    w, s = compiler_api
    adapter, context, task = external_context(w)
    original = adapter.lower_implementation
    seen = []
    def injected_product(definition, operator, execution):
        real = original(definition, operator, execution)
        assert len(real.steps) == 1 and not real.tensors
        step = real.steps[0]
        assert step.inputs == tuple(task.inputs) and step.outputs == tuple(task.outputs)
        seen.append(operator.name)
        if mutation == 'reversed-inputs':
            return replace(real, steps=(replace(step, inputs=step.inputs[::-1]),))
        if mutation == 'wrong-output':
            # A fresh name reaches the adapter's output guard; reusing an
            # input would be rejected earlier by ExecutionStep construction.
            return replace(real, steps=(replace(step, outputs=('val54.undeclared.output',)),))
        if mutation == 'missing-input':
            return replace(real, steps=(replace(step, inputs=step.inputs[:1]),))
        if mutation == 'hidden-temporary':
            return replace(real, tensors=(w.Tensor('val54.hidden', (1,), 'fp32'),))
        return replace(real, steps=(step, replace(step, name=step.name+'.extra')))
    monkeypatch.setattr(adapter, 'lower_implementation', injected_product)
    with prohibit_execution(monkeypatch, s):
        with pytest.raises(NotImplementedError, match='materialized|ordered'):
            adapter.lower(context)
    assert seen == [task.name]
