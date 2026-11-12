"""VAL54 behavioral boundaries; baseline failures are intentional evidence.

These tests use existing semantic/mapping/events entrances, not a guessed new
adapter API. The frozen SIM43 API will supply the additional installed binding,
ambiguity, options, source and task-group consumers in the next stage.
"""
import json
import os
import subprocess
import sys

import pytest


COLD_SCRIPT = r'''
import importlib
import importlib.abc
import importlib.util
import json
import sys

class NativeBoundaryCrossed(ImportError):
    pass

class DenyNative(importlib.abc.MetaPathFinder):
    attempts = []
    def find_spec(self, fullname, path=None, target=None):
        forbidden = (
            'arch_explorer.components.sm.implementations.accelsim',
            'arch_explorer.components.accelsim.program',
            'arch_explorer.components.accelsim.open_program',
        )
        if any(fullname == p or fullname.startswith(p+'.') for p in forbidden):
            self.attempts.append(fullname)
            raise NativeBoundaryCrossed('native boundary crossed: '+fullname)

blocker = DenyNative()
if '--native-allowed' not in sys.argv:
    sys.meta_path.insert(0, blocker)
mode = sys.argv[1]
if mode == 'sensitivity':
    try:
        importlib.import_module('arch_explorer.components.sm.implementations.accelsim.lowering_limits')
    except NativeBoundaryCrossed:
        print('native-import-denial-active')
        sys.exit(0)
    raise AssertionError('import blocker is inactive')

from arch_explorer.adf import WorkloadSpec, MappingSpec
from arch_explorer.workload.api import workload_from_spec, execute_reference
# Import/install of public semantics itself may not touch native capability
# modules. Later discovery may try optional registrars, whose local failure
# must leave this valid semantic/events workflow usable.
assert not blocker.attempts, ('public semantic import touched native', blocker.attempts)
graph = workload_from_spec(WorkloadSpec('operator', dict(op='matmul', m=3, k=3, n=5, dtype='fp32')))
produced = {name for op in graph.operators for name in op.outputs}
inputs = [name for name in graph.tensors if name not in produced]
assert len(inputs) == 2
graph.meta['inputs'] = {
    inputs[0]: [[[1, 2, -1], [0, 3, 2], [-2, 1, 4]]],
    inputs[1]: [[[2, 0, 1, -1, 3], [1, 4, -2, 2, 0], [3, -1, 0, 5, 2]]],
}
spec = WorkloadSpec('graph', graph.to_dict())
reference = execute_reference(spec)
assert next(iter(reference['outputs'].values())).tolist() == [[[1,9,-3,-2,1],[9,10,-6,16,4],[9,0,-4,24,2]]]
if mode == 'events':
    # Existing ordinary event path must survive disabling a different backend.
    from arch_explorer.adf import Param, SpacePoint, SpaceMatrix, evaluate
    from arch_explorer.components import register_components
    from arch_explorer.adf import COMPONENTS
    register_components()
    source = 'synthetic VAL54 CPU independence control'
    param = lambda value, unit: Param(value, unit, source)
    unit = SpacePoint('vector', 'vector_unit', {
        'lanes': param(8, 'count'), 'sfu_lanes': param(1, 'count'),
        'clock_ghz': param(1, 'GHz'), 'precision': param('fp32', '1'),
    })
    memory = SpacePoint('dram', 'dram', {
        'technology': param(COMPONENTS.get('dram').params['technology'].domain[0], '1'),
        'capacity': param(2**20, 'B'), 'bandwidth': param(8, 'B/ns'),
        'latency': param(2, 'ns'),
    })
    result = evaluate(SpaceMatrix('board', [unit, memory]), spec,
                      MappingSpec('execution_graph', {}), backend='events')
    assert result.latency_ns > 0
loaded = [name for name in sys.modules if any(name == prefix or name.startswith(prefix+'.')
          for prefix in ('arch_explorer.components.sm.implementations.accelsim',
                         'arch_explorer.components.accelsim.program',
                         'arch_explorer.components.accelsim.open_program'))]
assert not loaded, loaded
print(json.dumps({'mode':mode,'native_imports':loaded,'reference':'scalar-exact',
                  'optional_discovery_attempts':len(blocker.attempts)}))
'''


def cold_run(mode, *, native_allowed=False):
    # Preserve the selected checkout/wheel resolution. No product imports in
    # this parent can warm registrations and hide an unconditional dependency.
    env = {k: v for k, v in os.environ.items() if k != 'PYTHONHOME'}
    command = [sys.executable, '-c', COLD_SCRIPT, mode]
    if native_allowed:
        command.append('--native-allowed')
    result = subprocess.run(command, env=env,
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    return result.stdout.strip()


def test_cold_native_import_denial_is_sensitive():
    assert cold_run('sensitivity') == 'native-import-denial-active'


@pytest.mark.parametrize('mode', ['semantic', 'events'])
def test_unblocked_control_uses_the_same_legal_public_input(mode):
    assert json.loads(cold_run(mode, native_allowed=True))['reference'] == 'scalar-exact'


@pytest.mark.parametrize('mode', ['semantic', 'events'])
def test_public_semantics_and_events_survive_disabled_native_modules(mode):
    observed = json.loads(cold_run(mode))
    assert observed['mode'] == mode and observed['reference'] == 'scalar-exact'
    assert observed['native_imports'] == []
    assert observed['optional_discovery_attempts'] >= 0


def test_generic_mapping_uses_placement_and_registered_rule_without_gpu_options(monkeypatch):
    from arch_explorer.workload.api import compile_workload, Workload
    from arch_explorer.simulator import api as simulator_api
    from test_public_compilation_contract import roofline_input,prohibit_execution
    adf,graph,spec,mapping = roofline_input(m=3,k=3,n=5,tiles={'m':2,'k':2,'n':3})
    assert set(mapping.params)=={'compute','memory','rules'}
    with prohibit_execution(monkeypatch,simulator_api):
        compilation = compile_workload(adf,spec,mapping)
    mapped = Workload.from_dict(compilation['program'])
    assert {op.op for op in mapped.operators} == {'compiled.matmul', 'compiled.assemble'}
    assert all('threads_per_block' not in op.attrs for op in mapped.operators)
    assert sorted(t.shape for n,t in mapped.tensors.items() if n not in graph.tensors) == [
        (1, 1, 2), (1, 1, 3), (1, 2, 2), (1, 2, 3)]
