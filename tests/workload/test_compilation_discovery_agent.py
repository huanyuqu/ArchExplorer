"""Affected public discovery, custom-registry guard and Agent budget contracts."""
from copy import deepcopy
from uuid import uuid4

import pytest

from test_public_compilation_contract import compiler_api, prohibit_execution, roofline_input


def test_execution_guard_blocks_harmless_sentinels_and_restores_entrypoints(compiler_api, monkeypatch):
    """A missing with must fail without risking any real execution."""
    import ctypes
    import os
    import subprocess
    from arch_explorer.simulator.events import engine
    _, s = compiler_api
    entries = [(s, 'evaluate_request'), (subprocess, 'Popen'),
               (ctypes, 'CDLL'), (ctypes, 'PyDLL'), (os, 'system'),
               (engine, 'simulate'), (engine.ConsistentScheduler, 'run')]
    observed = []
    def harmless(*args, **kwargs):
        observed.append('sentinel')
        return 'harmless sentinel'
    for owner, name in entries:
        monkeypatch.setattr(owner, name, harmless)
        assert getattr(owner, name)() == 'harmless sentinel'
    before = len(observed)
    with prohibit_execution(monkeypatch, s):
        for owner, name in entries:
            with pytest.raises(AssertionError, match='attempted simulation'):
                getattr(owner, name)()
    assert len(observed) == before
    for owner, name in entries:
        assert getattr(owner, name) is harmless
        assert getattr(owner, name)() == 'harmless sentinel'


def test_adapter_discovery_is_execution_free_and_detects_mutated_registration(compiler_api, monkeypatch):
    w, s = compiler_api
    with prohibit_execution(monkeypatch, s):
        original = w.task_adapters()
        assert {'events', 'roofline', 'accelsim'} <= {r['id'] for r in original}
        assert all({'id', 'version', 'lower'} <= set(r) for r in original)
        assert w.task_adapters() == original
        def never_lower(context):
            raise AssertionError('discovery must not lower a workload')
        target = w.TaskAdapter('qa-discovery-' + uuid4().hex, '1', 'events',
                               ('compiled.matmul', 'compiled.assemble'), never_lower)
        w.register_task_adapter(target)
        assert any(r['id'] == target.id and r['version'] == '1' for r in w.task_adapters())
        try:
            object.__setattr__(target, 'version', '2')
            with pytest.raises(ValueError):
                w.task_adapters()
        finally:
            object.__setattr__(target, 'version', '1')
        assert any(r['id'] == target.id and r['version'] == '1' for r in w.task_adapters())


@pytest.mark.parametrize('operation', ['compile', 'validate', 'prepare'])
def test_process_local_registry_is_rejected_before_output(compiler_api, monkeypatch, tmp_path, operation):
    from arch_explorer.adf import COMPONENTS
    w, s = compiler_api
    adf, _, workload, mapping = roofline_input()
    compilation = w.compile_workload(adf, workload, mapping, registry=COMPONENTS)
    assert w.validate_compilation(compilation, registry=COMPONENTS) == compilation
    local = deepcopy(COMPONENTS)
    assert local is not COMPONENTS
    with prohibit_execution(monkeypatch, s):
        with pytest.raises(NotImplementedError, match='custom Registry'):
            if operation == 'compile':
                w.compile_workload(adf, workload, mapping, registry=local)
            elif operation == 'validate':
                w.validate_compilation(compilation, registry=local)
            else:
                s.prepare_compiled(compilation, output_dir=tmp_path/'must-not-exist', registry=local)
        assert not (tmp_path/'must-not-exist').exists()


def agent_scene():
    from arch_explorer.archs.compilation import events_scene
    from arch_explorer.agent_env import ExplorationEnv
    adf, workload, mapping = events_scene(fanout=True)
    env = ExplorationEnv({'seed_adf': adf.to_dict(), 'workloads': [workload.to_dict()],
                          'mapping_templates': ['execution_graph'], 'max_evaluations': 1})
    return env, mapping


def call_ok(env, tool, **arguments):
    response = env.call(tool, arguments)
    assert 'error' not in response, response
    return response['result']


def test_agent_compile_prepare_run_replay_and_exhausted_budget(compiler_api, monkeypatch, tmp_path):
    _, s = compiler_api
    env, mapping = agent_scene()
    seed, latest = deepcopy(env.adf.to_dict()), deepcopy(env.latest)
    with monkeypatch.context() as guard:
        with prohibit_execution(guard, s):
            compilation = call_ok(env, 'compile_workload', mapping=mapping.to_dict())
            request = call_ok(env, 'prepare_compiled', compilation=compilation,
                              output_dir=str(tmp_path/'agent'), options={'trace': True})
    assert env.used == 0
    result = call_ok(env, 'evaluate_request', payload=request)
    assert result['evaluation']['latency_ns'] > 0 and env.used == 1
    assert call_ok(env, 'evaluate_request', payload=result) == result
    assert env.used == 1
    exhausted = env.call('evaluate_request', {'payload': request})
    assert exhausted['error']['code'] == 'evaluation_budget_exhausted'
    assert env.used == 1 and env.latest == latest and env.adf.to_dict() == seed


def test_agent_failed_request_attempt_counts_but_bad_compile_does_not(compiler_api, tmp_path):
    env, mapping = agent_scene()
    invalid = env.call('compile_workload', {'mapping': mapping.to_dict(),
                                          'workload_index': 1})
    assert 'error' in invalid and env.used == 0
    compilation = call_ok(env, 'compile_workload', mapping=mapping.to_dict())
    request = call_ok(env, 'prepare_compiled', compilation=compilation, output_dir=str(tmp_path/'failed'))
    changed = deepcopy(request)
    changed['binding']['sha256'] = '0' * 64
    response = env.call('evaluate_request', {'payload': changed})
    assert 'error' in response and env.used == 1
    assert env.latest is None


def test_adapter_catalog_reaches_CLI_and_Agent_without_execution(compiler_api, monkeypatch):
    from contextlib import redirect_stdout
    import io
    import json
    from arch_explorer.cli import main
    w, s = compiler_api
    env, _ = agent_scene()
    def never_lower(context):
        raise AssertionError('catalog discovery attempted target conversion')
    unknown = w.TaskAdapter('qa-catalog-' + uuid4().hex, '1', 'events',
                            ('compiled.matmul', 'compiled.assemble'), never_lower)
    w.register_task_adapter(unknown)
    with prohibit_execution(monkeypatch, s):
        targets = w.task_adapters()
        assert any(row['id'] == unknown.id for row in targets)
        assert call_ok(env, 'task_adapters') == targets
        description = call_ok(env, 'describe_environment')
        assert description['task_adapters'] == targets
        assert 'task_adapters' in description['compilation']['tools']
        output = io.StringIO()
        with redirect_stdout(output):
            assert main(['task-adapters']) == 0
        assert json.loads(output.getvalue()) == targets
        assert env.used == 0


@pytest.mark.parametrize('operation', ['compile', 'prepare', 'migrate'])
def test_agent_keeps_custom_registry_rejection_before_output(compiler_api, monkeypatch, tmp_path, operation):
    from arch_explorer.adf import COMPONENTS, from_dict, WorkloadSpec, MappingSpec
    from arch_explorer.agent_env import ExplorationEnv
    w, s = compiler_api
    default, mapping = agent_scene()
    env = ExplorationEnv(default.config, components=deepcopy(COMPONENTS))
    compilation = w.compile_workload(from_dict(default.config['seed_adf']),
                                    WorkloadSpec(**default.config['workloads'][0]),
                                    mapping)
    with prohibit_execution(monkeypatch, s):
        if operation == 'compile':
            response = env.call('compile_workload', {'mapping': mapping.to_dict()})
        elif operation == 'prepare':
            response = env.call('prepare_compiled', {'compilation': compilation,
                                                   'output_dir': str(tmp_path/'forbidden')})
        else:
            # The registry guard must run before migration, even for this
            # well-formed historical fixture. Replacing its registry silently
            # would manufacture a different saved compilation identity.
            from pathlib import Path
            import json
            fixture = Path(__file__).resolve().parents[1]/'fixtures/adapter_boundary_extension/V1-COMPILATION.json'
            called = []
            def must_not_migrate(*args, **kwargs):
                called.append(True)
                raise AssertionError('Agent invoked migration before custom Registry refusal')
            monkeypatch.setattr(w, 'migrate_compilation_v1', must_not_migrate)
            response = env.call('migrate_compilation_v1', {'compilation':json.loads(fixture.read_text())})
            assert called == []
        assert 'error' in response and 'custom Registry' in str(response['error'])
        assert env.used == 0 and not (tmp_path/'forbidden').exists()


def test_agent_explicit_migration_is_free_and_retired_catalog_refuses(compiler_api, monkeypatch, capsys):
    import json
    from pathlib import Path
    from arch_explorer.cli import main
    _, s = compiler_api
    env, _ = agent_scene()
    fixture = Path(__file__).resolve().parents[1]/'fixtures/adapter_boundary_extension/V1-COMPILATION.json'
    with prohibit_execution(monkeypatch, s):
        migrated = call_ok(env, 'migrate_compilation_v1', compilation=json.loads(fixture.read_text()))
        assert migrated['schema'] == 'arch-explorer.compilation.v3'
        assert env.used == 0
        retired = env.call('compiler_targets', {})
        assert 'error' in retired and env.used == 0
        assert main(['compiler-targets']) == 2
    assert 'retired' in capsys.readouterr().err.lower()
