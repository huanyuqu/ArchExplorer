"""Retained package and scheduler contracts after removing the unused pipeline."""
import importlib.util
import pytest

from arch_explorer.simulator.events import engine


@pytest.mark.parametrize('name', [
    'arch_explorer.metadse_sim', 'arch_explorer.hardware', 'arch_explorer.runtime',
    'arch_explorer.simulator.st_env', 'arch_explorer.simulator.st_context',
    'arch_explorer.simulator.scheduler', 'arch_explorer.simulator.st_model',
    'arch_explorer.simulator.action_model', 'arch_explorer.simulator.config',
    'arch_explorer.simulator.evaluation_model', 'arch_explorer.simulator.state',
    'arch_explorer.simulator.sync', 'arch_explorer.workload.task_model',
    'arch_explorer.workload.task_checker',
])
def test_retired_module_is_not_installable(name):
    assert importlib.util.find_spec(name) is None


@pytest.mark.parametrize('scheduler', ['legacy', 'not-a-scheduler'])
def test_removed_and_unknown_scheduler_use_normal_validation(scheduler):
    assert not hasattr(engine, 'LegacyScheduler')
    assert not hasattr(engine, '_run_to_completion')
    with pytest.raises(ValueError, match='Unknown scheduler'):
        engine.simulate(engine.TaskGraph(), scheduler)


def test_default_and_unified_keep_contention_and_independent_progress():
    graph = engine.TaskGraph(iterations=2)
    for resource in ('shared', 'independent'):
        graph.add_resource(engine.Resource(resource, 1))
    graph.add(engine.Task('a', demand={'shared': 10}))
    graph.add(engine.Task('b', demand={'shared': 10}))
    graph.add(engine.Task('c', demand={'independent': 2}))
    default = engine.simulate(graph)
    unified = engine.simulate(graph, 'conservative')
    times = lambda trace: {(r.name, r.iteration): (r.start_ns, r.release_ns, r.end_ns)
                           for r in trace.records}
    assert times(default) == times(unified)
    assert default.resource_usage == unified.resource_usage
    assert default.memory_peak == unified.memory_peak
    assert times(unified)['c', 0] == (0, 2, 2)
    assert times(unified)['a', 0] == (0, 20, 20)
    assert times(unified)['c', 1] == (2, 4, 4)


def test_standalone_nonmatrix_evaluator_is_absent():
    from arch_explorer.simulator.events import neupims_program
    assert not hasattr(neupims_program, 'legacy_piece')
