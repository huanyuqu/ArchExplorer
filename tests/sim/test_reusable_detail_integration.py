from fractions import Fraction
import pytest
from arch_explorer.adf import evaluate, MappingSpec
from arch_explorer.devices.pim import neupims
from arch_explorer.simulator.events.engine import Task,TaskGraph,simulate
from arch_explorer.simulator.events.protocol import ExecutionRequest
from arch_explorer.simulator.events.controller_execution import ControllerExecutionModel
from arch_explorer.components.transaction_controller import Channel
from test_neupims_instruction_program import small_workload


def test_public_path_uses_graph_not_old_whole_program_callback(monkeypatch):
    import arch_explorer.simulator.events.neupims_memory as memory
    import arch_explorer.simulator.events.neupims_program as old
    def forbidden(*args,**kwargs):raise AssertionError('old standalone runner was called')
    monkeypatch.setattr(memory,'run_core',forbidden)
    monkeypatch.setattr(memory.TransactionMemory,'wait',forbidden)
    assert not hasattr(old, 'legacy_piece')
    result=evaluate(neupims(64),small_workload(),MappingSpec('pim_offload',
        {'npu_execution':'instructions','memory_execution':'transactions'}),cost_model=False)
    assert result.details['stats']['scheduler']=='conservative'
    assert result.details['stats']['execution_events']
    from arch_explorer.components.execution_profiles import NEUPIMS_INSTRUCTIONS
    original=NEUPIMS_INSTRUCTIONS.spad_bytes
    result.details['execution']['profile']['spad_bytes']=1
    assert NEUPIMS_INSTRUCTIONS.spad_bytes==original


def test_shared_endpoint_and_independent_clock_do_not_barrier():
    graph=TaskGraph();instances={}
    for name,clock in [('shared',1),('independent',Fraction(1,2))]:
        def factory(name=name,clock=clock):
            instances[name]=ControllerExecutionModel(Channel(),clock)
            return instances[name]
        graph.bind_execution(name,factory,(name+'.port',))
    requests=[('normal','shared','READ',{'address':0}),
              ('pim','shared','GWRITE',{'address':0}),
              ('other','independent','READ',{'address':0})]
    for name,owner,kind,payload in requests:
        graph.add(Task(name,execution=ExecutionRequest(name,owner,operation=kind,payload=payload)))
    result=simulate(graph,'conservative')
    times={r.name:r.end_ns for r in result.records}
    # Source prioritizes GWRITE, so the co-resident READ waits behind it.
    assert times['other']<times['pim']<times['normal']
    for model in instances.values():
        trace=model.controller.trace
        assert len({r['time'] for r in trace})==len(trace)
    assert set(instances['shared'].controller.counts)>={'RD','GWRITE'}


def test_request_and_demand_cannot_bill_same_port():
    graph=TaskGraph()
    graph.bind_execution('controller',lambda:ControllerExecutionModel(Channel()),('port',))
    from arch_explorer.simulator.events.engine import Resource
    graph.add_resource(Resource('port',1))
    graph.add(Task('duplicate',demand={'port':1}))
    with pytest.raises(ValueError,match='both Demand'):
        simulate(graph,'conservative')
