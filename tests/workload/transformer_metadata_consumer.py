"""Sole installed candidate: preflight request-only rows; optionally build tiny metadata.

No full-size Graph, numerical roots, compile, prepare or service is executed.
"""
import argparse
from contextlib import ExitStack
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
from unittest.mock import patch

from transformer_metadata_expectations import decoder_ledger, tiny_request

INPUT_SHA='66d68308d446967676b5818c84cdb3f758ebde2d32f42a58051bb14febc3ad14'


def formal_inputs(path):
    raw=Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest()!=INPUT_SHA:raise ValueError('Frozen TRANSFORMER-INPUTS identity changed')
    data=json.loads(raw)
    assert data['schema']=='sim52.transformer-inputs.v1' and len(data['rows'])==16
    names=[row['name'] for row in data['rows']]
    assert len(set(names))==len(names)
    assert {row['workload']['params']['model']['name'] for row in data['rows']}=={
        'llama2-7b','llama2-70b','llama3-70b','qwen-72b'}
    for row in data['rows']:
        assert row['workload']['kind']=='transformer'
        p=row['workload']['params']
        assert p['layers']==p['model']['layers'] and row['tp'] in (1,8)
        assert p['batch']==1 and p['seq_len']==2048 and p['phase'] in ('prefill','decode')
    return data


def no_graph_objects():
    from arch_explorer.workload import operators
    guards=ExitStack()
    def forbidden(*args,**kwargs):raise AssertionError('preflight/refusal attempted Graph/Tensor/Operator construction')
    for cls in (operators.Workload,operators.Tensor,operators.Operator):
        guards.enter_context(patch.object(cls,'__init__',forbidden))
    return guards


def audit_preflight(actual,params,tp):
    assert set(actual)=={'schema','request','budget','metrics','admitted','violations','semantics','plan','stages'}
    assert actual['schema']=='arch-explorer.transformer-preflight.v1'
    request=deepcopy(params);budget=request.pop('resource_budget');request['tp']=tp
    assert actual['request']==request and actual['budget']==budget
    expected=decoder_ledger(params,tp)
    assert set(actual['metrics'])==set(expected)|{'estimated_host_bytes'}
    for name,value in actual['metrics'].items():assert type(value) is int and value>=0,name
    for name,value in expected.items():assert actual['metrics'][name]==value,(name,actual['metrics'][name],value)
    for violation in actual['violations']:
        assert set(violation)=={'field','required','limit','budget_field'}
        assert type(violation['field']) is str and type(violation['budget_field']) is str
        assert type(violation['required']) is int and type(violation['limit']) is int
        assert violation['required']==actual['metrics'][violation['field']]
        assert violation['limit']==actual['budget'][violation['budget_field']]
        assert violation['required']>violation['limit']
    assert type(actual['admitted']) is bool and actual['admitted']==(not actual['violations'])
    assert actual['plan']==dict(status='mapping-required',minimum_task_instances=expected['operators']*request['iterations'],
                                replicas=None,physical_peak_bytes=None)
    assert type(actual['semantics']) is str and actual['semantics']
    assert set(actual['stages'])=={'graph','compile','service'}
    assert all(type(v) is str and v for v in actual['stages'].values())
    return expected


def audit_wrapper(wrapped,case):
    assert set(wrapped)=={'schema','case','status','producer','result','graph_executed','service_executed'}
    assert wrapped['schema']=='arch-explorer.scale-preflight.v1'
    assert wrapped['case']==case
    assert wrapped['producer']=='arch_explorer.workload.api.transformer_preflight'
    assert wrapped['status']==('admitted' if wrapped['result']['admitted'] else 'budget-refused')
    assert wrapped['graph_executed'] is False and wrapped['service_executed'] is False
    return wrapped['result']


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--inputs',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--tiny-graph',action='store_true')
    args=parser.parse_args()
    import arch_explorer
    assert 'site-packages' in Path(arch_explorer.__file__).parts,'Requires sole installed candidate'
    from arch_explorer.adf import WorkloadSpec
    from arch_explorer.workload import api as w
    from arch_explorer.archs.paper_scale import preflight_scale_case
    data=formal_inputs(args.inputs)
    args.output.mkdir(parents=True,exist_ok=False)
    facts=dict(schema='arch-explorer.val63.transformer-metadata-consumer.v1',input_sha256=INPUT_SHA,
        product_file=arch_explorer.__file__,executable=sys.executable,rows=[],tiny=None,passed=False,
        scope='logical metadata only; no full-size Graph/compile/service/numerical/physical-capacity acceptance')
    def save():
        (args.output/'RESULT.json').write_text(json.dumps(facts,indent=2,allow_nan=False)+'\n')
    try:
        for row in data['rows']:
            params=deepcopy(row['workload']['params']);record=dict(name=row['name'],tp=row['tp'])
            facts['rows'].append(record)
            case=dict(id=row['name'],family='llm-performance',workload=deepcopy(row['workload']),tp=row['tp'])
            before=deepcopy(case)
            with no_graph_objects():
                wrapped=preflight_scale_case(case)
            record['wrapper']=wrapped;save()
            assert case==before
            actual=audit_wrapper(wrapped,before);record['actual']=actual
            record['expected']=audit_preflight(actual,params,row['tp']);save()
        if args.tiny_graph:
            tiny=dict(scope='synthetic small metadata only',observations=[]);facts['tiny']=tiny
            common=deepcopy(data['rows'][0]['workload']['params']['resource_budget'])
            for tp in (1,2):
                params=tiny_request();params['resource_budget']=deepcopy(common)
                record=dict(tp=tp);tiny['observations'].append(record)
                with no_graph_objects():actual=w.transformer_preflight(**params,tp=tp)
                record['preflight']=actual;save()
                expected=audit_preflight(actual,params,tp);assert actual['admitted']
                graph=w.workload_from_spec(WorkloadSpec('transformer',params),tp=tp)
                graph_data=graph.to_dict();record['graph']=graph_data;save()
                assert len(graph.operators)==expected['operators'] and len(graph.tensors)==expected['tensors']
                assert graph.meta['resource_budget']==actual['budget']
                assert not graph.meta.get('inputs')
                old=deepcopy(params);old.pop('resource_budget')
                old_graph=w.workload_from_spec(WorkloadSpec('transformer',old),tp=tp).to_dict()
                without_budget=deepcopy(graph_data);without_budget['meta'].pop('resource_budget')
                assert without_budget==old_graph
                # The boundary is total logical bytes, not physical SRAM usage.
                for delta in (0,-1):
                    boundary=deepcopy(params)
                    boundary['resource_budget']['max_total_tensor_bytes']=expected['total_tensor_bytes']+delta
                    boundary['resource_budget']['max_root_bytes']=1  # No numeric roots are materialized here.
                    case=dict(id='tiny-override',family='llm-performance',tp=tp,
                              workload=dict(kind='transformer',params=deepcopy(params)))
                    before=deepcopy(case)
                    with no_graph_objects():
                        wrapped=preflight_scale_case(case,resource_budget=deepcopy(boundary['resource_budget']))
                    record.setdefault('boundaries',[]).append(dict(delta=delta,wrapper=wrapped));save()
                    assert case==before
                    answer=audit_wrapper(wrapped,before)
                    audit_preflight(answer,boundary,tp)
                    assert answer['admitted']==(delta==0)
                    if delta<0:
                        assert any(v['budget_field']=='max_total_tensor_bytes' for v in answer['violations'])
                        with no_graph_objects():
                            try:w.workload_from_spec(WorkloadSpec('transformer',boundary),tp=tp)
                            except ValueError as error:
                                assert str(error).startswith('transformer resource_budget refused:')
                                record['refusal']=str(error)
                            else:raise AssertionError('Missing before-Graph budget refusal')
            save()
        facts['passed']=True
    except BaseException as error:
        facts.update(error_type=type(error).__name__,error=str(error));raise
    finally:save()


if __name__=='__main__':main()
