"""Sole-installed-candidate typed observations; no source-tree assembly.

General numerical error observations do not automatically establish acceptance.
Exact Cast/storage checks and physical audits are separately labelled. A final
acceptance driver must review per-node errors against the frozen precision
contract before declaring the whole numerical stage passed.
"""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
from typed_onnx_contract_cases import scene,formal_data
from typed_onnx_expectations import cast,numeric_error
from typed_access_expectations import audit_matmul_access


def run(input_root,oracle_root,case_id,*,index=0,policy='views',output_dir,component=False):
    import arch_explorer
    from arch_explorer.adf import WorkloadSpec
    from arch_explorer.workload import api as w
    assert 'site-packages' in arch_explorer.__file__,'Consume only the sole installed package'
    root=Path(output_dir);root.mkdir(parents=True,exist_ok=False)
    facts=dict(schema='arch-explorer.val63.typed-observation.v1',case=case_id,index=index,layout=policy,
        product_file=arch_explorer.__file__,status='running',observations=[],original_errors={},mapped_errors={},
        numeric_acceptance='pending-per-node-contract-review',component_executed=False,
        component_numeric_output_not_claimed=True)
    save=lambda: (root/'FACTS.json').write_text(json.dumps(facts,indent=2,allow_nan=False)+'\n')
    save()
    try:
        data=formal_data(input_root);row=next(r for r in data['cases'] if r['id']==case_id)
        oracle_path=Path(oracle_root)/case_id/'RESULT.json';raw=oracle_path.read_bytes();oracle=json.loads(raw)
        assert oracle['model_sha256']==hashlib.sha256((Path(input_root)/row['model']).read_bytes()).hexdigest()
        facts['oracle_source']=dict(path=str(oracle_path),sha256=hashlib.sha256(raw).hexdigest(),
                                   original_framework_version='2.5.1+cpu',independent_torch_counterpart='2.7.1+cu128 CPU-only')
        adf,spec,mapping,preflight,description=scene(input_root,case_id,index,policy)
        facts['preflight']=preflight;facts['description']=description
        compiled=w.compile_workload(adf,spec,mapping)
        (root/'COMPILED.json').write_text(json.dumps(compiled,indent=2)+'\n')
        original=w.execute_reference(spec)
        facts['CPU_original_executed']=True
        for name,wanted in oracle['stages'].items():
            meta=description['values'][name]
            actual=np.asarray(original['tensors'][meta['tensor']]).reshape(meta['logical_shape'])
            dtype=oracle['stage_dtypes'][name]
            facts['original_errors'][name]=dict(dtype=dtype,**numeric_error(actual,wanted))
            assert meta['dtype']==dtype
            np.testing.assert_array_equal(cast(actual,dtype),actual)
        facts['cast_exact_controls']=[]
        for node in description['nodes']:
            if node['op_type']!='Cast':continue
            src=description['values'][node['inputs'][0]];dst=description['values'][node['outputs'][0]]
            left=original['tensors'][src['tensor']];right=original['tensors'][dst['tensor']]
            np.testing.assert_array_equal(cast(left,dst['dtype']).reshape(dst['logical_shape']),np.asarray(right).reshape(dst['logical_shape']))
            facts['cast_exact_controls'].append(dict(source=src['tensor'],target=dst['tensor'],
                                                     source_dtype=src['dtype'],target_dtype=dst['dtype']))
        mapped=w.execute_reference(WorkloadSpec('graph',compiled['program']))
        facts['CPU_mapped_executed']=True
        for output in description['outputs']:
            wanted=np.asarray(oracle['outputs'][output['name']])
            actual=np.asarray(mapped['outputs'][output['tensor']]).reshape(wanted.shape)
            facts['mapped_errors'][output['name']]=numeric_error(actual,wanted)
        facts['matmul_access']=[audit_matmul_access(step) for step in compiled['plan']['steps'] if step['semantic']=='compiled.matmul']
        assert facts['matmul_access']
        facts['returned_roots']=[v['tensor'] for v in description['outputs']]
        assert set(facts['returned_roots'])==set(compiled['program']['meta']['outputs'])
        if component:
            import pytest
            from arch_explorer.simulator import api as s
            from multilayer_contract_cases import observe_memory
            from view_storage_runtime_cases import audit_storage
            request=s.prepare_compiled(compiled,output_dir=root/'request',options={'trace':True,'check_capacity':True})
            with pytest.MonkeyPatch.context() as patch:
                with observe_memory(patch) as observed:
                    facts['observations']=observed # Refer to the live list before any execution/assertion.
                    result=s.evaluate_request(request)
                    facts['component_executed']=True
                    (root/'RESULT.json').write_text(json.dumps(result,indent=2)+'\n')
                    save() # Raw observations survive subsequent physical-audit failures.
                    facts['storage']=audit_storage(compiled,result,observed)
            facts['views_peak_reduction_not_assumed']=True
        facts['status']='observation-complete'
        return facts
    except BaseException as error:
        facts.update(status='failed',error_type=type(error).__name__,error=str(error));raise
    finally:save()


def main():
    p=argparse.ArgumentParser();p.add_argument('--input-root',required=True);p.add_argument('--oracle-root',required=True)
    p.add_argument('--case',required=True);p.add_argument('--index',type=int,default=0)
    p.add_argument('--layout',choices=['materialized','views'],default='views');p.add_argument('--output-dir',required=True)
    p.add_argument('--component',action='store_true');a=p.parse_args()
    run(a.input_root,a.oracle_root,a.case,index=a.index,policy=a.layout,output_dir=a.output_dir,component=a.component)


if __name__=='__main__':main()
