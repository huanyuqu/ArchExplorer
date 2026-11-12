"""Generate QA references for frozen actual framework exports, no product.

Consumes a declared no-answer INPUTS bundle. Stores independent intermediate
values separately from official intermediate values and records every comparison.
"""
import argparse
import hashlib
import json
from pathlib import Path
import traceback
import numpy as np
from framework_continuation_numpy import evaluate


def write(path,value):
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def run(package,case_id,output,*,framework_module=None):
    import onnx
    from onnx.reference import ReferenceEvaluator
    package=Path(package).resolve();output=Path(output);output.mkdir(parents=True,exist_ok=False)
    document=json.loads((package/'INPUTS.json').read_text())
    case=next(c for c in document['cases'] if c['id']==case_id)
    manifest=json.loads((package/'MANIFEST.json').read_text())
    relevant={'INPUTS.json',case['model']}
    directory=str(Path(case['model']).parent)+'/'
    relevant.update(r['path'] for r in manifest['files'] if r['path'].startswith(directory))
    inspected=[]
    for row in manifest['files']:
        if row['path'] not in relevant:continue
        path=package/row['path'];raw=path.read_bytes();digest=hashlib.sha256(raw).hexdigest()
        if len(raw)!=row['bytes'] or digest!=row['sha256']:raise ValueError('frozen input manifest mismatch: '+row['path'])
        inspected.append(row)
    result=dict(schema='arch-explorer.val64.framework-reference.v1',case_id=case_id,status='running',
                product_invocations=0,onnx=onnx.__version__,numpy=np.__version__,inputs=inspected,
                tolerances=dict(fp32=dict(atol=5e-5,rtol=5e-5)),official_is_low_precision_oracle=False,
                floating_policy='per original node: chosen sequential FP32 accumulation and declared output storage')
    try:
        model=onnx.load(package/case['model']);types={v.name:v.type.tensor_type.elem_type for v in model.graph.input}
        feeds={n:np.asarray(case['inputs'][n],dtype=np.int64 if types[n]==7 else np.float32) for n in case['input_names']}
        independent=evaluate(model,feeds)
        np.savez_compressed(output/'INDEPENDENT.npz',**independent['stages'])
        names=list(independent['stages'])
        official={} if case['dtype']!='fp32' else dict(zip(names,ReferenceEvaluator(model).run(names,feeds)))
        np.savez_compressed(output/'OFFICIAL.npz',**{k:v for k,v in official.items() if isinstance(v,np.ndarray)})
        rows=[]
        for name,expected in independent['stages'].items():
            if name not in official:continue
            actual=np.asarray(official[name]);expected=np.asarray(expected)
            exact=independent['types'][name] in ('int64','int32','bool')
            valid=actual.shape==expected.shape and np.isfinite(actual).all() and np.isfinite(expected).all()
            equal=valid and (np.array_equal(actual,expected) if exact else np.allclose(actual,expected,atol=5e-5,rtol=5e-5))
            rows.append(dict(value=name,dtype=independent['types'][name],shape=list(expected.shape),exact=exact,
                max_absolute_error=None if not valid else float(np.max(np.abs(actual.astype(np.float64)-expected.astype(np.float64)),initial=0)),
                passed=bool(equal)))
        result.update(nodes=independent['nodes'],comparisons=rows,output_names=list(independent['outputs']),
                      status=('passed' if all(r['passed'] for r in rows) else 'failed') if official else 'reference-generated')
        if framework_module:
            import importlib.util
            import torch
            torch.set_num_threads(1);torch.set_num_interop_threads(1)
            module_path=Path(framework_module).resolve()
            entry=importlib.util.spec_from_file_location('qa_frozen_framework',module_path)
            module=importlib.util.module_from_spec(entry);entry.loader.exec_module(module)
            actual,inputs,input_names,output_names,_,_=module.make_model_and_inputs(case['family'],case['dtype'])
            if input_names!=case['input_names'] or output_names!=case['output_names']:raise ValueError('framework input/output order differs')
            torch_dtypes={'fp32':torch.float32,'fp16':torch.float16,'bf16':torch.bfloat16,'int64':torch.int64}
            tensors=tuple(torch.tensor(inputs[n]['values'],dtype=torch_dtypes[inputs[n]['dtype']]) for n in input_names)
            if any(t.device.type!='cpu' for t in [*actual.parameters(),*tensors]):raise ValueError('QA requires CPU framework')
            if not all(np.array_equal(t.detach().float().numpy() if t.is_floating_point() else t.numpy(),feeds[n])
                       for n,t in zip(input_names,tensors)):raise ValueError('framework roots differ from frozen ONNX inputs')
            with torch.no_grad():answer=actual(*tensors)
            answer=answer if isinstance(answer,tuple) else (answer,)
            forward={n:t.detach().float().numpy() for n,t in zip(output_names,answer)}
            np.savez_compressed(output/'FRAMEWORK.npz',**forward)
            errors=[]
            for n,reference in forward.items():
                computed=independent['outputs'][n];delta=np.abs(reference-computed)
                errors.append(dict(output=n,max_absolute_error=float(delta.max(initial=0)),
                    passed_fp32_tolerance=bool(np.allclose(reference,computed,atol=5e-5,rtol=5e-5)) if case['dtype']=='fp32' else None))
            result['framework']=dict(version=torch.__version__,module_sha256=hashlib.sha256(module_path.read_bytes()).hexdigest(),
                cpu=True,new_export_calls=0,errors=errors,low_precision_classification='informational; no tuned threshold')
            if case['dtype']=='fp32' and not all(r['passed_fp32_tolerance'] for r in errors):result['status']='failed'
        if result['status']=='failed':raise AssertionError('independent/official value difference')
    except BaseException as error:
        result.update(status='failed',error_type=type(error).__name__,error=str(error),traceback=traceback.format_exc())
        raise
    finally:write(output/'RESULT.json',result)
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--package',required=True);p.add_argument('--case',required=True);p.add_argument('--output',required=True)
    p.add_argument('--framework-module')
    args=p.parse_args();r=run(args.package,args.case,args.output,framework_module=args.framework_module)
    print(json.dumps(dict(case=r['case_id'],status=r['status'],outputs=r['output_names'])))


if __name__=='__main__':main()
