"""Independent typed and receipt checks; no product or lowering imports."""
import hashlib
import json
from pathlib import Path
import numpy as np
from framework_continuation_numpy import explicit_cast
from typed_onnx_expectations import cast,numeric_error


def digest(path):
    data=Path(path).read_bytes()
    return dict(path=str(Path(path).resolve()),bytes=len(data),sha256=hashlib.sha256(data).hexdigest())


def shaped(value,shape,dtype):
    a=np.asarray(value)
    if dtype in ('int32','int64'):
        if a.dtype.kind not in 'iu':raise ValueError('integer observation passed through float/bool storage')
        limits=np.iinfo(np.dtype(dtype))
        if any(not limits.min<=int(v)<=limits.max for v in a.flat):raise ValueError('integer observation outside declared dtype')
    elif dtype=='bool':
        if a.dtype.kind!='b':raise ValueError('bool observation lost boolean type')
    elif dtype in ('fp32','fp16','bf16'):
        a=np.asarray(value,dtype=np.float64)
        if not np.isfinite(a).all():raise ValueError('floating observation is nonfinite')
    else:raise ValueError('unknown observed dtype')
    try:return a.reshape(tuple(shape))
    except ValueError as error:raise ValueError('observation logical shape differs') from error


def compare_value(actual,expected,*,dtype,shape):
    a,b=shaped(actual,shape,dtype),shaped(expected,shape,dtype)
    if dtype in ('int32','int64','bool'):
        exact=np.array_equal(a,b)
        if not exact:raise ValueError('integer/bool observation differs exactly from original value')
        return dict(dtype=dtype,shape=list(shape),exact=True)
    # Report numerical drift separately; low-precision tolerance needs the
    # formal producer accumulation contract, not a post-hoc fit to observations.
    if not np.array_equal(cast(a,dtype),a):raise ValueError('floating observation is outside declared storage grid')
    return dict(dtype=dtype,exact=bool(np.array_equal(a,b)),storage_grid=True,**numeric_error(a,b),
        passed_fp32_tolerance=bool(np.allclose(a,b,atol=5e-5,rtol=5e-5)) if dtype=='fp32' else None)


def check_cast(source,target,*,target_dtype,target_shape):
    expected=explicit_cast(source,target_dtype).reshape(tuple(target_shape))
    actual=shaped(target,target_shape,target_dtype)
    if not np.array_equal(actual,expected):raise ValueError('explicit Cast differs from independent conversion')
    if target_dtype in ('fp32','fp16','bf16'):
        zeros=(expected==0)&(actual==0)
        if not np.array_equal(np.signbit(actual[zeros]),np.signbit(expected[zeros])):
            raise ValueError('explicit Cast zero sign differs')
    return dict(target_dtype=target_dtype,shape=list(target_shape),exact=True)


def check_ledger(result,*,limit):
    """Audit pass-through receipts; complete means all reported attempts closed.

    An interrupted process without a final receipt is not inferred complete.
    This checker does not predict how many proposals a policy should generate.
    """
    if type(limit) is not int or limit<1:raise ValueError('invalid evaluation limit')
    attempts=result['evaluation_attempts'];used=result['evaluations_used']
    if type(used) is not int or not isinstance(attempts,list) or not 0<=used<=limit or used!=len(attempts):
        raise ValueError('evaluation budget/receipt count differs')
    scores=[]
    if result['max_evaluations']!=limit:raise ValueError('saved budget differs from caller limit')
    for index,attempt in enumerate(attempts):
        if type(attempt['index']) is not int or attempt['index']!=index:raise ValueError('attempt identity/order differs')
        if type(attempt['valid']) is not bool or type(attempt['execution_attempted']) is not bool:
            raise ValueError('attempt flags must be boolean')
        objective=(attempt.get('score') or {}).get('objective')
        if attempt['valid'] and objective is None:
            # Actual environment receipts bind a saved compiled result. They
            # do not duplicate the candidate score in each workload attempt.
            best=result['best']
            if best is None or len(best['workloads'])!=1 or attempt.get('workload')!=0:
                raise ValueError('receipt-only objective audit requires one complete workload')
            config=result['objective_config']
            if config['cost_budget_usd'] is not None or config['area_budget_mm2'] is not None:
                raise ValueError('receipt-only objective audit requires unpenalized latency')
            receipt=attempt['public_result'];raw=Path(receipt['path']).read_bytes()
            if hashlib.sha256(raw).hexdigest()!=receipt['sha256']:
                raise ValueError('attempt public result digest differs')
            compiled=json.loads(raw)
            if compiled['schema']!='arch-explorer.compiled-result.v3':
                raise ValueError('attempt public result schema differs')
            if (compiled['binding']!=attempt['result_binding'] or
                compiled['request']['binding']!=attempt['request_binding'] or
                compiled['request']['compilation']['binding']!=attempt['compilation_binding']):
                raise ValueError('attempt public result bound envelope differs')
            source=compiled['request']['compilation']['source']
            if json.dumps(source['mapping'],sort_keys=True)!=json.dumps(attempt['effective_mapping'],sort_keys=True):
                raise ValueError('attempt effective mapping differs from bound source')
            weight=best['workloads'][0]['workload']['weight']
            if isinstance(weight,bool) or not isinstance(weight,(int,float)) or not np.isfinite(weight) or weight<=0:
                raise ValueError('receipt workload weight must be finite and positive')
            objective=compiled['evaluation']['latency_ns']*weight
        if attempt['valid'] and objective is not None:
            if isinstance(objective,bool) or not isinstance(objective,(int,float)) or not np.isfinite(objective):
                raise ValueError('successful receipt objective is nonfinite')
            scores.append(objective)
    best=result['best']
    if best is None:
        if scores:raise ValueError('successful receipts have no winner')
    else:
        objective=best['score']['objective']
        if isinstance(objective,bool) or not isinstance(objective,(int,float)) or not np.isfinite(objective):
            raise ValueError('winner objective is not a finite number')
        if not scores or objective!=min(scores):raise ValueError('winner does not match receipt objectives')
        for item in best['workloads']:
            source=item['compiled_result']['request']['compilation']['source']
            # JSON keeps bool and integer identity distinct.
            for key in ('mapping','workload'):
                if json.dumps(item[key],sort_keys=True)!=json.dumps(source[key],sort_keys=True):
                    raise ValueError('winner '+key+' differs from bound compilation source')
    return dict(used=used,limit=limit,invalid_attempts=sum(not a['valid'] for a in attempts),
        executed_attempts=sum(a['execution_attempted'] for a in attempts),winner_present=best is not None,
        status='closed-receipt-consistent; no assertion about unrecorded attempts')


def original_roots(model,input_shapes):
    """Logical root byte expectations from original typed protobuf metadata.

    A scalar occupies one element; float constants never relabel integer or
    bool roots. These are source sizes, not a prediction of device liveness.
    """
    import math
    widths={1:('fp32',4),10:('fp16',2),16:('bf16',2),6:('int32',4),7:('int64',8),9:('bool',1)}
    roots={}
    def add(name,shape,code,role):
        if name in roots:raise ValueError('original root identities overlap')
        if code not in widths:raise ValueError('original root dtype outside prepared domain')
        dtype,width=widths[code]
        if any(type(d) is not int or d<0 for d in shape):raise ValueError('original root shape differs')
        elements=math.prod(shape)
        roots[name]=dict(logical_shape=list(shape),dtype=dtype,elements=elements,bytes=elements*width,role=role)
    for tensor in model.graph.input:
        add(tensor.name,input_shapes[tensor.name],tensor.type.tensor_type.elem_type,'input')
    for tensor in model.graph.initializer:add(tensor.name,list(tensor.dims),tensor.data_type,'initializer')
    for node in model.graph.node:
        if node.op_type!='Constant':continue
        attributes={a.name:a for a in node.attribute}
        if set(attributes)!={'value'} or len(node.output)!=1:raise ValueError('prepared roots require Tensor-valued Constant')
        tensor=attributes['value'].t;add(node.output[0],list(tensor.dims),tensor.data_type,'constant')
    return roots


def bind_reference(case,package,record):
    """Bind a saved independent result to exact model/manifest metadata.

    Full frozen input closure is checked by the installation harness once.
    This check reuses manifest digests without rereading every weight file.
    """
    package=Path(package);record=Path(record)
    report=json.loads(record.read_text())
    if report['case_id']!=case['id']:raise ValueError('reference case differs')
    manifest=json.loads((package/'MANIFEST.json').read_text())
    known={r['path']:r for r in manifest['files']}
    referenced={}
    for row in report['inputs']:
        name=row['path'];relative=Path(name)
        if relative.is_absolute() or '..' in relative.parts or name in referenced:
            raise ValueError('reference manifest paths must be unique and package-relative')
        referenced[name]=row
    for name in ('INPUTS.json',case['model']):
        if name not in referenced or name not in known:raise ValueError('reference lacks frozen input identity')
        if referenced[name]['sha256']!=known[name]['sha256']:raise ValueError('reference input digest differs')
    if digest(package/case['model'])['sha256']!=known[case['model']]['sha256']:raise ValueError('model no longer matches frozen manifest')
    types={name:dtype for node in report['nodes'] for name,dtype in zip(node['outputs'],node['output_dtypes'])}
    values=np.load(record.with_name('INDEPENDENT.npz'),allow_pickle=False)
    if set(values.files)!=set(types):raise ValueError('reference original value identities differ')
    if report['output_names']!=case['output_names']:raise ValueError('reference output order differs')
    # A retained wrapper failure remains visible and requires final review.
    return dict(record=digest(record),values=digest(record.with_name('INDEPENDENT.npz')),
                raw_reference_status=report['status'],reference_requires_review=report['status']=='failed'),types,values
