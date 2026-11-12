"""Candidate-only SIM51 import consumers. Collection is not acceptance."""
from copy import deepcopy
import base64,hashlib,importlib,json
from pathlib import Path
import numpy as np
import onnx
import pytest
from onnx_import_cases import mlp_case,attention_case,save_case
from onnx_import_expectations import ABS_TOL,REL_TOL,mlp,attention
from onnx_numpy_interpreter import evaluate_model
from test_operator_mapping_attention_contract import no_backend

FROZEN=Path(__file__).resolve().parents[1]/'fixtures/onnx_import'

@pytest.fixture
def onnx_api():
    import numpy.testing  # Its first import may probe lscpu; outside backend guards.
    import arch_explorer
    assert 'site-packages' in arch_explorer.__file__,'consume only the lead sole installed candidate'
    from arch_explorer.components.extensions import get_provider
    get_provider('events')
    module=importlib.import_module('arch_explorer.workload.onnx_import')
    w=importlib.import_module('arch_explorer.workload.api')
    s=importlib.import_module('arch_explorer.simulator.api')
    assert callable(module.import_onnx) and callable(module.describe_onnx)
    return module,w,s

def frozen(key):
    top=json.loads((FROZEN/'INPUTS.json').read_text());row=top['cases'][key]
    path=FROZEN/row['model'];data=json.loads((FROZEN/row['manifest']).read_text())
    assert hashlib.sha256(path.read_bytes()).hexdigest()==row['model_sha256']
    return path,data

def import_frozen(module,key):
    path,data=frozen(key)
    spec=module.import_onnx(path,inputs={n:r['values'] for n,r in data['feeds'].items()},
        external_root=path.parent if data['external_data'] else None)
    return spec,module.describe_onnx(spec)

def compare_original_values(reference,description,expected):
    compared=[]
    for original,wanted in expected.items():
        if np.asarray(wanted).dtype==np.int64:continue
        row=description['values'][original];actual=np.asarray(reference['tensors'][row['tensor']])
        assert tuple(row['logical_shape'])==np.asarray(wanted).shape
        assert actual.size==np.asarray(wanted).size
        np.testing.assert_allclose(actual.reshape(np.asarray(wanted).shape),wanted,atol=ABS_TOL,rtol=REL_TOL)
        compared.append(original)
    return compared

@pytest.mark.parametrize('key',['mlp','attention','mlp_external'])
def test_real_file_import_CPU_Graph_and_snapshot_are_complete(onnx_api,monkeypatch,key):
    module,w,s=onnx_api;path,data=frozen(key)
    case=attention_case() if key=='attention' else mlp_case()
    expected=(attention if key=='attention' else mlp)(case.roots)
    with no_backend(monkeypatch):
        spec,description=import_frozen(module,key)
        assert description['parser_dependencies']['onnx']=='1.16.2'
        assert description['parser_dependencies']['modules']
        assert spec.kind=='onnx' and set(spec.params)=={'name','source','inputs','input_shapes'}
        assert spec.params['source']['model']['sha256']==hashlib.sha256(path.read_bytes()).hexdigest()
        assert base64.b64decode(spec.params['source']['model']['base64'],validate=True)==path.read_bytes()
        graph=w.workload_from_spec(spec);reference=w.execute_reference(spec)
        assert compare_original_values(reference,description,expected)
        for original in case.outputs:
            assert description['values'][original]['tensor'] in graph.meta['outputs']
        assert len(graph.meta['outputs'])==len(case.outputs)
        assert module.describe_onnx(spec)==description
        tensors=[r['tensor'] for r in description['values'].values() if r['tensor'] is not None]
        assert len(tensors)==len(set(tensors))
        for name,record in data['external_data'].items():
            entry=spec.params['source']['external'][name]
            assert entry['sha256']==record['sha256']
            assert base64.b64decode(entry['base64'])==(path.parent/name).read_bytes()

def test_snapshot_is_portable_after_original_files_are_deleted(onnx_api,monkeypatch,tmp_path):
    from arch_explorer.adf import WorkloadSpec
    module,w,s=onnx_api;case=mlp_case();path=save_case(case,tmp_path/'files',external=True)
    with no_backend(monkeypatch):
        spec=module.import_onnx(path,inputs={'X':case.feeds['X'].tolist()},external_root=path.parent)
        original=deepcopy(spec.to_dict());expected=mlp(case.roots);source=json.loads(json.dumps(original))
        path.unlink();(path.parent/'weights/data.bin').unlink()
        saved=WorkloadSpec(**source);description=module.describe_onnx(saved)
        assert compare_original_values(w.execute_reference(saved),description,expected)
        assert saved.to_dict()==original
        for component in ('model','external'):
            bad=deepcopy(source)
            entry=(bad['params']['source']['model'] if component=='model'
                   else next(iter(bad['params']['source']['external'].values())))
            raw=bytearray(base64.b64decode(entry['base64']));raw[-1]^=1
            entry['base64']=base64.b64encode(raw).decode()
            with pytest.raises(ValueError,match='ONNX source digest/encoding mismatch'):
                w.workload_from_spec(WorkloadSpec(**bad))

def test_changed_real_model_node_changes_product_not_filename(onnx_api,monkeypatch,tmp_path):
    module,w,s=onnx_api;case=mlp_case();first=save_case(case,tmp_path/'first')
    changed=onnx.ModelProto();changed.CopyFrom(case.model)
    next(a for a in changed.graph.node[0].attribute if a.name=='alpha').f=.75
    other=tmp_path/'other';other.mkdir();path=other/'same-model-name.onnx';path.write_bytes(changed.SerializeToString())
    expected=evaluate_model(changed,case.feeds)['stages']
    with no_backend(monkeypatch):
        spec=module.import_onnx(path,inputs={'X':case.feeds['X'].tolist()})
        description=module.describe_onnx(spec)
        assert compare_original_values(w.execute_reference(spec),description,expected)
        previous=module.import_onnx(first,inputs={'X':case.feeds['X'].tolist()})
        assert previous.params['source']!=spec.params['source']
        a=module.describe_onnx(previous)['values']['Y']['tensor'];b=description['values']['Y']['tensor']
        assert not np.array_equal(w.execute_reference(previous)['tensors'][a],w.execute_reference(spec)['tensors'][b])
