"""Frozen input wiring through public SIM51/CORE33 APIs, no private lowering."""
from copy import deepcopy
import hashlib,json
from pathlib import Path
import numpy as np
import onnx
from onnx_numpy_interpreter import evaluate_model

ROOT=Path(__file__).resolve().parents[1]/'fixtures/onnx_import/sim51'
INPUT_SHA='622a0f5a3dff07a6504bcb33c8eea3aecbb37fb283a011c37e772a1a5b791e19'

def formal_input(key):
    raw=(ROOT/'INPUTS.json').read_bytes();assert hashlib.sha256(raw).hexdigest()==INPUT_SHA
    document=json.loads(raw);row=deepcopy(document['cases'][key])
    path=ROOT/row['model']
    for name,record in document['files'].items():
        value=(ROOT/name).read_bytes()
        assert len(value)==record['bytes'] and hashlib.sha256(value).hexdigest()==record['sha256']
    options=dict(inputs=row['inputs'],input_shapes=row['input_shapes'],
                 external_root=ROOT/row['external_root'] if row['external_root'] else None)
    model=onnx.load(str(path));feeds={k:np.asarray(v,np.float32) for k,v in row['inputs'].items()}
    return path,options,evaluate_model(model,feeds)

def formal_spec(key):
    from arch_explorer.workload.onnx_import import import_onnx
    path,options,expected=formal_input(key)
    return import_onnx(path,**options),expected

def scene(key,index,policy):
    from arch_explorer.archs.onnx_compilation import onnx_scene
    spec,expected=formal_spec(key)
    adf,same,mapping=onnx_scene(spec,candidate=index,layout=policy)
    assert same.to_dict()==spec.to_dict()
    return adf,spec,mapping,expected
