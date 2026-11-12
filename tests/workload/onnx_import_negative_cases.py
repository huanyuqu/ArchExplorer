"""Input mutations for exact product refusals once SIM51 freezes diagnostics.

No product invocation or simulated refusal result is provided by this module.
"""
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from onnx_import_cases import mlp_case,single_case,save_case,_onnx

@dataclass
class RefusalInput:
    path:Path
    feeds:dict
    condition:str
    standard_model_legal:bool

MODEL_CONDITIONS=('opset18','IR9','Sin','unknown_attribute','symbolic_batch','conflicting_shape',
                  'fp16','fp64','missing_root','nonfinite_root','overridable_initializer')

def model_mutation(directory,condition):
    if condition not in MODEL_CONDITIONS:raise ValueError('unknown QA condition')
    o=_onnx();case=mlp_case()
    if condition=='opset18':case.model.opset_import[0].version=18
    elif condition=='IR9':case.model.ir_version=9
    elif condition=='Sin':
        case.model.graph.node[1].op_type='Sin'
    elif condition=='unknown_attribute':
        case.model.graph.node[1].attribute.append(o.helper.make_attribute('alpha',.25))
    elif condition=='symbolic_batch':
        axis=case.model.graph.input[0].type.tensor_type.shape.dim[0]
        axis.ClearField('dim_value');axis.dim_param='batch'
    elif condition=='conflicting_shape':
        case.model.graph.input[0].type.tensor_type.shape.dim[1].dim_value=5
    elif condition in ('fp16','fp64'):
        dtype=np.float16 if condition=='fp16' else np.float64
        proto=o.TensorProto.FLOAT16 if condition=='fp16' else o.TensorProto.DOUBLE
        for value in list(case.model.graph.input)+list(case.model.graph.output):
            value.type.tensor_type.elem_type=proto
        for tensor in case.model.graph.initializer:
            value=o.numpy_helper.to_array(tensor).astype(dtype)
            tensor.CopyFrom(o.numpy_helper.from_array(value,tensor.name))
        case.feeds={name:value.astype(dtype) for name,value in case.feeds.items()}
    elif condition=='missing_root':case.feeds={}
    elif condition=='nonfinite_root':case.feeds['X'][0,0]=np.nan
    elif condition=='overridable_initializer':
        case.model.graph.input.append(o.helper.make_tensor_value_info('W2',o.TensorProto.FLOAT,[4,2]))
        case.feeds['W2']=np.zeros((4,2),np.float32)
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=False)
    path=directory/'model.onnx';path.write_bytes(case.model.SerializeToString(deterministic=True))
    return RefusalInput(path,deepcopy(case.feeds),condition,
                        condition not in ('unknown_attribute','conflicting_shape'))

EXTERNAL_CONDITIONS=('missing','truncated','offset_beyond_file','length_short','negative_offset',
                     'escape','absolute_location','symlink_escape')

def external_mutation(directory,condition):
    if condition not in EXTERNAL_CONDITIONS:raise ValueError('unknown external condition')
    o=_onnx();case=mlp_case();path=save_case(case,directory,external=True)
    model=o.load(str(path),load_external_data=False);tensor=next(t for t in model.graph.initializer if t.name=='W1')
    data=path.parent/'weights/data.bin'
    def field(key,value):
        row=next(v for v in tensor.external_data if v.key==key);row.value=str(value)
    if condition=='missing':data.unlink()
    elif condition=='truncated':data.write_bytes(data.read_bytes()[:-12])
    elif condition=='offset_beyond_file':field('offset',data.stat().st_size+1)
    elif condition=='length_short':field('length',case.roots['W1'].nbytes-4)
    elif condition=='negative_offset':field('offset',-1)
    else:
        # All escaped bytes are generated test data, never host/private files.
        outside=path.parent.parent/(path.parent.name+'-outside.bin')
        outside.write_bytes(data.read_bytes())
        if condition=='escape':field('location','../'+outside.name)
        elif condition=='absolute_location':field('location',str(outside.resolve()))
        else:
            data.unlink();data.symlink_to(outside.resolve())
    path.write_bytes(model.SerializeToString(deterministic=True))
    return RefusalInput(path,case.feeds,condition,False)
