"""Independent standard ONNX test models; no ArchExplorer imports or answers."""
from dataclasses import dataclass
import hashlib,json
from pathlib import Path
import numpy as np

OPSET=17
IR_VERSION=8
@dataclass
class Case:
    name:str
    model:object
    feeds:dict
    roots:dict
    outputs:tuple

def _onnx():
    import onnx
    assert onnx.__version__=='1.16.2','freeze a new release before changing ONNX semantics'
    return onnx

def _model(name,nodes,inputs,outputs,initializers=()):
    o=_onnx()
    graph=o.helper.make_graph(nodes,name,inputs,outputs,list(initializers))
    model=o.helper.make_model(graph,producer_name='VAL62 independent synthetic inputs',
        opset_imports=[o.helper.make_opsetid('',OPSET)])
    model.ir_version=IR_VERSION
    return model

def _value(name,shape,dtype=None):
    o=_onnx();return o.helper.make_tensor_value_info(name,dtype or o.TensorProto.FLOAT,list(shape))

def _tensor(name,value):
    return _onnx().numpy_helper.from_array(np.asarray(value),name=name)

def mlp_case():
    o=_onnx();n=o.helper.make_node
    roots={
        'X':np.array([[1,-2,.5],[-.25,.75,2]],np.float32),
        'W1':np.array([[.5,-.25,.75],[-1,.5,.25],[.25,1,-.5],[.75,.25,1]],np.float32),
        'b1':np.array([.125,-.25,.375,.5],np.float32),
        'W2':np.array([[.5,-.25],[-.5,.75],[1,.25],[.125,-1]],np.float32),
        'b2':np.array([.25,-.5],np.float32)}
    nodes=[
        n('Gemm',['X','W1','b1'],['H'],name='',alpha=.5,beta=1.5,transB=1),
        n('Relu',['H'],['A0'],name='duplicate_display_name'),
        n('Constant',[],['shape'],name='',value=_tensor('shape_value',np.array([0,-1],np.int64))),
        n('Reshape',['A0','shape'],['A'],name='duplicate_display_name',allowzero=0),
        n('Gemm',['A','W2','b2'],['Z'],name='',alpha=1.25,beta=-.5),
        n('Identity',['Z'],['Y'],name='duplicate_display_name'),
        n('Add',['A','b1'],['shared'],name='duplicate_display_name')]
    model=_model('qa-mlp',nodes,[_value('X',(2,3))],
                 [_value('Y',(2,2)),_value('shared',(2,4))],
                 [_tensor(k,v) for k,v in roots.items() if k!='X'])
    return Case('mlp',model,{'X':roots['X'].copy()},roots,('Y','shared'))

def attention_case():
    o=_onnx();n=o.helper.make_node
    roots={
        'X':(np.arange(24,dtype=np.float32).reshape(2,3,4)-11)/16,
        'mask':np.array([[[0,-4,-8],[0,0,-4],[0,0,0]]],np.float32),
        'Wqk':np.array([[.5,-.25],[.25,.75],[-.5,.125],[1,-.5]],np.float32),
        'Wv':np.array([[.5,.25,-.5],[-.25,.75,.125],[1,-.5,.25],[.125,.25,.5]],np.float32),
        'kb':np.array([.125,-.25],np.float32),
        'scale':np.array([.5],np.float32)}
    nodes=[
        n('MatMul',['X','Wqk'],['Q'],name=''),
        n('MatMul',['X','Wqk'],['K0'],name='same'),
        n('Add',['K0','kb'],['K'],name='same'),
        n('MatMul',['X','Wv'],['V'],name='same'),
        n('Transpose',['K'],['KT'],name='',perm=[0,2,1]),
        n('MatMul',['Q','KT'],['scores'],name='same'),
        n('Mul',['scores','scale'],['scaled'],name='same'),
        n('Add',['scaled','mask'],['masked'],name='same'),
        n('Softmax',['masked'],['P'],name='same',axis=-1),
        n('MatMul',['P','V'],['C'],name='same'),
        n('Transpose',['C'],['T'],name='same',perm=[1,0,2]),
        n('Constant',[],['shape'],name='',value=_tensor('shape_value',np.array([3,6],np.int64))),
        n('Reshape',['T','shape'],['R'],name='same'),
        n('Identity',['R'],['Y'],name='same')]
    model=_model('qa-attention',nodes,[_value('X',(2,3,4)),_value('mask',(1,3,3))],
                 [_value('Y',(3,6)),_value('P',(2,3,3))],
                 [_tensor(k,v) for k,v in roots.items() if k not in ('X','mask')])
    return Case('attention',model,{k:roots[k].copy() for k in ('X','mask')},roots,('Y','P'))

def single_case(op,feeds,output_shape,*,attrs=None,initializers=None):
    """Real one-node models for release-level hand/boundary controls."""
    o=_onnx();initializers=initializers or {}
    arrays={k:np.asarray(v) for k,v in feeds.items()}
    arrays.update({k:np.asarray(v) for k,v in initializers.items()})
    values=[_value(k,v.shape,o.helper.np_dtype_to_tensor_dtype(v.dtype)) for k,v in arrays.items() if k in feeds]
    node=o.helper.make_node(op,list(arrays),['Y'],name='',**(attrs or {}))
    model=_model('qa-'+op,[node],values,[_value('Y',output_shape)],
        [_tensor(k,v) for k,v in initializers.items()])
    return Case(op,model,{k:arrays[k].copy() for k in feeds},arrays,('Y',))

def save_case(case,directory,*,external=False):
    """Write actual ModelProto bytes and an explicit nonzero-offset data file."""
    o=_onnx();directory=Path(directory);directory.mkdir(parents=True,exist_ok=False)
    model=o.ModelProto();model.CopyFrom(case.model)
    files={}
    if external:
        tensor=next(t for t in model.graph.initializer if t.name=='W1')
        raw=bytes(tensor.raw_data);assert raw
        relative='weights/data.bin';blob=b'VAL62-PREFIX-16!'+raw+b'END'
        assert len(b'VAL62-PREFIX-16!')==16
        path=directory/relative;path.parent.mkdir();path.write_bytes(blob)
        o.external_data_helper.set_external_data(tensor,relative,offset=16,length=len(raw))
        tensor.ClearField('raw_data')
        files[relative]=dict(sha256=hashlib.sha256(blob).hexdigest(),bytes=len(blob),offset=16,length=len(raw))
    path=directory/'model.onnx';path.write_bytes(model.SerializeToString(deterministic=True))
    # The public checker receives the real filename so external paths resolve.
    o.checker.check_model(str(path),full_check=True)
    feeds={name:dict(dtype=str(v.dtype),shape=list(v.shape),values=v.tolist()) for name,v in case.feeds.items()}
    manifest=dict(schema='arch-explorer.val62.independent-input.v1',case=case.name,opset=OPSET,IR=IR_VERSION,
        model=dict(path='model.onnx',sha256=hashlib.sha256(path.read_bytes()).hexdigest(),bytes=path.stat().st_size),
        external_data=files,feeds=feeds,outputs=list(case.outputs),
        initializers=[dict(name=t.name,dtype=int(t.data_type),shape=list(t.dims)) for t in model.graph.initializer],
        contains_expected_answers=False)
    (directory/'INPUTS.json').write_text(json.dumps(manifest,indent=2)+'\n')
    return path

def feed_values(manifest):
    return {name:np.asarray(row['values'],dtype=row['dtype']).reshape(row['shape']) for name,row in manifest['feeds'].items()}
