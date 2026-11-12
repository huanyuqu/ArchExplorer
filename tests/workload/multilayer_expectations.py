"""Independent VAL57 data and scalar layer chain; no production imports.

This describes expected dataflow, not the future public composition schema.
Only X and per-instance parameters are inputs. Intermediate layer answers are
computed here for comparison and must never become model input fixtures.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

from precision_transformer_expectations import (
    FORMATS, frozen_block_reference, store, tensor_bytes,
)

PARAMETER_NAMES=frozenset(('Wq','Wk','Wv','Wo','bq','bk','bv','bo',
    'norm1_gamma','norm1_beta','norm2_gamma','norm2_beta','W1','b1','W2','b2'))
INPUT_FIXTURE=Path(__file__).resolve().parents[1]/'fixtures/precision_transformer/INPUTS.json'
INPUT_SHA='f7538a04334fab9476e60f8e50ffece461bc2a1cf7caea9298e9cc7b8f150c06'


def leaves(value):
    if isinstance(value,list):
        for item in value:yield from leaves(item)
    else:yield float(value)


def transform(value,function):
    counter=iter(range(1000000))
    def visit(item):
        if isinstance(item,list):return [visit(child) for child in item]
        return function(float(item),next(counter))
    return visit(value)


def stack_inputs(layers=2,*,dtype='fp32',case='regular'):
    """Nontrivial, distinct weights; one external X, never later-layer X."""
    if type(layers) is not int or layers not in (2,4):raise ValueError('VAL57 requires two or four layers')
    if dtype not in FORMATS or case not in ('regular','boundary'):raise ValueError('unsupported test data')
    raw=INPUT_FIXTURE.read_bytes();assert hashlib.sha256(raw).hexdigest()==INPUT_SHA
    original=json.loads(raw)['events-fp32']['workload']['params']
    config={k:original[k] for k in ('batch','sequence','hidden','heads','mlp_hidden','epsilon')}
    x=deepcopy(original['inputs']['X'])
    if case=='boundary':
        precision={'fp32':24,'fp16':11,'bf16':8}[dtype]
        unit=2.**(1-precision)
        offsets=(unit/2,unit*5/8,unit*3/8,unit*3/2)
        x=transform(x,lambda value,i:(-1 if i%3==0 else 1)*(1+offsets[i%4]))
    instances=[]
    for index in range(layers):
        parameters={}
        for key in sorted(PARAMETER_NAMES):
            scale=1+index/8
            step=(index+1)/(128 if key.startswith('W') else 512)
            parameters[key]=transform(original['inputs'][key],
                lambda value,i:value*scale+step*((i%3)-1))
        instances.append(dict(id='unit-'+str(index),parameters=parameters))
    return dict(config=config,dtype=dtype,X=x,instances=instances)


def chain_reference(config,x,instances,*,dtype):
    """Reuse the independently validated single-block scalar equations.

    Each stage consumes the actual stored preceding output, including RNE at
    every internal materialization. There is no delayed rounding across layers.
    This function does not construct or run an ArchExplorer workload graph.
    """
    if dtype not in FORMATS or not instances:raise ValueError('explicit storage and instances required')
    expected={'batch','sequence','hidden','heads','mlp_hidden','epsilon'}
    if set(config)!=expected:raise ValueError('explicit block dimensions required')
    current=deepcopy(x);stages={};order=[]
    for instance in instances:
        if set(instance)!={'id','parameters'}:raise ValueError('instance requires id and parameters')
        identifier=instance['id'];parameters=instance['parameters']
        if not isinstance(identifier,str) or not identifier or identifier in stages:
            raise ValueError('unique explicit instance IDs required')
        if set(parameters)!=PARAMETER_NAMES:raise ValueError('parameters exclude X and every intermediate activation')
        params=dict(config,dtype=dtype,inputs=dict(deepcopy(parameters),X=current))
        stage=frozen_block_reference(params)
        stages[identifier]=stage;order.append(identifier)
        current=deepcopy(stage['output'])
    return dict(order=order,layers=stages,output=current)


def reference_for(data):
    return chain_reference(data['config'],data['X'],data['instances'],dtype=data['dtype'])


def parameter_storage(data):
    """Count materialized parameter scalars independently of host containers."""
    unit=4 if data['dtype']=='fp32' else 2
    return {row['id']:sum(sum(1 for _ in leaves(value))*unit
                         for value in row['parameters'].values()) for row in data['instances']}


def matmul_tiles(batch,m,k,n,tiles):
    """Independent rectangular output regions and explicit ascending K ranges."""
    tm,tk,tn=(tiles[key] for key in ('m','k','n'))
    if any(type(v) is not int or v<1 for v in (batch,m,k,n,tm,tk,tn)):
        raise ValueError('positive integer dimensions and tiles required')
    return [dict(batch=b,rows=(i,min(i+tm,m)),columns=(j,min(j+tn,n)),
                 k_ranges=[(q,min(q+tk,k)) for q in range(0,k,tk)])
            for b in range(batch) for i in range(0,m,tm) for j in range(0,n,tn)]
