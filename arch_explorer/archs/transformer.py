"""Explicit encoder-block scene data and clients of the public compilation API.

Operator semantics, graph construction, precision boundaries and mapping rules
belong to workload.api. This module supplies inputs and ordinary configurations
for the existing compiler, search and Agent tools.
"""
from copy import deepcopy
import math


def _values(shape, *, offset=0, scale=1.0, shift=0.0, pattern='regular'):
    """Generate numbers only; public precision semantics round them later."""
    if math.prod(shape)>2**20:
        raise ValueError('Teaching input exceeds 2**20 declared elements per tensor')
    boundary=(0.0,-0.0,2.0**-24,-2.0**-24,
              1+2.0**-11-2.0**-22,1+2.0**-11,1+2.0**-11+2.0**-22,
              1+2.0**-8-2.0**-17,1+2.0**-8,1+2.0**-8+2.0**-17,
              -1-2.0**-11,-1-2.0**-8,-3.0,4.0,-0.25,0.5)
    cursor=offset
    def visit(axis):
        nonlocal cursor
        if axis<len(shape):return [visit(axis+1) for _ in range(shape[axis])]
        index=cursor;cursor+=1
        value=(0.25 if pattern=='constant' else boundary[index%len(boundary)]
               if pattern=='boundary' else ((index*7)%23-11)/16.0)
        scaled=value*scale
        return scaled if shift==0.0 else scaled+shift
    return visit(0)


def block_inputs(*, batch=2, seq=3, hidden=8, heads=2, mlp=16, input_case='regular'):
    """Explicit synthetic X, weights, biases and two norm parameter sets.

    Public input-shape validation is authoritative. The boundary case includes
    values just below/at/above FP16 and BF16 ties; constant has identical X
    values within every row. Weights/norm parameters stay finite and nontrivial.
    No normalization, projection, rounding or output answer is computed here.
    """
    from arch_explorer.workload.api import transformer_block_input_shapes
    if input_case not in ('regular','boundary','constant'):
        raise ValueError('input_case must be regular, boundary or constant')
    shapes=transformer_block_input_shapes(batch=batch,sequence=seq,hidden=hidden,
                                          heads=heads,mlp_hidden=mlp)
    result={}
    for index,(name,shape) in enumerate(sorted(shapes.items())):
        if name=='X':
            result[name]=_values(shape,pattern=input_case)
        elif name in ('norm1_gamma','norm2_gamma'):
            result[name]=_values(shape,offset=index*5,scale=0.125,shift=1.0)
        elif name in ('norm1_beta','norm2_beta','bq','bk','bv','bo','b1','b2'):
            result[name]=_values(shape,offset=index*5,scale=0.0625)
        else:
            result[name]=_values(shape,offset=index*5,scale=0.25)
    return result


def block_workload(*, dtype='fp32', batch=2, seq=3, hidden=8, heads=2, mlp=16,
                   epsilon=1e-5, input_case='regular', inputs=None):
    """Call the public pre-LN builder with complete, unrounded numerical data."""
    from arch_explorer.workload.api import transformer_encoder_block
    values=block_inputs(batch=batch,seq=seq,hidden=hidden,heads=heads,mlp=mlp,
                        input_case=input_case) if inputs is None else deepcopy(inputs)
    return transformer_encoder_block(inputs=values,batch=batch,sequence=seq,hidden=hidden,
        heads=heads,mlp_hidden=mlp,dtype=dtype,epsilon=epsilon)


def block_scene(*, dtype='fp32', batch=2, seq=3, hidden=8, heads=2, mlp=16,
                epsilon=1e-5, input_case='regular', inputs=None, hardware=None,
                tile_m=2, tile_k=3, tile_n=5):
    """Return standard ADF, WorkloadSpec and MappingSpec for a full block.

    Default Events/Roofline boards retain hardware precision=fp32: they declare
    FP32 arithmetic service for each storage dtype and explicit conversions,
    with no invented low-precision acceleration. Public adapters check support.
    """
    from .compilation import events_scene,compile_mapping
    adf,_,placement=(events_scene if hardware is None else hardware)()
    workload=block_workload(dtype=dtype,batch=batch,seq=seq,hidden=hidden,heads=heads,
        mlp=mlp,epsilon=epsilon,input_case=input_case,inputs=inputs)
    mapping=compile_mapping(compute=placement.params['compute'],memory=placement.params['memory'],
        tile_m=tile_m,tile_k=tile_k,tile_n=tile_n)
    return adf,workload,mapping


def _configuration(adf, workload, first, second):
    source="bounded encoder block example; explicit synthetic hardware proposal"
    return {'seed_adf':adf.to_dict(),'workloads':[workload.to_dict()],
            'compilation':{},'backend_options':{'trace':True},
            'mapping_templates':[first.template],
            'mapping_selectors':{first.template:[deepcopy(first.params),deepcopy(second.params)]},
            'max_evaluations':6,'parameter_samples':1,'mapping_samples':2,
            'parameter_domains':[
                {'path':'board/sram','param':'capacity','values':[1,65536],'source':source},
                {'path':'board/sram','param':'bandwidth','values':[16.0,32.0],'source':source}],
            'random_seed':0,'search':{'max_steps':3,'max_proposals':3,'proposals':[
                {'tool':'set_parameters','args':{'parameters':[{
                    'path':'board/sram','param':key,'value':value,'source':source}]}}
                for key,value in (('capacity',1),('bandwidth',32.0))]}}


def block_search_config(*, dtype='fp32', batch=2, seq=3, hidden=8, heads=2, mlp=16,
                        epsilon=1e-5, input_case='regular', inputs=None, hardware=None,
                        tile_m=2, tile_k=3, tile_n=5):
    """Six evaluation units with two mappings and public parameter proposals.

The existing search engine controls the policy, budget, candidate validation,
compilation, preparation and evaluation. No block-specific search path is used.
"""
    from .compilation import compile_mapping
    adf,workload,first=block_scene(dtype=dtype,batch=batch,seq=seq,hidden=hidden,
        heads=heads,mlp=mlp,epsilon=epsilon,input_case=input_case,inputs=inputs,
        hardware=hardware,tile_m=tile_m,tile_k=tile_k,tile_n=tile_n)
    other=(3,5,3) if (tile_m,tile_k,tile_n)!=(3,5,3) else (2,3,5)
    second=compile_mapping(compute=first.params['compute'],memory=first.params['memory'],
        tile_m=other[0],tile_k=other[1],tile_n=other[2])
    return _configuration(adf,workload,first,second)


def block_agent_config(*, dtype='fp32', batch=2, seq=3, hidden=8, heads=2, mlp=16,
                       epsilon=1e-5, input_case='regular', inputs=None, hardware=None,
                       tile_m=2, tile_k=3, tile_n=5):
    """Offline Agent that discovers rules, executes one mapping and submits."""
    config=block_search_config(dtype=dtype,batch=batch,seq=seq,hidden=hidden,
        heads=heads,mlp=mlp,epsilon=epsilon,input_case=input_case,inputs=inputs,
        hardware=hardware,tile_m=tile_m,tile_k=tile_k,tile_n=tile_n)
    mapping={'template':config['mapping_templates'][0],
             'params':deepcopy(next(iter(config['mapping_selectors'].values()))[0])}
    config.update(max_evaluations=1,agent={'max_steps':3,'max_tool_calls':3,'script':[
        {'tool':'operator_mappings','args':{}},
        {'tool':'evaluate','args':{'optimize':'none','mappings':[mapping]}},
        {'tool':'submit','args':{}}]})
    config.pop('search')
    return config
