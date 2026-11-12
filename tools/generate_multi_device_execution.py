"""Generate explicit non-GPU systolic/vector multi-device fixtures (synthetic rates).

Usage: python tools/generate_multi_device_execution.py OUTPUT_DIRECTORY
Requires an installed ArchExplorer; also works from a source checkout.
See docs/non-gpu-generators.md for precision and execution boundaries.
"""
import json
from pathlib import Path

from arch_explorer.adf import WorkloadSpec
from arch_explorer.archs.adf_seeds import non_gpu_adf
from arch_explorer.adf.multi_device import validate_request
from arch_explorer.devices.topology import compose, link_resource
from arch_explorer.workload.llm import workload_from_spec

SOURCE = 'synthetic non-GPU multi-device execution fixture, not measured hardware'


def configuration(kind='vit', count=2, profile=None, tp=False):
    if type(count) is not int or count < 1:
        raise ValueError('count must be a positive integer')
    if type(tp) is not bool:
        raise ValueError('tp must be a boolean')
    if type(kind) is not str or kind not in ('vit', 'clip', 'transformer', 'operator'):
        raise ValueError('kind must be vit, clip, transformer or operator')
    if profile is not None and kind not in ('vit', 'clip'):
        raise ValueError('profile is only valid for vit or clip')
    names = [f'rank{i}' for i in range(count)]
    networks, edges = [], []
    for i in range(count-1):
        for src, dst in ((i, i+1), (i+1, i)):
            name = f'link{src}_{dst}'
            networks.append(link_resource(name, 25.0, 3.0, source=SOURCE))
            edges.append(dict(src='system/'+names[src], dst='system/'+names[dst], network='system/'+name))
    if kind in ('vit', 'clip'):
        wp = dict(profile=profile if profile is not None else ('vit_b16_224' if kind=='vit' else 'clip_vit_b32'), dtype='fp16', iterations=1)
    elif kind == 'transformer':
        wp = dict(model='llama2-7b', layers=1, batch=1, seq_len=8, dtype='fp16', iterations=1)
    else:
        wp = dict(op='matmul', batch=1, m=8, k=5, n=8, dtype='fp16', iterations=1)
    workload = WorkloadSpec(kind, wp)
    ir = workload_from_spec(workload)
    adf = compose('system', {name:non_gpu_adf(matrix_precision=ir.dtype) for name in names}, networks)
    params = dict(device_paths=['system/'+n for n in names], placements={}, shards=[], collectives=[],
                  topology_edges=edges, routing='shortest_hops')
    local = dict(template='gsm_tiled', params=dict(tile_m=64, tile_n=64, rows_per_cta=4))
    for i, op in enumerate(ir.operators):
        rank = min(count-1, i*count//len(ir.operators))
        params['placements'][op.name] = dict(ranks=[rank], mapping=local)
        is_tp = tp and (kind == 'operator' or op.name.endswith(('o_proj', 'ffn_down')))
        if is_tp:
            if op.op != 'matmul':
                raise ValueError('TP fixture requires a GEMM')
            ranks = list(range(count))
            params['placements'][op.name]['ranks'] = ranks
            k = op.attrs['k']
            if count > k:
                raise ValueError('TP rank count cannot exceed the GEMM K dimension')
            ranges = [dict(rank=r, start=k*r//count, stop=k*(r+1)//count) for r in ranks]
            for tensor, offset in zip(op.inputs, (1, 2)):
                params['shards'].append(dict(tensor=tensor, axis=len(ir.tensors[tensor].shape)-offset,
                                              size=k, ranges=ranges))
            params['collectives'].append(dict(after=op.name, tensor=op.outputs[0], operation='allreduce',
                algorithm='tree', ranks=ranks, dtype='fp32', elements=op.attrs['batch']*op.attrs['m']*op.attrs['n']))
    validate_request(adf, params)
    return dict(seed_adf=adf.to_dict(), workloads=[workload.to_dict()],
                mapping_templates=['multi_device'], mapping_selectors={'multi_device':params}, backend='events',
                max_evaluations=4, mapping_samples=1, parameter_samples=1, random_seed=17,
                area_budget_mm2=1000, cost_budget_usd=10000,
                parameter_domains=[dict(path='system/link0_1',param='bandwidth',values=[7.0,25.0,50.0],source=SOURCE)] if count>1 else [])


def main():
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output',type=Path)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    for name, config in [('vit-two',configuration()), ('clip-four',configuration('clip',4)),
                         ('matrix-tp',configuration('operator',2,tp=True)),
                         ('transformer-tp',configuration('transformer',4,tp=True))]:
        (args.output/(name+'.json')).write_text(json.dumps(config,indent=2)+'\n')


if __name__ == '__main__':
    main()
