"""Explicit v1 input migration. Historical results retain their old identity."""
from dataclasses import dataclass


@dataclass(frozen=True)
class CompilerTarget:
    id: str
    version: str
    lower: object


def migrate(document):
    from .compilation import check,compile_workload
    from .adapters import AdapterBinding
    from arch_explorer.adf.evaluation import MappingSpec,WorkloadSpec
    from arch_explorer.adf.model import from_dict
    check(document,{'schema','source','target','implementations','program','plan','components',
                    'lowered','sources','binding'},'arch-explorer.compilation.v1')
    source = document['source']
    mapping = MappingSpec(**source['mapping'])
    if mapping.template != 'compile_graph' or set(mapping.params) != {'compute','memory','tiles','threads_per_block'}:
        raise ValueError('unsupported historical compilation mapping')
    target = document['target']
    if target['version'] != '1' or target['id'] not in ('events','roofline','accelsim'):
        raise NotImplementedError('historical external targets require an explicit new adapter binding')
    params = dict(mapping.params)
    threads = params.pop('threads_per_block')
    options = {'threads_per_block':threads} if target['id'] == 'accelsim' else {}
    params['rules'] = {'matmul':dict(id='matmul.tiled',version='1',options=dict(tiles=params.pop('tiles')))}
    return compile_workload(from_dict(source['adf']),WorkloadSpec(**source['workload']),
        MappingSpec('compile_graph_v2',params), bindings={params['compute']:AdapterBinding(target['id'],'1',options)},
        implementations=document['implementations'])
