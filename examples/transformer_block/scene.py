"""Save a complete synthetic encoder-block input for the public tools.

This command creates data/configuration only. Compilation, preparation,
evaluation, bounded search and replay remain separate public operations.
"""
import argparse
import json
from pathlib import Path


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True,help='New ADF/workload/mapping JSON file')
    parser.add_argument('--search-config',type=Path,help='Also save a bounded existing-search configuration')
    parser.add_argument('--agent-config',type=Path,help='Also save an offline scripted Agent configuration')
    parser.add_argument('--inputs',type=Path,help='Explicit complete numerical-input JSON; overrides the generated teaching values')
    parser.add_argument('--roofline',action='store_true',help='Use the existing synthetic Roofline hardware scene')
    parser.add_argument('--dtype',choices=('fp32','fp16','bf16'),default='fp32')
    parser.add_argument('--input-case',choices=('regular','boundary','constant'),default='regular')
    for name,default in (('batch',2),('seq',3),('hidden',8),('heads',2),('mlp',16)):
        parser.add_argument('--'+name,type=int,default=default)
    parser.add_argument('--epsilon',type=float,default=1e-5)
    parser.add_argument('--tile-m',type=int,default=2)
    parser.add_argument('--tile-k',type=int,default=3)
    parser.add_argument('--tile-n',type=int,default=5)
    args=parser.parse_args(argv)
    paths=[p for p in (args.output,args.search_config,args.agent_config) if p is not None]
    if len({p.resolve() for p in paths})!=len(paths) or any(p.exists() for p in paths):
        raise ValueError('Block source, search and Agent files require distinct new paths')
    inputs=json.loads(args.inputs.read_text()) if args.inputs else None
    from arch_explorer.archs.transformer import block_scene,block_search_config,block_agent_config
    from arch_explorer.archs.compilation import roofline_scene
    options=dict(dtype=args.dtype,batch=args.batch,seq=args.seq,hidden=args.hidden,
                 heads=args.heads,mlp=args.mlp,epsilon=args.epsilon,input_case=args.input_case,
                 inputs=inputs,hardware=roofline_scene if args.roofline else None,
                 tile_m=args.tile_m,tile_k=args.tile_k,tile_n=args.tile_n)
    adf,workload,mapping=block_scene(**options)
    documents=[(args.output,{'adf':adf.to_dict(),'workload':workload.to_dict(),'mapping':mapping.to_dict()})]
    if args.search_config:
        config=block_search_config(**options)
        config['trace_path']=str(args.search_config.resolve().with_suffix('.tools.jsonl'))
        documents.append((args.search_config,config))
    if args.agent_config:
        config=block_agent_config(**options)
        config['trace_path']=str(args.agent_config.resolve().with_suffix('.tools.jsonl'))
        documents.append((args.agent_config,config))
    for path,document in documents:
        path.parent.mkdir(parents=True,exist_ok=True)
        with path.open('x',encoding='utf-8') as stream:
            json.dump(document,stream,indent=2,sort_keys=True,allow_nan=False)
            stream.write('\n')
    return 0


if __name__=='__main__':
    try:
        raise SystemExit(main())
    except (ImportError,KeyError,OSError,RuntimeError,TypeError,ValueError) as error:
        raise SystemExit(f'transformer block scene: {error}') from None
