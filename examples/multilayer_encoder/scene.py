"""Write explicit multi-layer source and configurations for the public tools."""
import argparse
import json
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--search-config', type=Path)
    parser.add_argument('--agent-config', type=Path)
    parser.add_argument('--inputs', type=Path, help='JSON with one X and a list of layer parameter dictionaries')
    parser.add_argument('--layers', type=int, choices=(2, 3, 4), default=2)
    parser.add_argument('--dtype', choices=('fp32', 'fp16', 'bf16'), default='fp32')
    parser.add_argument('--input-case', choices=('regular', 'boundary', 'constant'), default='regular')
    parser.add_argument('--roofline', action='store_true')
    parser.add_argument('--layout', choices=('materialized', 'views'), default='materialized',
                        help='Public mapping policy for the same workload; unsupported sharing remains explicit materialization')
    parser.add_argument('--single-layout-search', action='store_true',
                        help='Limit search to --layout instead of comparing both public layout candidates')
    parser.add_argument('--candidate', type=int, choices=(0, 1), default=0,
                        help='Select one of the two explicit per-instance tile and placement configurations')
    for name, default in (('batch', 2), ('seq', 3), ('hidden', 8), ('heads', 2), ('mlp', 16)):
        parser.add_argument('--' + name, type=int, default=default)
    parser.add_argument('--epsilon', type=float, default=1e-5)
    args = parser.parse_args(argv)
    paths = [p for p in (args.output, args.search_config, args.agent_config) if p is not None]
    if len({p.resolve() for p in paths}) != len(paths) or any(p.exists() for p in paths):
        raise ValueError('Source, search and Agent documents require distinct new paths')
    from arch_explorer.archs.multilayer import encoder_scene, encoder_search_config, encoder_agent_config
    from arch_explorer.archs.compilation import roofline_scene
    options = dict(layers=args.layers, dtype=args.dtype, batch=args.batch, seq=args.seq,
                   hidden=args.hidden, heads=args.heads, mlp=args.mlp, epsilon=args.epsilon,
                   input_case=args.input_case,
                   inputs=json.loads(args.inputs.read_text()) if args.inputs else None,
                   hardware=roofline_scene if args.roofline else None, candidate=args.candidate,
                   layout=args.layout)
    adf, workload, mapping = encoder_scene(**options)
    documents = [(args.output, {'adf': adf.to_dict(), 'workload': workload.to_dict(),
                                'mapping': mapping.to_dict()})]
    for path, make in ((args.search_config, encoder_search_config),
                       (args.agent_config, encoder_agent_config)):
        if path is not None:
            extra = {'compare_layouts': not args.single_layout_search} if path == args.search_config else {}
            config = make(**options, **extra)
            config['trace_path'] = str(path.resolve().with_suffix('.tools.jsonl'))
            documents.append((path, config))
    for path, document in documents:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('x', encoding='utf-8') as stream:
            json.dump(document, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write('\n')
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (ImportError, KeyError, OSError, RuntimeError, TypeError, ValueError) as error:
        raise SystemExit(f'multi-layer encoder scene: {error}') from None
