"""Write explicit CLIP roots and ordinary public compilation/search configurations."""
import argparse
import json
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--search-config', type=Path)
    parser.add_argument('--agent-config', type=Path)
    parser.add_argument('--inputs', type=Path, help='Complete root JSON, preserving integer token IDs')
    parser.add_argument('--layers', type=int, choices=(2, 3, 4))
    parser.add_argument('--text-layers', type=int, choices=(2, 3, 4))
    parser.add_argument('--dtype', choices=('fp32', 'fp16', 'bf16'), default='fp32')
    parser.add_argument('--input-case', choices=('regular', 'boundary', 'constant'), default='regular')
    parser.add_argument('--layout', choices=('materialized', 'views'), default='materialized')
    parser.add_argument('--candidate', type=int, choices=(0, 1), default=0)
    parser.add_argument('--roofline', action='store_true')
    parser.add_argument('--single-layout-search', action='store_true')
    for name in ('image-batch', 'text-batch', 'image-size', 'patch', 'width', 'heads', 'mlp',
                 'context', 'vocab', 'text-width', 'text-heads', 'projection'):
        parser.add_argument('--' + name, type=int)
    args = parser.parse_args(argv)
    paths = [path for path in (args.output, args.search_config, args.agent_config) if path is not None]
    if len({path.resolve() for path in paths}) != len(paths) or any(path.exists() for path in paths):
        raise ValueError('Source, search and Agent documents require distinct new paths')
    from arch_explorer.archs.clip_compilation import clip_scene, clip_search_config, clip_agent_config
    from arch_explorer.archs.compilation import roofline_scene
    options = dict(dtype=args.dtype, input_case=args.input_case,
        inputs=json.loads(args.inputs.read_text()) if args.inputs else None,
        hardware=roofline_scene if args.roofline else None,
        candidate=args.candidate, layout=args.layout)
    for name in ('layers', 'text_layers', 'image_batch', 'text_batch', 'image_size', 'patch',
                 'width', 'heads', 'mlp', 'context', 'vocab', 'text_width', 'text_heads', 'projection'):
        value = getattr(args, name)
        if value is not None:
            options[name] = value
    adf, workload, mapping = clip_scene(**options)
    documents = [(args.output, {'adf': adf.to_dict(), 'workload': workload.to_dict(), 'mapping': mapping.to_dict()})]
    for path, make in ((args.search_config, clip_search_config), (args.agent_config, clip_agent_config)):
        if path is not None:
            extra = {'compare_layouts': not args.single_layout_search} if path == args.search_config else {}
            config = make(**options, **extra)
            config['trace_path'] = str(path.resolve().with_suffix('.tools.jsonl'))
            documents.append((path, config))
    encoded = [(path, json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')
               for path, value in documents]
    for path, content in encoded:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('x', encoding='utf-8') as stream:
            stream.write(content)
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (ImportError, KeyError, OSError, RuntimeError, TypeError, ValueError) as error:
        raise SystemExit(f'CLIP compilation scene: {error}') from None
