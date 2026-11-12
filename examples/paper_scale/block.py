"""Emit a source-complete block scene using only public graph/mapping functions."""
import argparse
import json
from pathlib import Path


def main(argv=None):
    from arch_explorer.adf import from_dict
    from arch_explorer.archs.block_compilation import block_scene, block_hardware_scene
    parser = argparse.ArgumentParser()
    parser.add_argument('--parameters', required=True, help='Complete mla_stream kwargs, with roots and any resource_budget')
    hardware_source = parser.add_mutually_exclusive_group(required=True)
    hardware_source.add_argument('--adf', help='Actual caller-supplied ADF JSON')
    hardware_source.add_argument('--hardware', help='Explicit block_memory_hardware keyword JSON; use its returned paths')
    parser.add_argument('--compute', nargs='+', help='Actual compute paths in slot order for --adf')
    parser.add_argument('--memories', nargs='+', help='Actual memory paths in slot order for --adf')
    parser.add_argument('--external-memory', help='Actual external path required by external cache mode')
    parser.add_argument('--output', required=True, help='New scene directory')
    args = parser.parse_args(argv)
    if args.adf is not None and (args.compute is None or args.memories is None):
        parser.error('--adf requires --compute and --memories')
    if args.hardware is not None and any(value is not None for value in (args.compute, args.memories, args.external_memory)):
        parser.error('--hardware uses the factory slot metadata; explicit path overrides require --adf')
    root = Path(args.output).resolve()
    root.mkdir(parents=True, exist_ok=False)
    record = {'schema': 'arch-explorer.block-scene.v1', 'status': 'started',
        'parameters_file': args.parameters, 'adf_file': args.adf, 'hardware_options_file': args.hardware,
        'compute': args.compute, 'memories': args.memories, 'external_memory': args.external_memory,
        'construction_attempted': True, 'scene_emitted': False, 'component_execution': False,
        'hardware_accuracy_established': False}
    try:
        parameters = json.loads(Path(args.parameters).read_text())
        if args.hardware is not None:
            options = json.loads(Path(args.hardware).read_text())
            record['hardware_options'] = options
            adf, spec, mapping, metadata = block_hardware_scene(parameters, hardware=options)
            (root / 'HARDWARE.json').write_text(json.dumps(metadata, indent=2, allow_nan=False) + '\n')
            record.update(compute=metadata['compute'], memories=metadata['memories'],
                external_memory=metadata['external_memory'], hardware_candidate=metadata['candidate'])
        else:
            adf = from_dict(json.loads(Path(args.adf).read_text()))
            adf, spec, mapping = block_scene(parameters, adf=adf, compute=args.compute,
                memories=args.memories, external_memory=args.external_memory)
        for name, value in [('WORKLOAD', spec.to_dict()), ('MAPPING', mapping.to_dict()),
                            ('SCENE', {'adf': adf.to_dict(), 'workload': spec.to_dict(), 'mapping': mapping.to_dict()})]:
            (root / (name + '.json')).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
        record.update(status='scene-emitted', scene_emitted=True,
            graph_builder='arch_explorer.workload.api.mla_stream',
            mapping_builder='arch_explorer.workload.api.block_mapping')
    except BaseException as exc:
        record.update(status='failed', error={'type': type(exc).__name__, 'message': str(exc)})
        raise
    finally:
        (root / 'RESULT.json').write_text(json.dumps(record, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'status': record['status'], 'output': str(root)}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
