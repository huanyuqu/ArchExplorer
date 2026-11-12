"""Import an actual export directory and emit ordinary model-independent inputs."""
import argparse
import json
from pathlib import Path


def emit(directory, output, *, max_evaluations=12, resource_budget=None,
         candidate=None, layout=None):
    from arch_explorer.archs.framework_export import import_export
    from arch_explorer.archs.onnx_compilation import onnx_scene, onnx_search_config, onnx_agent_config
    if type(max_evaluations) is not int or max_evaluations < 1:
        raise ValueError('max_evaluations must be a positive integer upper bound')
    workload = import_export(directory, resource_budget=resource_budget)
    selected_candidate = 0 if candidate is None else candidate
    selected_layout = 'materialized' if layout is None else layout
    adf, _, mapping = onnx_scene(workload, candidate=selected_candidate, layout=selected_layout)
    config = onnx_search_config(workload, candidate=candidate, layout=layout)
    config['max_evaluations'] = min(max_evaluations, config['max_evaluations'])
    agent = onnx_agent_config(workload, candidate=selected_candidate, layout=selected_layout)
    root = Path(output).resolve()
    root.mkdir(parents=True, exist_ok=False)
    for name, value in [('WORKLOAD', workload.to_dict()),
                        ('SCENE', {'adf': adf.to_dict(), 'workload': workload.to_dict(), 'mapping': mapping.to_dict()}),
                        ('CONFIG', config), ('AGENT-CONFIG', agent),
                        ('SCOPE', {'input': 'actual framework-exported ONNX',
                                   'graph_builder': 'registered onnx', 'hardware_calibrated': False,
                                   'max_evaluations_is_upper_bound': True,
                                   'search_candidate': candidate, 'search_layout': layout,
                                   'scene_and_agent_candidate': selected_candidate,
                                   'scene_and_agent_layout': selected_layout,
                                   'read_only': 'archexplorer evaluate BEST.json',
                                   'explicit_execution': 'archexplorer rerun BEST.json --work-dir NEW'})]:
        (root / f'{name}.json').write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('directory')
    parser.add_argument('--output', required=True)
    parser.add_argument('--max-evaluations', type=int, default=12)
    parser.add_argument('--resource-budget', help='Authoritative resource-budget JSON')
    parser.add_argument('--candidate', type=int, choices=(0, 1), help='Limit search to one placement')
    parser.add_argument('--layout', choices=('materialized', 'views'), help='Limit search to one layout')
    args = parser.parse_args()
    emit(args.directory, args.output, max_evaluations=args.max_evaluations,
         resource_budget=json.loads(Path(args.resource_budget).read_text()) if args.resource_budget else None,
         candidate=args.candidate, layout=args.layout)
