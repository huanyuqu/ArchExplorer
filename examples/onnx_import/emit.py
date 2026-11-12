"""Emit ordinary public compiler/search/Agent inputs from one actual ONNX file."""
import argparse
import json
from pathlib import Path


def emit(model, inputs, output, *, input_shapes=None, external_root=None, name='onnx'):
    from arch_explorer.archs.onnx_compilation import (
        onnx_workload, onnx_scene, onnx_search_config, onnx_agent_config)
    # Import once. Every downstream user path carries exactly this source bundle.
    spec = onnx_workload(model, inputs=inputs, input_shapes=input_shapes, external_root=external_root, name=name)
    adf, spec, mapping = onnx_scene(spec)
    config, agent = onnx_search_config(spec), onnx_agent_config(spec)
    directory = Path(output).resolve()
    directory.mkdir(parents=True, exist_ok=False)
    config['trace_path'] = str(directory / 'search-tools.jsonl')
    agent['trace_path'] = str(directory / 'agent-tools.jsonl')
    for name, value in (
        ('WORKLOAD.json', spec.to_dict()),
        ('SCENE.json', {'adf': adf.to_dict(), 'workload': spec.to_dict(), 'mapping': mapping.to_dict()}),
        ('CONFIG.json', config), ('AGENT-CONFIG.json', agent),
        ('SCOPE.json', {'frontend': 'standard ONNX file', 'precision': 'fp32',
                        'backend': 'external Events', 'hardware_calibration': False,
                        'candidates': 'two placements/tiles by original Graph operator IDs, two public layouts',
                        'max_evaluations': config['max_evaluations'],
                        'root_values': 'explicit caller-supplied JSON; no default runtime inputs'})):
        (directory / name).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    return directory


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model', type=Path)
    parser.add_argument('--inputs', type=Path, required=True)
    parser.add_argument('--input-shapes', type=Path)
    parser.add_argument('--external-root', type=Path)
    parser.add_argument('--name', default='onnx')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    emit(args.model, json.loads(args.inputs.read_text()), args.output,
         input_shapes=json.loads(args.input_shapes.read_text()) if args.input_shapes else None,
         external_root=args.external_root, name=args.name)
