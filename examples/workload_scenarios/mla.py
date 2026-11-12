"""Emit public CLI/API/Agent inputs for the bounded absorbed MLA decode."""
import argparse
import json
from pathlib import Path

from arch_explorer.archs.mla import mla_agent_config, mla_scene, mla_search_config


def emit(output, *, shared, multicast):
    adf, spec, mapping = mla_scene(shared=shared, multicast=multicast)
    config = mla_search_config(shared=shared, multicast=multicast)
    agent = mla_agent_config(shared=shared, multicast=multicast, candidate=0, layout='materialized')
    directory = Path(output).resolve()
    directory.mkdir(parents=True, exist_ok=False)
    config['trace_path'] = str(directory / 'search-tools.jsonl')
    agent['trace_path'] = str(directory / 'agent-tools.jsonl')
    for name, value in (
        ('SCENE.json', {'adf': adf.to_dict(), 'workload': spec.to_dict(), 'mapping': mapping.to_dict()}),
        ('CONFIG.json', config), ('AGENT-CONFIG.json', agent),
        ('SCOPE.json', {'workload': 'absorbed-MLA-decode', 'precision': 'fp32',
                        'cache': 'explicit compressed cache including current token; read only',
                        'whole_model': False, 'hardware_calibration': False,
                        'shared': shared, 'multicast': multicast, 'backend': 'external Events',
                        'candidates': 'two placements/tiles and two public layouts',
                        'max_evaluations': config['max_evaluations']})):
        (directory / name).write_text(json.dumps(value, indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--sharing', choices=('grouped', 'private'), required=True)
    parser.add_argument('--multicast', choices=('on', 'off'), required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    emit(args.output, shared=args.sharing == 'grouped', multicast=args.multicast == 'on')
