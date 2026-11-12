"""Emit explicit existing-Env inputs or prepare one source-bound events receipt."""
import argparse
from copy import deepcopy
import json
from pathlib import Path

from arch_explorer.adf import MappingSpec, WorkloadSpec, from_dict
from arch_explorer.archs.adf_seeds import non_gpu_adf
from arch_explorer.archs.llm_scenarios import llm_scene, llm_search_config, llm_workload


def emit(input_path, output):
    data = json.loads(Path(input_path).read_text())
    expected = {'schema', 'hardware', 'workloads', 'mapping', 'backend', 'backend_options', 'search'}
    if set(data) != expected or data['schema'] != 'archexplorer.llm-scenario.v1':
        raise ValueError('Complete llm-scenario.v1 input fields required')
    hardware = data['hardware']
    if (set(hardware) != {'factory', 'parameters'} or hardware['factory'] != 'non_gpu_adf'
            or set(hardware['parameters']) != {'num_cores', 'matrix_precision', 'dram_capacity_bytes'}):
        raise ValueError('The example requires explicit non_gpu_adf parameters')
    adf = non_gpu_adf(**hardware['parameters'])
    workloads = [llm_workload(**row) for row in data['workloads']]
    mapping = data['mapping']
    if set(mapping) != {'template', 'requests'}:
        raise ValueError('Explicit mapping template and request list required')
    search = data['search']
    if set(search) != {'max_evaluations', 'parameter_samples', 'mapping_samples', 'random_seed', 'parameter_domains', 'settings'}:
        raise ValueError('Complete finite search settings required')
    config = llm_search_config(adf=adf, workloads=workloads, backend=data['backend'],
        backend_options=data['backend_options'], mapping_template=mapping['template'], mapping_requests=mapping['requests'],
        parameter_domains=search['parameter_domains'], max_evaluations=search['max_evaluations'],
        parameter_samples=search['parameter_samples'], mapping_samples=search['mapping_samples'],
        random_seed=search['random_seed'], search=search['settings'])
    # Each scene retains a concrete mapping. The Env selects among the checked
    # request union; no extra search implementation lives in this example.
    scenes = []
    for workload in workloads:
        for request in mapping['requests']:
            _, _, selected = llm_scene(adf=adf, workload=workload,
                                      mapping=MappingSpec(mapping['template'], request))
            scenes.append({'adf': adf.to_dict(), 'workload': workload.to_dict(), 'mapping': selected.to_dict(),
                           'provider': data['backend'], 'options': data['backend_options']})
    directory = Path(output).resolve()
    directory.mkdir(parents=True, exist_ok=False)
    config['trace_path'] = str(directory / 'tools.jsonl')
    (directory / 'CONFIG.json').write_text(json.dumps(config, indent=2) + '\n')
    agent = deepcopy(config)
    agent['max_evaluations'], agent['mapping_samples'] = len(workloads), 1
    agent['trace_path'] = str(directory / 'agent-tools.jsonl')
    agent['agent'] = {'max_steps': 2, 'max_tool_calls': 2, 'script': [
        {'tool': 'evaluate', 'args': {'optimize': 'none', 'mappings': [
            MappingSpec(mapping['template'], mapping['requests'][0]).to_dict() for _ in workloads]}},
        {'tool': 'submit', 'args': {}}]}
    (directory / 'AGENT-CONFIG.json').write_text(json.dumps(agent, indent=2) + '\n')
    (directory / 'SCENES.json').write_text(json.dumps(scenes, indent=2) + '\n')
    (directory / 'SCOPE.json').write_text(json.dumps({
        'kind': 'layer-level-performance-graph', 'weights_present': False,
        'full_model_numeric_validation': False, 'hardware_calibrated': False,
        'legacy_search_replay': 're-evaluates', 'events_receipt_replay': 'read-only-original-receipt',
        'selected_workloads': [w.to_dict() for w in workloads]}, indent=2) + '\n')


def prepare(scene_path, index, output):
    from arch_explorer.simulator.api import prepare_request
    scenes = json.loads(Path(scene_path).read_text())
    if type(index) is not int or not 0 <= index < len(scenes):
        raise ValueError('scene index is outside the emitted scene list')
    scene = scenes[index]
    directory = Path(output).resolve()
    directory.mkdir(parents=True, exist_ok=False)
    request = prepare_request(from_dict(scene['adf']), WorkloadSpec(**scene['workload']),
        MappingSpec(**scene['mapping']), output_dir=str(directory / 'provider'),
        provider=scene['provider'], options=scene['options'])
    (directory / 'REQUEST.json').write_text(json.dumps(request, indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest='command', required=True)
    generate = commands.add_parser('emit')
    generate.add_argument('--input', required=True)
    generate.add_argument('--output', required=True)
    bind = commands.add_parser('prepare')
    bind.add_argument('--scenes', required=True)
    bind.add_argument('--index', type=int, required=True)
    bind.add_argument('--output', required=True)
    args = parser.parse_args()
    if args.command == 'emit':
        emit(args.input, args.output)
    else:
        prepare(args.scenes, args.index, args.output)
