"""A finite structure/parameter/mapping search using the existing Env only."""
import argparse
import json
from pathlib import Path

from arch_explorer.adf import fingerprint
from arch_explorer.archs.adf_seeds import bus, mesh, non_gpu_adf
from arch_explorer.archs.llm_scenarios import llm_workload
from arch_explorer.structure import apply_edit, replay, share_private_memory


def emit(output):
    private = non_gpu_adf(num_cores=2, matrix_precision='fp16', dram_capacity_bytes=64 * 2**30)
    pooled = share_private_memory(private, 'chip', 2, 'sram', bus('group_bus', bandwidth=256.0),
                                 mesh('root_noc', cols=2, bandwidth=64.0))
    seed, edits = pooled.adf, list(pooled.records)
    # Move the pool into the root GSM level, then flatten the temporary group.
    # Each change still passes the existing immutable primitive validator.
    for edit in (
        {'op': 'create', 'container': 'chip', 'component': 'chip/memory_group_0/shared_sram'},
        {'op': 'remove', 'level': 'chip/memory_group_0'},
        {'op': 'delete', 'component': 'chip/root_noc'},
        {'op': 'delete', 'component': 'chip/group_bus'},
        {'op': 'create', 'container': 'chip', 'spec': bus('fabric', bandwidth=256.0, latency=1.0).to_dict()}):
        result = apply_edit(seed, edit)
        seed = result.adf
        edits.append(result.record)
    assert fingerprint(replay(private, edits)) == fingerprint(seed)
    workload = llm_workload(model='llama2-7b', phase='prefill', batch=1, seq_len=4,
                            layers=1, dtype='fp16', iterations=1)
    origin = 'explicit synthetic joint-search scenario'
    config = {'seed_adf': seed.to_dict(), 'workloads': [workload.to_dict()],
        'backend': 'events', 'backend_options': {'scheduler': 'consistent', 'systolic_model': 'native',
                                              'iterations': 1, 'trace': False, 'check_capacity': True},
        'mapping_templates': ['gsm_tiled', 'dmc_partition'], 'max_evaluations': 3,
        'parameter_samples': 1, 'mapping_samples': 1, 'random_seed': 7,
        'parameter_domains': [{'path': 'chip/dram', 'param': 'bandwidth',
                               'values': [64.0, 128.0], 'source': origin}],
        'search': {'max_steps': 2, 'max_proposals': 2, 'temperature': 1.0,
            'cooling': 0.95, 'max_relative_degradation': 0.0, 'proposals': [
                {'op': 'gsm_to_dmc', 'chip': 'chip', 'shared_memory': 'chip/shared_sram', 'parts': 2,
                 'noc_spec': mesh('noc', cols=3, bandwidth=64.0).to_dict(),
                 'bus_spec': bus('local_bus', bandwidth=128.0).to_dict()},
                {'tool': 'set_parameters', 'args': {'parameters': [
                    {'path': 'chip/dram', 'param': 'bandwidth', 'value': 128.0, 'source': origin}]}}]}}
    directory = Path(output).resolve()
    directory.mkdir(parents=True, exist_ok=False)
    config['trace_path'] = str(directory / 'tools.jsonl')
    (directory / 'CONFIG.json').write_text(json.dumps(config, indent=2) + '\n')
    (directory / 'ORIGIN.json').write_text(json.dumps({'private_seed': private.to_dict(),
        'gsm_seed': seed.to_dict(), 'primitive_records': edits,
        'scope': 'one model layer, two structural/parameter proposals, one formal mapping sample per attempt'}, indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    emit(parser.parse_args().output)
