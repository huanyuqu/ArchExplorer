"""Emit private/grouped SRAM trees and their real primitive replay records."""
import argparse
import json
from pathlib import Path

from arch_explorer.adf import fingerprint
from arch_explorer.archs.adf_seeds import bus, mesh, non_gpu_adf
from arch_explorer.structure import replay, share_private_memory


def generate(output):
    seed = non_gpu_adf(num_cores=4, matrix_precision='fp16', dram_capacity_bytes=64 * 2**30)
    grouped = share_private_memory(seed, 'chip', 2, 'sram',
        bus('group_bus', bandwidth=256.0, latency=1.0),
        mesh('root_noc', cols=3, bandwidth=64.0, latency=1.0))
    assert fingerprint(replay(seed, grouped.records)) == fingerprint(grouped.adf)
    directory = Path(output).resolve()
    directory.mkdir(parents=True, exist_ok=False)
    for name, value in (('PRIVATE.json', seed.to_dict()), ('GROUPED.json', grouped.adf.to_dict()),
                        ('EDITS.json', grouped.records)):
        (directory / name).write_text(json.dumps(value, indent=2) + '\n')
    (directory / 'SCOPE.json').write_text(json.dumps({
        'capability': 'structure-and-primitive-replay', 'component_execution': False,
        'multicast_execution': False, 'hardware_calibration': False,
        'root_network': 'chip/root_noc',
        'group_memories': ['chip/memory_group_0/shared_sram', 'chip/memory_group_1/shared_sram']}, indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    generate(parser.parse_args().output)
