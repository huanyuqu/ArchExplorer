# Concrete Accel-Sim implementation; owned by the component provider.
"""Generate a tiny, explicitly synthetic native integration fixture, without CUDA.

The text implements load -> dependent integer add -> store -> exit in one
warp. It is a trace-format example, with no functional/hardware accuracy claim.
"""
import argparse
import json
from pathlib import Path

from arch_explorer.adf import MappingSpec, WorkloadSpec
from arch_explorer.components.accelsim.scenarios import sm90_adf
from arch_explorer.components.accelsim.profile import ACCELSIM_SHA, GPGPUSIM_SHA, PROFILE
from arch_explorer.components.accelsim.trace import TRACE_SCHEMA, _trace_info, digest, identity

TEXT = '''-kernel name = synthetic_memory_chain
-kernel id = 1
-grid dim = (1,1,1)
-block dim = (32,1,1)
-shmem = 0
-nregs = 8
-binary version = 90
-cuda stream id = 0
-enable lineinfo = 0
-nvbit version = synthetic-no-nvbit
-accelsim tracer version = 3
-shmem base_addr = 0x0
-local mem base_addr = 0x0
# synthetic trace: PC mask ndst dst opcode nsrc src mem-width address-mode base stride imm
#BEGIN_TB
thread block = 0,0,0
warp = 0
insts = 4
0000 ffffffff 1 R1 LDG.E 0 4 1 100000 4 0
0010 ffffffff 1 R2 IADD3 1 R1 0 0
0020 ffffffff 0 STG.E 1 R2 4 1 200000 4 0
0030 ffffffff 0 EXIT 0 0 0
#END_TB
'''


def create_example(directory, engine=None):
    """Create public synthetic trace, versioned manifest, and CLI request."""
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    raw = TEXT.encode()
    commands = b'kernel-1.traceg\n'
    workload = WorkloadSpec('operator', {'op': 'trace_kernel', 'kernel': 'synthetic_memory_chain',
                                         'grid': [1, 1, 1], 'block': [32, 1, 1]})
    mapping = MappingSpec('accelsim_trace', {'trace_id': 'synthetic-memory-chain-v1',
                                           'compiler': 'hand-authored-no-gpu', 'grid': [1, 1, 1], 'block': [32, 1, 1]})
    manifest = {'schema': TRACE_SCHEMA, 'format': 'traceg', 'isa': 'sm90', 'tracer_version': 3,
                'source': {'kind': 'synthetic', 'gpu': 'none', 'toolchain': 'hand-authored traceg v3',
                           'description': 'One warp load/add/store/exit; software integration only; never collected on GPU'},
                'engine': {'accelsim': ACCELSIM_SHA, 'gpgpusim': GPGPUSIM_SHA, 'profile': PROFILE},
                'scope': 'single-kernel-device-cycles', 'workload': identity(workload), 'mapping': mapping.to_dict(),
                'commands': {'file': 'kernelslist.g', 'sha256': digest(commands)},
                'kernels': [{'file': 'kernel-1.traceg', 'sha256': digest(raw), 'original_sha256': digest(raw), **_trace_info(raw)}]}
    data = (json.dumps(manifest, indent=2) + '\n').encode()
    (directory / 'kernel-1.traceg').write_bytes(raw)
    (directory / 'kernelslist.g').write_bytes(commands)
    (directory / 'trace-manifest.json').write_bytes(data)
    workload.params['accelsim'] = {'manifest': str(directory / 'trace-manifest.json'), 'sha256': digest(data)}
    request = {'schema': 'arch-explorer.accelsim-request.v1', 'adf': sm90_adf().to_dict(),
               'workload': workload.to_dict(), 'mapping': mapping.to_dict(),
               'options': {'engine': str(Path(engine).resolve()) if engine is not None else None,
                           'output_dir': str(directory / 'runs')}}
    (directory / 'request.json').write_text(json.dumps(request, indent=2) + '\n')
    return request


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    parser.add_argument('--engine', type=Path)
    args = parser.parse_args()
    create_example(args.output, args.engine)
    print(args.output.resolve() / 'request.json')


if __name__ == '__main__':
    main()
