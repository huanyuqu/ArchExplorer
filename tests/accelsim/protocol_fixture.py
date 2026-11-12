"""Synthetic completed-run fixtures. No native/GPU execution or real commit claim."""
from copy import deepcopy
import io
import json
from pathlib import Path
import tarfile
from unittest.mock import patch
import pytest
from arch_explorer.adf import MappingSpec, WorkloadSpec, from_dict
from arch_explorer.simulator.accelsim.backend import CAPABILITIES, REQUEST_SCHEMA, RESULT_SCHEMA
from arch_explorer.simulator.accelsim.capabilities import assess
from arch_explorer.simulator.accelsim.config import compile_adf
from arch_explorer.simulator.accelsim.example import create_example
from arch_explorer.simulator.accelsim.profile import ACCELSIM_SHA, GPGPUSIM_SHA
from arch_explorer.simulator.accelsim.protocol import HISTORY_SCHEMA, SOURCE_PREFIX, prepare_review
from arch_explorer.simulator.accelsim.stats import parse_stats
from arch_explorer.simulator.accelsim.trace import digest, load_trace
from arch_explorer.simulator.accelsim.trace_stream import file_digest
from arch_explorer.simulator.accelsim.result_contract import DETAIL_SCHEMA, COMPONENT_MAP_SCHEMA


def write(file, value):
    file.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    return file


def frozen(spec, role):
    return next(r for r in spec['frozen'] if r['role'] == role)


def replace_role(spec, role, file):
    frozen(spec, role).update(path=str(file), sha256=file_digest(file))


def edit_file(spec, role, update):
    file = Path(frozen(spec, role)['path'])
    value = json.loads(file.read_text())
    update(value)
    write(file, value)
    replace_role(spec, role, file)
    return value


@pytest.fixture
def review(tmp_path):
    root = tmp_path/'synthetic'
    root.mkdir()
    binary = root/'fake-engine.out'
    binary.write_bytes(b'not an executed native binary')
    binary.chmod(0o755)
    library = root/'libcudart.so'
    library.write_bytes(b'synthetic library, never loaded')
    engine = {'schema':'arch-explorer.accelsim-engine.v1',
        'sources':{'accelsim':ACCELSIM_SHA,'gpgpusim':GPGPUSIM_SHA},
        'binary':{'path':str(binary),'sha256':file_digest(binary)},
        'libraries':[{'path':str(library),'sha256':file_digest(library)}],
        'build':{'kind':'synthetic metadata; no native build/execution'}}
    engine_path = write(root/'engine.json', engine)
    request = create_example(root/'input', engine_path)
    request['options']['output_dir'] = str(root/'must-not-start')
    write(root/'input/request.json', request)
    manifest, trace, manifest_hash, expected = load_trace(WorkloadSpec(**request['workload']), MappingSpec(**request['mapping']))
    config = compile_adf(from_dict(request['adf']))
    native = root/'retained-synthetic-record'
    native.mkdir()
    inputs = dict(config.artifacts, **{'limits.config':'-gpgpu_max_cycle 1000000\n',
        'kernelslist.g':'kernel-1.traceg\n', 'kernel-1.traceg':trace.decode()})
    for name, data in inputs.items():
        (native/name).write_text(data)
    stdout = f'''SYNTHETIC transcript; not a native run. labels {ACCELSIM_SHA[:7]} {GPGPUSIM_SHA[:7]}
launching kernel name: synthetic_memory_chain uid: 1 cuda_stream_id: 0
kernel_name = synthetic_memory_chain
kernel_launch_uid = 1
gpu_sim_cycle = 1980
gpu_tot_sim_cycle = 1980
gpu_sim_insn = 128
gpu_tot_sim_insn = 128
gpu_tot_issued_cta = 1
GPGPU-Sim uArch: clock periods: 0.00000000050505050505:0:0:0
GPGPU-Sim: *** simulation thread exiting ***
GPGPU-Sim: *** exit detected ***
'''
    (native/'stdout.log').write_text(stdout)
    (native/'stderr.log').write_text('')
    stats = parse_stats(stdout, '', expected, 1980)
    archive = root/'synthetic-source.tar'
    package = Path(__file__).resolve().parents[2]/'arch_explorer'
    sources = list((package/'simulator/accelsim').glob('*.py'))+[package/'components/accelsim.py',package/'archs/accelsim.py']
    source_hashes = {str(p.relative_to(package)):file_digest(p) for p in sources}
    fake_commit = 'a'*40  # Explicit synthetic label, never a real repository commit.
    with tarfile.open(archive,'w',format=tarfile.PAX_FORMAT,pax_headers={'comment':fake_commit}) as out:
        for file in sources:
            data=file.read_bytes()
            info=tarfile.TarInfo(SOURCE_PREFIX+str(file.relative_to(package)))
            info.size=len(data)
            out.addfile(info,io.BytesIO(data))
    source = write(root/'source-origin.json', {'source_sha':fake_commit,'archive_sha256':file_digest(archive),'clean':True,
        'note':'synthetic source label and archive oracle; never a measured run'})
    run = {'schema':'arch-explorer.accelsim-run.v1',
        'argv':[str(binary),'-config','gpgpusim.config','-config','limits.config','-trace','./kernelslist.g'],
        'cwd':str(native),'environment':{'PATH':'/usr/bin:/bin','LC_ALL':'C','CUDA_VISIBLE_DEVICES':'','LD_LIBRARY_PATH':str(root)},
        'engine':engine,'engine_manifest_sha256':file_digest(engine_path),'config_sha256':config.sha256,'adf_sha256':config.adf_sha256,
        'limits':{'timeout_s':120,'max_cycles':1000000,'memory_limit_mb':4096},
        'files':{name:file_digest(native/name) for name in inputs},
        'status':'complete','adapter_sources_sha256':source_hashes,'pid':1,'exit_code':0,
        'stdout_sha256':file_digest(native/'stdout.log'),'stderr_sha256':file_digest(native/'stderr.log'),
        'stats':stats,'trace_manifest_sha256':manifest_hash}
    write(native/'run.json',run)
    details = {'schema':DETAIL_SCHEMA,'contract':REQUEST_SCHEMA,'capabilities':CAPABILITIES,'preflight':assess(config,manifest,expected),
        'trace_contract':manifest['schema'],'trace_facts':expected,'engine':engine,'engine_manifest_sha256':file_digest(engine_path),
        'trace_manifest_sha256':manifest_hash,'trace_sha256':digest(trace),'trace_source':manifest['source'],
        'config_sha256':config.sha256,'adf_sha256':config.adf_sha256,'component_map':config.component_map,
        'native':stats,'evidence':str(native),'run':run,
        'device_facts':config.device_facts,'component_map_schema':COMPONENT_MAP_SCHEMA}
    evaluation = {'latency_ns':1000.0,'breakdown':[],'utilization':{},'area_mm2':None,'cost_usd':None,
        'warnings':['SYNTHETIC transcript'],
        'provenance':[{k:row[k] for k in ('path','param','value','unit','source')} for row in config.component_map],
        'backend':'accelsim','details':details}
    prediction = write(root/'prediction.json', {'schema':RESULT_SCHEMA,'request':request,'evaluation':evaluation})
    history=write(root/'history.json',{'schema':HISTORY_SCHEMA,'case_identities':[],'semantic_identities':[],
        'closed_binary_sha256':{},'parents':[],'policy':'synthetic registry, not evidence of hardware novelty'})
    return prepare_review(root/'input/request.json',prediction,source,archive,history,root/'review')

