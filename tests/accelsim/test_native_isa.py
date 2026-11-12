"""Native dispatch oracle versus adapter eligibility; synthetic ELF metadata."""
import json
from pathlib import Path
import struct
from unittest.mock import patch
import pytest
from arch_explorer.adf import MappingSpec,WorkloadSpec
from arch_explorer.simulator.accelsim.capabilities import preflight_cubin,preflight_request,assess
from arch_explorer.simulator.accelsim.config import compile_adf
from arch_explorer.simulator.accelsim.example import create_example
from arch_explorer.simulator.accelsim.native_isa import NATIVE_ISAS,isa_facts
from arch_explorer.simulator.accelsim.trace import digest,load_trace
from arch_explorer.simulator.accelsim.trace_v2 import create_request


def save_manifest(request, manifest):
    ref=request['workload']['params']['accelsim']
    file=Path(ref['manifest']);raw=(json.dumps(manifest)+'\n').encode()
    file.write_bytes(raw);ref['sha256']=digest(raw)


def test_exact_pinned_dispatch_families():
    assert NATIVE_ISAS=={35:'kepler',60:'pascal',61:'pascal',70:'volta',75:'turing',80:'ampere',86:'ampere',90:'hopper'}


@pytest.mark.parametrize('number',[35,60,61,70,75,80,86,90,120,0])
def test_synthetic_static_header_native_and_adapter_layers(tmp_path,number):
    header=bytearray(64);header[:6]=b'\x7fELF\x02\x01'
    struct.pack_into('<H',header,18,190);struct.pack_into('<I',header,48,number<<8)
    file=tmp_path/'synthetic-not-a-kernel.cubin';file.write_bytes(header)
    result=preflight_cubin(file);kinds={r['kind'] for r in result['blockers']}
    assert 'dynamic_trace_missing' in kinds and result['device_queried'] is False
    assert result['isa_capability']['native_dispatch']==(number in NATIVE_ISAS)
    if number==90:
        assert kinds=={'dynamic_trace_missing'}
    else:
        assert kinds==({'adapter_isa','dynamic_trace_missing'} if number in NATIVE_ISAS else {'native_isa','dynamic_trace_missing'})


@pytest.mark.parametrize('number,kind',[(80,'adapter_isa'),(86,'adapter_isa'),(120,'native_isa')])
def test_same_layer_in_target_and_v1_declared_and_actual_trace(tmp_path,number,kind):
    request=create_example(tmp_path/'input')
    workload,mapping=WorkloadSpec(**request['workload']),MappingSpec(**request['mapping'])
    manifest,_,_,actual=load_trace(workload,mapping)
    config=compile_adf(__import__('arch_explorer.adf',fromlist=['from_dict']).from_dict(request['adf']))
    target={'gpu':'synthetic identity, no query','isa':'sm'+str(number),'evidence_sha256':'b'*64}
    result=assess(config,manifest,actual,purpose='hardware-accuracy',target=target)
    assert {r['kind'] for r in result['blockers']}=={kind,'physical_parameters'}
    original=json.loads(json.dumps(manifest))
    manifest['isa']='sm'+str(number);save_manifest(request,manifest)
    assert preflight_request(request)['blockers'][0]['kind']==kind
    manifest=original
    file=Path(request['workload']['params']['accelsim']['manifest']).parent/manifest['kernels'][0]['file']
    raw=file.read_text().replace('-binary version = 90','-binary version = '+str(number)).encode()
    file.write_bytes(raw)
    manifest['kernels'][0].update(sha256=digest(raw),original_sha256=digest(raw));save_manifest(request,manifest)
    assert preflight_request(request)['blockers'][0]['kind']==kind


@pytest.mark.parametrize('number,kind',[(80,'adapter_isa'),(86,'adapter_isa'),(120,'native_isa')])
def test_v2_actual_and_manifest_isa_share_dispatch(tmp_path,number,kind):
    create_example(tmp_path/'input')
    trace=tmp_path/'input/kernel-1.traceg'
    request=create_request(trace,tmp_path/'review',source={'kind':'synthetic','gpu':'none','toolchain':'CPU only','description':'not collected'},trace_id='cpu',compiler='none')
    path=Path(request['workload']['params']['accelsim']['manifest'])
    original=json.loads(path.read_text())
    declared=json.loads(path.read_text());declared['isa']='sm'+str(number);save_manifest(request,declared)
    assert preflight_request(request)['blockers'][0]['kind']==kind
    raw=(path.parent/'kernel-1.traceg').read_text().replace('-binary version = 90','-binary version = '+str(number)).encode()
    (path.parent/'kernel-1.traceg').write_bytes(raw)
    original['kernels'][0].update(sha256=digest(raw),original_sha256=digest(raw));save_manifest(request,original)
    assert preflight_request(request)['blockers'][0]['kind']==kind


@pytest.mark.parametrize('update',[{'format':'unknown'},{'tracer_version':7}])
def test_v2_format_limits_not_mislabeled_missing_native_isa(tmp_path,update):
    create_example(tmp_path/'input')
    request=create_request(tmp_path/'input/kernel-1.traceg',tmp_path/'review',
        source={'kind':'synthetic','gpu':'none','toolchain':'CPU only','description':'not collected'},trace_id='cpu',compiler='none')
    file=Path(request['workload']['params']['accelsim']['manifest'])
    manifest=json.loads(file.read_text());manifest.update(update);save_manifest(request,manifest)
    assert preflight_request(request)['blockers'][0]['kind']=='adapter'


def test_capability_functions_never_spawn():
    with patch('subprocess.Popen',side_effect=AssertionError('forbidden')) as spy:
        assert isa_facts('sm80')['native_family']=='ampere'
        assert isa_facts('sm120')['native_dispatch'] is False
        assert isa_facts('unknown')['adapter_supported'] is False
        assert spy.call_count==0

