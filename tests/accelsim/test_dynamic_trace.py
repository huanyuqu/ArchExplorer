"""Independent dynamic-input oracles; all embedded traces are synthetic."""
from copy import deepcopy
import ctypes as C
import json
import os
from pathlib import Path
import struct
import subprocess
import sys

import pytest

from arch_explorer.adf import MappingSpec, WorkloadSpec
from arch_explorer.simulator.accelsim.backend import evaluate_request
from arch_explorer.simulator.accelsim.capabilities import preflight_request
from arch_explorer.simulator.accelsim.example import TEXT
from arch_explorer.simulator.accelsim.stats import parse_stats
from arch_explorer.simulator.accelsim.trace import digest, load_trace
from arch_explorer.simulator.accelsim.trace_stream import facts
from arch_explorer.simulator.accelsim.trace_v2 import create_request


SOURCE = dict(kind='synthetic', gpu='none', toolchain='CPU test text, not a GPU capture',
              description='Dynamic mask/PC/EXIT completeness oracle only')


def dynamic_text():
    header = TEXT.split('# synthetic')[0].replace('synthetic_memory_chain', '_Z23another_kernel_name_v7').replace('(1,1,1)', '(2,1,1)', 1).replace('(32,1,1)', '(48,1,1)')
    body = []
    for cta in (1, 0):  # Complete noncanonical CTA order is legal for singleton clusters.
        body += ['#BEGIN_TB', f'thread block = {cta},0,0',
                 f'cluster id = {cta},0,0', 'cluster cta = 0,0,0', 'cluster rank = 0',
                 'warp = 1', 'insts = 2',
                 '0000 0000ffff 1 R1 IADD3 0 0 0', '0010 0000ffff 0 EXIT 0 0 0',
                 'warp = 0', 'insts = 8',
                 '0000 ffffffff 1 R1 LDG.E 0 4 1 100000 4 0',
                 '0010 ffffffff 1 R2 IADD3 1 R1 0 0',
                 '0010 0000ffff 1 R2 IADD3 1 R1 0 0',  # Executed loop PC again.
                 '0020 00000000 1 R2 IADD3 1 R1 0 0',  # Retain predicated-off record.
                 '0030 0000ffff 0 EXIT 0 0 0',          # Half the warp exits early.
                 '0040 ffff0000 1 R2 IADD3 1 R1 0 0',
                 '0050 ffff0000 0 STG.E 1 R2 4 1 200000 4 0',
                 '0060 ffff0000 0 EXIT 0 0 0', '#END_TB']
    return header+'# synthetic dynamic trace format\n'+'\n'.join(body)+'\n'


def make_request(tmp_path, text=None):
    trace = tmp_path/'kernel-9.traceg'
    trace.write_text(dynamic_text() if text is None else text)
    return create_request(trace, tmp_path, source=deepcopy(SOURCE), trace_id='different-valid-input', compiler='CPU synthetic')


def rewrite(request, update):
    ref = request['workload']['params']['accelsim']
    path = Path(ref['manifest'])
    manifest = json.loads(path.read_text())
    update(manifest)
    raw = (json.dumps(manifest)+'\n').encode()
    path.write_bytes(raw)
    ref['sha256'] = digest(raw)


def test_independent_dynamic_mask_and_loop_oracle(tmp_path):
    request = make_request(tmp_path)
    actual = load_trace(WorkloadSpec(**request['workload']), MappingSpec(**request['mapping']))[3]
    # Hand sums per CTA: 32+32+16+0+16+16+16+16 + 16+16 = 176.
    assert actual['ctas'] == 2 and actual['warp_streams'] == 4
    assert actual['warp_instructions'] == 20 and actual['thread_instructions'] == 352
    assert actual['repeated_dynamic_pcs'] == 2 and actual['masks']['0'] == 2
    assert actual['memory_records'] == 4
    assert preflight_request(request)['status'] == 'passed'


@pytest.mark.parametrize('edit,match', [
    (lambda s: s[:s.rfind('#BEGIN_TB')], 'missing CTA'),
    (lambda s: s.replace('warp = 1\ninsts = 2\n0000 0000ffff 1 R1 IADD3 0 0 0\n0010 0000ffff 0 EXIT 0 0 0\n', ''), 'missing/extra warp'),
    (lambda s: s.replace('insts = 8', 'insts = 9'), 'truncated'),
    (lambda s: s.replace('thread block = 0,0,0', 'thread block = 1,0,0'), 'duplicate'),
    (lambda s: s.replace('warp = 0', 'warp = 1'), 'duplicate'),
    (lambda s: s.replace('0060 ffff0000', '0060 0000ffff'), 'already EXITed'),
    (lambda s: s.replace('0000 0000ffff', '0000 ffffffff'), 'nonexistent'),
    (lambda s: s.replace('0060 ffff0000 0 EXIT', '0060 ffff0000 0 IADD3'), 'EXIT'),
    (lambda s: s.replace('LDG.E 0 4 1 100000 4', 'LDG.E 0 0'), 'memory address'),
    (lambda s: s.replace('IADD3', 'UNKNOWN_NEW_OP'), 'native_isa'),
    (lambda s: s.replace('IADD3', 'IADD3.NONSENSE'), 'adapter:'),
    (lambda s: s.replace('IADD3', 'GMMA'), 'adapter:'),
    (lambda s: s.replace('IADD3', 'HGMMA'), 'adapter:'),
    (lambda s: s.replace('IADD3', 'UTMACCTL'), 'adapter:'),
    (lambda s: s.replace('-binary version = 90', '-binary version = 120'), 'native_isa'),
    (lambda s: s.replace('cluster cta = 0,0,0', 'cluster cta = 1,0,0'), 'multi-CTA cluster'),
])
def test_incomplete_or_unsupported_dynamic_input(tmp_path, edit, match):
    path = tmp_path/'kernel-9.traceg'
    path.write_text(edit(dynamic_text()))
    with pytest.raises(ValueError, match=match):
        facts(path)


@pytest.mark.parametrize('edit', [
    lambda m: m['kernels'][0]['facts'].update(warp_instructions=19),
    lambda m: m['kernels'][0].update(thread_instructions=640),
    lambda m: m['kernels'][0]['facts'].update(nregs=True),
    lambda m: m['kernels'][0].update(grid=[3,1,1]),
])
def test_rehashed_false_facts_refused_before_native(tmp_path, monkeypatch, edit):
    request = make_request(tmp_path)
    rewrite(request, edit)
    monkeypatch.setattr('arch_explorer.simulator.accelsim.backend.run_native', lambda *a, **k: pytest.fail('invalid facts reached native'))
    with pytest.raises(ValueError):
        evaluate_request(request)


def test_resource_and_blackwell_preflight_are_not_engine_calls(tmp_path, monkeypatch):
    request = make_request(tmp_path, dynamic_text().replace('-shmem = 0', '-shmem = 233473'))
    monkeypatch.setattr('arch_explorer.simulator.accelsim.backend.run_native', lambda *a, **k: pytest.fail('resource refusal reached native'))
    assert preflight_request(request)['blockers'][0]['kind'] == 'adapter_resource'
    with pytest.raises(ValueError, match='preflight blocked'):
        evaluate_request(request)
    target = dict(gpu='existing SM120 evidence', isa='sm120', evidence_sha256='a'*64)
    result = preflight_request(request, purpose='hardware-accuracy', target=target)
    assert {'native_isa', 'physical_parameters'} <= {r['kind'] for r in result['blockers']}


def compress_independently(text, path):
    """Tiny tracez writer from fixed footer layout, independent of production reader."""
    try:
        zstd = C.CDLL('libzstd.so.1')
    except OSError:
        pytest.skip('optional native zstd dependency unavailable')
    zstd.ZSTD_compress.argtypes = [C.c_void_p,C.c_size_t,C.c_void_p,C.c_size_t,C.c_int]
    zstd.ZSTD_compress.restype = C.c_size_t
    header, body = text.split('#BEGIN_TB', 1)
    output = bytearray(header.encode()+b'\0'+struct.pack('<II', 0, 1024))
    tbs = []
    for tb in ('#BEGIN_TB'+body).split('#BEGIN_TB')[1:]:
        before, rest = tb.split('warp = ', 1)
        xyz = [int(x) for x in before.split('thread block = ')[1].splitlines()[0].split(',')]
        warps = []
        records = [('warp = '+rest).split('warp = ')[i] for i in range(1,len(('warp = '+rest).split('warp = ')))]
        # tracez index stores warp ID implicitly, so canonical order is required.
        for record in sorted(records, key=lambda r:int(r.splitlines()[0])):
            rows = record.splitlines()[2:]
            rows = [r for r in rows if r and not r.startswith('#')]
            data = ('\n'.join(rows)+'\n').encode()
            destination = C.create_string_buffer(len(data)+1024)
            size = zstd.ZSTD_compress(destination,len(destination),data,len(data),3)
            assert size < len(destination)
            offset = len(output)
            output += struct.pack('<III',len(rows),1,size)+destination.raw[:size]
            warps.append((len(rows),offset))
        tbs.append((xyz,warps))
    index = len(output)
    output += struct.pack('<I',len(tbs))
    for xyz, warps in tbs:
        output += struct.pack('<11I', *xyz, *xyz, 0,0,0,0,len(warps))
        for n, offset in warps:
            output += struct.pack('<IIQI',n,1,offset,n)
    output += struct.pack('<Q',index)
    path.write_bytes(output)


def test_compressed_and_text_dynamic_stream_parity_and_bounds(tmp_path):
    text_path = tmp_path/'kernel-9.traceg'
    text_path.write_text(dynamic_text())
    compressed = tmp_path/'kernel-9.tracez'
    compress_independently(dynamic_text(),compressed)
    assert facts(text_path)[0] == facts(compressed)[0]
    raw = compressed.read_bytes()
    for corrupt in (raw[:-8], raw[:-8]+struct.pack('<Q',2**63), raw[:40]):
        compressed.write_bytes(corrupt)
        with pytest.raises(ValueError):
            facts(compressed)


def test_retirement_oracle_does_not_equate_warp_records_to_threads():
    from test_contract import TRANSCRIPT
    expected = dict(name='example',ctas=1,thread_instructions=128,warp_instructions=5)
    stdout = TRANSCRIPT+'gpgpu_n_tot_w_icount = 5\n'
    stats = parse_stats(stdout,'',expected,1980)
    assert stats['gpgpu_n_tot_w_icount'] == 5
    assert 'issued CTAs' in stats['cta_counter_semantics']
    for bad in (TRANSCRIPT, stdout.replace('icount = 5','icount = 4'),stdout+'gpgpu_n_tot_w_icount = 5\n'):
        with pytest.raises(ValueError,match='warp instructions'):
            parse_stats(bad,'',expected,1980)


def test_static_input_identity_is_not_a_collected_trace(tmp_path):
    from arch_explorer.simulator.accelsim.capabilities import preflight_cubin
    # Synthetic ELF metadata oracle, explicitly unrelated to GPU collection.
    header = bytearray(64)
    header[:6] = b'\x7fELF\x02\x01'
    struct.pack_into('<H',header,18,190)
    struct.pack_into('<I',header,48,(120 << 8) | 0x0a)
    path = tmp_path/'synthetic-metadata.cubin'
    path.write_bytes(header)
    result = preflight_cubin(path)
    assert result['isa'] == 'sm120' and result['device_queried'] is False
    assert {r['kind'] for r in result['blockers']} == {'native_isa','dynamic_trace_missing'}


def test_capability_cli_reachable_pass_and_blocked(tmp_path):
    request = make_request(tmp_path)
    path = tmp_path/'request.json'
    target = tmp_path/'target.json'
    target.write_text(json.dumps(dict(gpu='existing evidence identity',isa='sm120',evidence_sha256='a'*64)))
    command = [sys.executable,'-m','arch_explorer.simulator.accelsim.capabilities',str(path)]
    first = subprocess.run(command,capture_output=True,text=True,check=False)
    assert first.returncode == 0 and json.loads(first.stdout)['status'] == 'passed'
    blocked = subprocess.run(command+['--purpose','hardware-accuracy','--target',str(target)],capture_output=True,text=True,check=False)
    assert blocked.returncode == 2
    assert {r['kind'] for r in json.loads(blocked.stdout)['blockers']} == {'native_isa','physical_parameters'}


@pytest.mark.parametrize('replacement', ['100000 4294967296 0','10000000000000000 4 0'])
def test_native_memory_integer_bounds(tmp_path,replacement):
    path=tmp_path/'kernel-9.traceg'
    path.write_text(dynamic_text().replace('100000 4 0',replacement))
    with pytest.raises(ValueError,match='integer range'):
        facts(path)


@pytest.mark.skipif(not os.environ.get('ARCH_EXPLORER_ACCELSIM_ENGINE'), reason='optional CPU native engine')
def test_native_shared_barrier_scope(tmp_path):
    header = TEXT.split('# synthetic')[0].replace('(32,1,1)','(64,1,1)').replace('-shmem = 0','-shmem = 256')
    rows = [header+'# synthetic shared/barrier\n#BEGIN_TB\nthread block = 0,0,0']
    for warp in (0,1):
        rows += [f'warp = {warp}', 'insts = 4',
                 '0000 ffffffff 0 STS 0 4 1 1000 4 0',
                 '0010 ffffffff 0 BAR.SYNC 0 0 0',
                 '0020 ffffffff 1 R1 LDS 0 4 1 1000 4 0',
                 '0030 ffffffff 0 EXIT 0 0 0']
    request = make_request(tmp_path,'\n'.join(rows)+'\n#END_TB\n')
    request['options']['engine'] = os.environ['ARCH_EXPLORER_ACCELSIM_ENGINE']
    native = evaluate_request(request)['evaluation']['details']['native']
    assert native['gpu_tot_sim_insn'] == 256 and native['gpgpu_n_tot_w_icount'] == 8


@pytest.mark.skipif(not os.environ.get('SIM27_REAL_TRACE'), reason='external collected trace deliberately excluded from public fixtures')
def test_external_real_trace_and_fixed_tool_parity():
    original = Path(os.environ['SIM27_REAL_TRACE'])
    actual = facts(original)[0]
    assert actual == facts(original.with_suffix('.traceg'))[0]
    assert (actual['ctas'],actual['warp_streams'],actual['warp_instructions'],actual['thread_instructions']) == (32,256,22784,704512)


@pytest.mark.skipif(not os.environ.get('ARCH_EXPLORER_ACCELSIM_ENGINE'), reason='optional CPU native engine')
def test_dynamic_synthetic_native_retirement(tmp_path):
    request = make_request(tmp_path)
    request['options']['engine'] = os.environ['ARCH_EXPLORER_ACCELSIM_ENGINE']
    result = evaluate_request(request)['evaluation']['details']['native']
    assert result['gpu_tot_sim_insn'] == 352
    assert result['gpu_tot_issued_cta'] == 2
    assert result['gpgpu_n_tot_w_icount'] == 20
