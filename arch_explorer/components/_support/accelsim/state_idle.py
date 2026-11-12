"""Idle-device admission: one shared lock, bounded observation, no owner approval.

Import, request creation and all validators are CPU-only. Only acquire() runs
the frozen management query, after acquiring the common lock. It never launches
an application or stops a foreign process. Existing handoffs remain replayable.
"""
import argparse
import fcntl
import json
import os
import re
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

from arch_explorer.components.accelsim.microbundle import json_file
from arch_explorer.components.accelsim.microexperiment import ref, write_new
from arch_explorer.components.accelsim.microrecords import original
from arch_explorer.components.accelsim.trace import exact, same_json

REQUEST = 'arch-explorer.measurement-idle-request.v1'
ADMISSION = 'arch-explorer.measurement-idle-admission.v1'
QUERY = 'arch-explorer.measurement-idle-observation.v1'
_NS = 10**9
_BASE = {'schema', 'work_item', 'run_id', 'design_sha256', 'registry_sha256',
         'plan_sha256', 'budget', 'lock_file', 'claims_directory',
         'issued_realtime_ns', 'absolute_deadline_realtime_ns', 'close_reserve_s',
         'device_ordinal', 'gpu_uuid'}


def require(value, message):
    if not value:
        raise ValueError('idle device admission: ' + message)


def query_argv(request):
    return ['/usr/bin/nvidia-smi', '--id='+str(request['device_ordinal']), '-q', '-x']


def validate_request(bundle, design, data):
    from arch_explorer.components.accelsim.measurement_state import sha_plan
    from arch_explorer.components.accelsim.microcontract import work_item
    from arch_explorer.components.accelsim.state_resource import _role_budget
    exact(data, _BASE, 'idle admission request')
    values = _role_budget(bundle)
    require(data['schema'] == REQUEST and data['work_item'] == work_item(bundle)
            and data['run_id'] == design['run_id']
            and re.fullmatch(r'[A-Za-z0-9_-]{1,128}', data['run_id']) is not None
            and data['design_sha256'] == design['design_sha256']
            and data['registry_sha256'] == bundle.sha256
            and data['plan_sha256'] == design['plan_sha256'] == sha_plan(bundle)
            and same_json(data['budget'], values) and same_json(design['budget'], values),
            'new run/design/registry/plan/full budget required')
    for key in ('lock_file', 'claims_directory'):
        require(isinstance(data[key], str) and Path(data[key]).is_absolute()
                and str(Path(data[key]).resolve()) == data[key], 'canonical absolute '+key+' required')
    require(data['claims_directory'] == str(Path(data['lock_file']).parent/'.measurement-runs'),
            'persistent run claims must share the common lock directory')
    require(type(data['device_ordinal']) is int and data['device_ordinal'] == 0
            and isinstance(data['gpu_uuid'], str)
            and re.fullmatch(r'GPU-[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}', data['gpu_uuid']) is not None,
            'frozen GPU0 UUID required')
    start, end = data['issued_realtime_ns'], data['absolute_deadline_realtime_ns']
    require(type(start) is int and type(end) is int
            and design['created_realtime_ns'] < start < end <= start+values['total_window_s']*_NS
            and type(data['close_reserve_s']) is int and data['close_reserve_s'] == 60,
            'absolute deadline and sixty-second cleanup reserve required')
    return data


def make_request(bundle, design, lock_file, gpu_uuid, *, now_ns=None):
    from arch_explorer.components.accelsim.measurement_state import sha_plan
    from arch_explorer.components.accelsim.microcontract import work_item, budget
    start = time.time_ns() if now_ns is None else now_ns
    lock = Path(lock_file).resolve()
    data = {'schema': REQUEST, 'work_item': work_item(bundle), 'run_id': design['run_id'],
            'design_sha256': design['design_sha256'], 'registry_sha256': bundle.sha256,
            'plan_sha256': sha_plan(bundle), 'budget': budget(bundle), 'lock_file': str(lock),
            'claims_directory': str(lock.parent/'.measurement-runs'), 'issued_realtime_ns': start,
            'absolute_deadline_realtime_ns': start+budget(bundle)['total_window_s']*_NS,
            'close_reserve_s': 60, 'device_ordinal': 0, 'gpu_uuid': gpu_uuid}
    return validate_request(bundle, design, data)


def occupancy(stdout, stderr, exit_code, expected_uuid):
    """Classify only unambiguous identity, process and utilization observations.

    Unrelated management fields remain optional. Admission-critical fields must
    have one occurrence at their prescribed location, with plain scalar leaves;
    a second identical value is just as ambiguous as a conflicting value.
    """
    if exit_code != 0 or stderr.strip():
        return 'unknown', None
    try:
        require(len(stdout) <= 2**20 and b'<!ENTITY' not in stdout.upper(), 'bounded XML required')
        root = ET.fromstring(stdout)
        require(root.tag == 'nvidia_smi_log', 'unexpected management XML')
        devices = root.findall('gpu')
        require(len(devices) == 1, 'exactly one selected device required')
        gpu = devices[0]
        require(list(root.iter('gpu')) == devices, 'unexpected nested device observation')

        def field(parent, name):
            nodes = parent.findall(name)
            require(len(nodes) == 1 and list(gpu.iter(name)) == nodes,
                    'unique, correctly located '+name+' required')
            node = nodes[0]
            require(not node.attrib and not (node.tail or '').strip(),
                    'invalid '+name+' structure')
            return node

        def scalar(node):
            require(len(node) == 0 and not node.attrib,
                    'plain scalar management value required')
            return (node.text or '').strip()

        require(scalar(field(gpu, 'uuid')) == expected_uuid, 'selected GPU UUID differs')
        processes = field(gpu, 'processes')
        process_text = (processes.text or '').strip()
        require(process_text in ('', 'None') and (process_text != 'None' or len(processes) == 0),
                'process list unavailable')
        # Ownership is an invariant over the entire selected GPU subtree, not
        # just the nodes visited while reading the expected process list.
        parents = {child: parent for parent in gpu.iter() for child in parent}
        process_nodes = list(gpu.iter('process_info'))
        require(all(parents[node] is processes for node in process_nodes),
                'process observation outside the selected process list')
        require(all(parents[node] in process_nodes for node in gpu.iter('pid')),
                'process identity outside a process observation')
        pids = []
        for process in processes:
            require(process.tag == 'process_info' and not process.attrib
                    and not (process.text or '').strip() and not (process.tail or '').strip(),
                    'unknown process observation')
            nodes = process.findall('pid')
            require(len(nodes) == 1 and list(process.iter('pid')) == nodes,
                    'unique process identity required')
            pid = scalar(nodes[0])
            require(not (nodes[0].tail or '').strip() and re.fullmatch(r'[0-9]+', pid)
                    is not None and int(pid) > 0, 'unknown process identity')
            pids.append(int(pid))
        utilization_node = field(gpu, 'utilization')
        require(not (utilization_node.text or '').strip(), 'invalid utilization structure')
        usage = scalar(field(utilization_node, 'gpu_util'))
        require(re.fullmatch(r'\s*\d+\s*%\s*', usage) is not None, 'GPU utilization unavailable')
        utilization = int(usage.strip().rstrip('%').strip())
        require(0 <= utilization <= 100, 'invalid GPU utilization')
        facts = {'gpu_uuid': expected_uuid, 'process_ids': pids, 'gpu_utilization_percent': utilization,
                 'memory_used_raw': gpu.findtext('fb_memory_usage/used')}
        return ('busy' if pids or utilization else 'idle'), facts
    except (ValueError, ET.ParseError, TypeError):
        return 'unknown', None


def validate_observation(request, data):
    exact(data, {'schema', 'kind', 'run_id', 'request', 'argv', 'exit_code', 'stdout', 'stderr',
                 'lock', 'status', 'facts', 'started_realtime_ns', 'ended_realtime_ns',
                 'started_monotonic_ns', 'ended_monotonic_ns'}, 'idle observation')
    require(data['schema'] == QUERY and data['kind'] == 'hardware'
            and data['run_id'] == request['run_id'] and data['argv'] == query_argv(request),
            'observation scope/command differs')
    raw = json_file(original(data['request']))[0]
    require(same_json(raw, request), 'observation borrowed another request')
    lock = data['lock'];exact(lock, {'path', 'device', 'inode', 'fd', 'exclusive'}, 'idle common lock')
    require(lock['path'] == request['lock_file'] and lock['exclusive'] is True
            and all(type(lock[k]) is int and lock[k] >= 0 for k in ('device', 'inode', 'fd')),
            'exclusive common-lock identity required')
    a, b, m, n = (data[k] for k in ('started_realtime_ns', 'ended_realtime_ns',
                                    'started_monotonic_ns', 'ended_monotonic_ns'))
    require(all(type(x) is int for x in (a, b, m, n))
            and request['issued_realtime_ns'] <= a <= b < request['absolute_deadline_realtime_ns']-60*_NS
            and 0 < m <= n and b-a <= 11*_NS and n-m <= 11*_NS
            and abs((b-a)-(n-m)) < _NS, 'idle query outside fixed work boundary')
    stdout = original(data['stdout'], maximum=2**20).read_bytes()
    stderr = original(data['stderr'], maximum=2**20).read_bytes()
    require(type(data['exit_code']) is int or data['exit_code'] is None, 'query exit code required')
    status, facts = occupancy(stdout, stderr, data['exit_code'], request['gpu_uuid'])
    require(data['status'] == status and same_json(data['facts'], facts), 'idle claim differs from raw query')
    return data


def validate_admission(bundle, design, data):
    exact(data, _BASE | {'request', 'observation', 'claim'}, 'idle admission')
    require(data['schema'] == ADMISSION, 'idle admission schema required')
    request = json_file(original(data['request']))[0]
    validate_request(bundle, design, request)
    require(all(same_json(data[k], request[k]) for k in _BASE-{'schema'}), 'admission changed frozen request')
    query = validate_observation(request, json_file(original(data['observation']))[0])
    require(query['request'] == data['request'] and query['status'] == 'idle', 'occupied/unknown device cannot admit hosts')
    claim_path = original(data['claim']);claim = json_file(claim_path)[0]
    exact(claim, {'schema', 'run_id', 'request'}, 'unique run claim')
    require(claim == {'schema': 'arch-explorer.measurement-run-claim.v1',
                     'run_id': request['run_id'], 'request': data['request']}
            and claim_path == Path(request['claims_directory'])/request['run_id']/'CLAIM.json',
            'persistent unique run claim differs')
    return data


def acquire(transport, request_ref, *, runner=None):
    """One lock/query attempt. Failures stay terminal; release belongs to Campaign."""
    request = json_file(original(request_ref))[0]
    validate_request(transport.bundle, transport.design, request)
    runner = subprocess.run if runner is None else runner
    now = time.time_ns()
    require(request['issued_realtime_ns'] <= now, 'request is in the future; no lock/query/host started')
    remaining = (request['absolute_deadline_realtime_ns']-now)/_NS-60
    require(remaining > 11, 'request expired; no lock/query/host started')
    transport.lock_fd = os.open(request['lock_file'], os.O_RDWR | os.O_NOFOLLOW)
    try:
        fcntl.flock(transport.lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BaseException:
        os.close(transport.lock_fd);transport.lock_fd = None
        raise
    transport.resource_lock_acquired = True
    transport.started = time.monotonic()
    transport.deadline = transport.started+min(remaining, transport.budget['total_window_s']-60)
    require(transport.remaining() > 11, 'work boundary reached before query')
    claims = Path(request['claims_directory']);claims.mkdir(exist_ok=True)
    claim_root = claims/request['run_id'];claim_root.mkdir(exist_ok=False)
    claim = write_new(claim_root/'CLAIM.json', {'schema': 'arch-explorer.measurement-run-claim.v1',
                      'run_id': request['run_id'], 'request': request_ref})
    info = os.fstat(transport.lock_fd)
    lock = {'path': request['lock_file'], 'device': info.st_dev, 'inode': info.st_ino,
            'fd': transport.lock_fd, 'exclusive': True}
    start, mono = time.time_ns(), time.monotonic_ns()
    try:
        result = runner(query_argv(request), stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                        timeout=min(10, transport.remaining()), check=False)
        stdout, stderr, code = result.stdout, result.stderr, result.returncode
    except subprocess.TimeoutExpired as error:
        stdout, stderr, code = error.stdout or b'', error.stderr or b'', None
    end, end_mono = time.time_ns(), time.monotonic_ns()
    transport.root.joinpath('idle.stdout').write_bytes(stdout)
    transport.root.joinpath('idle.stderr').write_bytes(stderr)
    status, facts = occupancy(stdout, stderr, code, request['gpu_uuid'])
    observation = {'schema': QUERY, 'kind': 'hardware', 'run_id': request['run_id'],
                   'request': request_ref, 'argv': query_argv(request), 'exit_code': code,
                   'stdout': ref(transport.root/'idle.stdout'), 'stderr': ref(transport.root/'idle.stderr'),
                   'lock': lock, 'status': status, 'facts': facts, 'started_realtime_ns': start,
                   'ended_realtime_ns': end, 'started_monotonic_ns': mono, 'ended_monotonic_ns': end_mono}
    observation_ref = write_new(transport.root/'IDLE-OBSERVATION.json', observation)
    validate_observation(request, observation)
    require(status == 'idle', 'occupied or unknown GPU; coordinate conflict; no host started')
    admission = {**request, 'schema': ADMISSION, 'request': request_ref,
                 'observation': observation_ref, 'claim': claim}
    validate_admission(transport.bundle, transport.design, admission)
    return write_new(transport.root/'IDLE-ADMISSION.json', admission), admission


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    plan = sub.add_parser('plan', help='CPU-only; freeze a new local request, not resource authority')
    plan.add_argument('design');plan.add_argument('request')
    plan.add_argument('--lock-file', required=True);plan.add_argument('--gpu-uuid', required=True)
    run = sub.add_parser('run', help='one supervised idle check and bounded campaign; no owner reply required')
    run.add_argument('design');run.add_argument('request');run.add_argument('window');run.add_argument('supervisor')
    a = parser.parse_args(argv)
    from arch_explorer.components.accelsim.state_device import validate_design
    bundle, design = validate_design(a.design, installation=True)
    from arch_explorer.components.accelsim.microcampaign import check_installation
    check_installation(design)
    if a.command == 'plan':
        data = make_request(bundle, design, a.lock_file, a.gpu_uuid)
        print(json.dumps(write_new(Path(a.request).resolve(), data)))
        return 0
    request_ref = ref(a.request);data = validate_request(bundle, design, json_file(original(request_ref))[0])
    require(data['issued_realtime_ns'] <= time.time_ns()
            and data['absolute_deadline_realtime_ns']-time.time_ns() >= 90*_NS,
            'request expired or in the future; no worker started')
    require(not Path(a.window).exists() and not Path(a.supervisor).exists(), 'existing window/supervisor cannot be repeated')
    from arch_explorer.components.accelsim.microguard import supervise
    result = supervise([sys.executable, '-m', 'arch_explorer.components.accelsim.microworker',
                        str(Path(a.design).resolve()), str(Path(a.request).resolve()), str(Path(a.window).resolve())],
                       a.supervisor, data['absolute_deadline_realtime_ns'], work=data['work_item'],
                       run_id=data['run_id'], design=ref(a.design), handoff=request_ref,
                       lock=data['lock_file'], reserve_s=60)
    print(json.dumps(result))
    return 0 if result['exit_code'] == 0 else 1


if __name__ == '__main__':
    raise SystemExit(main())
