# Concrete Accel-Sim implementation; owned by the component provider.
"""Strict original SIM30 observations; labels and hashes alone prove no hardware.

The collection supervisor supplies provenance; these CPU routines close every
accepted value to that receipt, the frozen design, live functions and outputs.
They make no device calls. Files are reread on replay and before fitting.
"""
import math
import struct
from pathlib import Path
from arch_explorer.components.accelsim.trace import exact, same_json
from arch_explorer.components.accelsim.trace_stream import file_digest
from arch_explorer.components.accelsim.microbundle import json_file, integer, BUDGET, oracle
from arch_explorer.components.accelsim.microcontract import (budget,modern,work_item,identity_fields,schema,check_identity,validate_handoff)

BASE = {'registry_sha256','case','family','partition','steps','function','cubin_sha256','text_sha256','input_sha256','auxiliary_sha256','expected_sha256','grid','block','dynamic_shared','context','stream','pid','freeze_sha256','lease_sha256'}
EVENTS = {'start_event','end_event'}
SAMPLE_FIELDS = BASE | EVENTS | {'schema','sample_index','kind','mode','status','launch_return','synchronize_return','event_elapsed_ms','event_elapsed_ms_float32_bits','elapsed_ns','unit','correct','output_sha256','all_function_handles_live','hidden_warmup_launches','helper_launches','hardware_core_mhz','clock_state','cache_state'}
PREFLIGHT_FIELDS = {'schema','status','registry_sha256','cubin_sha256','module_load','gpu','uuid_hex','driver_api_version','driver_version_file','driver_branch','driver_version','pid','parent_pid','host_realtime_ns','host_monotonic_ns','module','cc','context','stream','limits','functions','all_function_handles_live','application_launches','grid','block','physical_core_mhz'}
PROCESS_FIELDS = {'schema','kind','phase','registry_sha256','design_sha256','design','freeze_sha256','prediction_freeze','lease','command','environment','launch_environment','pid','supervisor_pid','exit_code','status','started_realtime_ns','ended_realtime_ns','started_monotonic_ns','ended_monotonic_ns','host','output','stdout','stderr','lock','budget_account'}
INV_FIELDS = {'schema','registry_sha256','host_sha256','pid','parent_pid','host_realtime_ns','mode','lease_file','lease_sha256','freeze_sha256','common_lock','lock_fd','receipt_validation','monotonic_ns'}
ENV_KEYS = {'CUDA_VISIBLE_DEVICES','CUDA_DEVICE_ORDER','CUDA_MODULE_LOADING','CUDA_LAUNCH_BLOCKING','CUDA_FORCE_PTX_JIT','CUDA_CACHE_DISABLE','CUDA_CACHE_PATH','CUDA_MPS_PIPE_DIRECTORY','CUDA_MPS_ACTIVE_THREAD_PERCENTAGE','LD_LIBRARY_PATH','LD_PRELOAD','LD_AUDIT','CUDA_INJECTION64_PATH','CUDA_INJECTION32_PATH','NVBIT_TOOL','NVBIT_TOOL_PATH','CUDA_PROFILE','CUDA_ENABLE_PROFILING'}
ALLOCATION = BASE | {'schema','buffers','mul','add','input_file','auxiliary_file','expected_file','output_initial_file','actual_output_files','correct','output_status'}


def original(row, maximum=64*2**20):
    exact(row, {'path','sha256'}, 'original record reference')
    p = Path(row['path'])
    if not p.is_absolute() or not p.is_file() or p.stat().st_size > maximum or file_digest(p) != row['sha256']:
        raise ValueError('original record changed/missing')
    return p.resolve()


def validate_preflight(bundle, data):
    exact(data, PREFLIGHT_FIELDS, 'live all-function preflight')
    if data['schema'] != 'arch-explorer.sim30-live-preflight.v1' or data['status'] != 'passed' or data['registry_sha256'] != bundle.sha256 or data['cubin_sha256'] != bundle.document['cubin']['sha256']:
        raise ValueError('fresh live preflight did not pass for this registry/ELF')
    if data['module_load'] != 'cuModuleLoadData: exact frozen ELF image' or not data['module'] or not data['context']:
        raise ValueError('actual live module/context absent')
    if data['cc'] != [12,0] or any(type(x) is not int for x in data['cc']) or not data['gpu'] or not isinstance(data['uuid_hex'], str) or len(data['uuid_hex']) != 32 or data['all_function_handles_live'] is not True or type(data['application_launches']) is not int or data['application_launches'] != 0:
        raise ValueError('live device/function/prelaunch identity missing')
    int(data['uuid_hex'], 16)
    if data['grid'] != bundle.document['grid'] or data['block'] != bundle.document['block'] or data['stream'] != 0 or data['physical_core_mhz'] is not None:
        raise ValueError('preflight geometry/clock claim differs')
    integer(data['driver_api_version'], 12080, 99999, 'driver API version')
    if not data['driver_version_file'] or data['driver_version'] is None:
        raise ValueError('actual driver version unavailable')
    limits = data['limits']
    exact(limits, {'sm_count','warp_size','max_threads_block','registers_sm','registers_block','shared_sm','shared_block','managed_memory','concurrent_managed_access'}, 'live device limits')
    for name, value in limits.items():
        integer(value, 0 if name in ('managed_memory','concurrent_managed_access') else 1, 2**30, 'live '+name)
    if limits['warp_size'] != 32 or limits['max_threads_block'] < 32:
        raise ValueError('live warp/block limit')
    functions = data['functions']
    if type(functions) is not list or len(functions) != len(bundle.cases):
        raise ValueError('missing/extra actual live function')
    indexed = {}
    for f in functions:
        exact(f, {'case','function','handle','registers','static_shared','local_bytes','max_threads','binary_version','ptx_version','active_blocks_per_sm','legal'}, 'actual live function resource')
        c = bundle.cases.get(f['case'])
        if c is None or c['id'] in indexed or f['function'] != c['function'] or not f['handle'] or f['legal'] is not True:
            raise ValueError('unknown/duplicate/illegal actual function')
        for name in ('registers','static_shared','local_bytes','max_threads','binary_version','ptx_version','active_blocks_per_sm'):
            integer(f[name], 0, 2**30, 'actual '+name)
        r = c['resources']
        if f['registers'] != r['registers_per_thread'] or f['static_shared'] != r['static_shared_bytes'] or f['local_bytes'] or f['binary_version'] != 120 or f['max_threads'] < 32 or f['active_blocks_per_sm'] < 1 or f['registers']*32 > min(limits['registers_sm'], limits['registers_block']):
            raise ValueError('actual function resources/occupancy differ from complete offline program')
        indexed[c['id']] = f
    if len({f['handle'] for f in functions}) != len(functions):
        raise ValueError('all function handles must remain distinct and live')
    return indexed


def validate_process(bundle, receipt):
    if receipt.get('status') != 'returned' or type(receipt.get('exit_code')) is not int or receipt['exit_code'] != 0:
        raise ValueError('failed/unstarted process is not a verified hardware observation')
    return validate_process_context(bundle,receipt)


def validate_process_context(bundle, receipt):
    """Close aborted process identity without promoting it to an observation."""
    exact(receipt, identity_fields(bundle,PROCESS_FIELDS), 'actual host process receipt')
    check_identity(bundle,receipt,'process-receipt')
    if receipt['kind'] != 'hardware' or receipt['status'] not in ('returned','failed','timeout') or type(receipt['exit_code']) is not int:
        raise ValueError('failed/unstarted/synthetic process is not a hardware observation')
    if receipt['registry_sha256'] != bundle.sha256 or receipt['phase'] not in ('structure','fit','heldout'):
        raise ValueError('process registry/phase differs')
    for n in ('pid','supervisor_pid'):
        integer(receipt[n], 1, 2**31, 'actual '+n)
    for n in ('started_realtime_ns','ended_realtime_ns','started_monotonic_ns','ended_monotonic_ns'):
        integer(receipt[n], 1, 2**64-1, n)
    if not 0 < receipt['ended_monotonic_ns']-receipt['started_monotonic_ns'] <= budget(bundle)['host_timeout_s' if receipt['status']=='returned' else 'total_window_s']*10**9 or receipt['ended_realtime_ns'] < receipt['started_realtime_ns']:
        raise ValueError('actual host timeout/order')
    from arch_explorer.components.accelsim.microexperiment import validate_design
    design, _ = json_file(original(receipt['design']))
    validate_design(bundle, design)
    if receipt['design_sha256'] != design['design_sha256'] or design['created_realtime_ns'] >= receipt['started_realtime_ns']:
        raise ValueError('design not frozen before device operation')
    phase = receipt['phase']
    if phase=='structure':
        if receipt['prediction_freeze'] is not None or receipt['freeze_sha256']!=design['design_sha256']:raise ValueError('structure registration differs from pre-device freeze')
    else:
        frozen,_=json_file(original(receipt['prediction_freeze']))
        check_identity(bundle,frozen,'prediction-freeze')
        if receipt['freeze_sha256']!=receipt['prediction_freeze']['sha256'] or frozen.get('stage')!=('initial' if phase=='fit' else 'final') or frozen.get('design_sha256')!=design['design_sha256'] or frozen.get('registry_sha256')!=bundle.sha256 or frozen.get('run_id')!=design['run_id'] or frozen['created_realtime_ns']>=receipt['started_realtime_ns']:raise ValueError('prediction freeze missing/wrong/after actual timing')
    if not same_json(receipt['host'], design['hosts']['capture' if phase == 'structure' else 'timing']):
        raise ValueError('actual executable differs from pre-device freeze')
    host = original(receipt['host'])
    lease_path = original(receipt['lease'])
    original(receipt['stdout']); original(receipt['stderr'])
    lease, _ = json_file(lease_path)
    from arch_explorer.components.accelsim.measurement_state import stateful,slots,sha_plan
    lease_extra={'plan_sha256'} if stateful(bundle) and design.get('state_protocol') else set()
    exact(lease, lease_extra|{'schema','run_id','work_item','design_sha256','registry_sha256','new_window','handoff','issued_realtime_ns','deadline_realtime_ns','device_ordinal','lock_file','budget'}, 'fresh coordinated lease')
    if lease['schema'] != schema(bundle,'lease') or lease['work_item'] != work_item(bundle) or lease['new_window'] is not True or lease['run_id'] != design['run_id'] or not same_json(lease['budget'], budget(bundle)) or type(lease['device_ordinal']) is not int or lease['device_ordinal'] != 0:
        raise ValueError('old/unknown lease or changed window budget')
    if lease_extra and lease.get('plan_sha256')!=sha_plan(bundle):raise ValueError('lease state plan differs')
    if lease['registry_sha256'] != bundle.sha256 or lease['design_sha256'] != receipt['design_sha256'] or not lease['issued_realtime_ns'] <= receipt['started_realtime_ns'] <= receipt['ended_realtime_ns'] <= lease['deadline_realtime_ns'] or not 0 < lease['deadline_realtime_ns']-lease['issued_realtime_ns'] <= budget(bundle)['total_window_s']*10**9:
        raise ValueError('process outside newly coordinated design/window')
    handoff, _ = json_file(original(lease['handoff']))
    if not design.get('state_protocol') and (handoff.get('schema') != schema(bundle,'resource-handoff') or handoff.get('work_item') != work_item(bundle) or handoff.get('run_id') != lease['run_id'] or handoff.get('resource_released') is not True or handoff.get('external_compute_empty') is not True):
        raise ValueError('fresh actual queue handoff absent; old RefUnit/terminal cannot authorize a window')
    if modern(bundle):
        validate_handoff(bundle,design,handoff)
        from arch_explorer.components.accelsim.state_resource import handoff_evidence
        for row in handoff_evidence(handoff):original(row)
        if lease['deadline_realtime_ns']>handoff['absolute_deadline_realtime_ns']-handoff['close_reserve_s']*10**9:raise ValueError('lease extends absolute reservation or cleanup reserve')
    exact(receipt['lock'], {'path','device','inode','fd','exclusive'}, 'inherited common-lock identity')
    if receipt['lock']['path'] != lease['lock_file'] or receipt['lock']['exclusive'] is not True:
        raise ValueError('common-lock identity missing')
    from arch_explorer.components.accelsim.state_idle import ADMISSION
    if handoff.get('schema')==ADMISSION:
        observation=json_file(original(handoff['observation']))[0]
        if not same_json(observation['lock'],receipt['lock']) or lease['issued_realtime_ns']<observation['ended_realtime_ns']:
            raise ValueError('idle observation/host lease lost the same held lock or time order')
    for k in ('device','inode','fd'):
        integer(receipt['lock'][k], 0, 2**64-1, 'common lock '+k)
    exact(receipt['budget_account'], {'process_index','case_ids','launch_quota'}, 'frozen process budget')
    account = receipt['budget_account']; ids = account['case_ids']
    order = ['structure'] + design['order']['fit'] + design['order']['heldout']
    integer(account['process_index'], 0, budget(bundle)['host_processes']-1, 'process budget index')
    expected = 'structure' if phase == 'structure' else ids[0] if isinstance(ids,list) and len(ids)==1 else None
    if order[account['process_index']] != expected or ids != (list(bundle.cases) if phase == 'structure' else [expected]) or account['launch_quota'] != (len(bundle.cases) if phase == 'structure' else len(slots(bundle,expected)) if stateful(bundle) and design.get('state_protocol') else budget(bundle)['samples_per_case']) or (phase != 'structure' and bundle.cases[expected]['partition'] != phase):
        raise ValueError('attempt phase/order or budget differs from registered partition')
    argv = receipt['command']; env = receipt['environment']
    full_env=receipt['launch_environment']
    if type(full_env) is not dict or env!={k:full_env.get(k) for k in ENV_KEYS} or full_env.get('LC_ALL')!='C':raise ValueError('actual launch environment is not closed')
    if not isinstance(argv,list) or not argv or Path(argv[0]).resolve() != host or type(env) is not dict or set(env) != ENV_KEYS or env.get('CUDA_VISIBLE_DEVICES') != '0':
        raise ValueError('actual host executable/environment identity differs')
    options = dict(zip(argv[1::2], argv[2::2])) if len(argv)%2 == 1 else {}
    expected_options = {'--bundle':str(bundle.path.parent),'--output':receipt['output'],'--lease':str(lease_path),'--lease-sha':receipt['lease']['sha256'],'--freeze':receipt['freeze_sha256'],'--lock-file':lease['lock_file'],'--lock-fd':str(receipt['lock']['fd'])}
    if phase != 'structure': expected_options['--case'] = ids[0]
    if stateful(bundle) and design.get('state_protocol'):
        expected_options.update({'--run':design['run_id'],'--start-ns':str(receipt['started_realtime_ns']),'--deadline-ns':str(min(lease['deadline_realtime_ns'],receipt['started_realtime_ns']+budget(bundle)['host_timeout_s']*10**9))})
    if options != expected_options or len(argv) != 1+2*len(expected_options):
        raise ValueError('actual command flags differ from closed registration')
    if phase == 'structure':
        origin,_=json_file(original(design['source']))
        if full_env.get('NVDISASM')!=str(original(origin['runtime_tools']['nvdisasm'])) or env.get('LD_PRELOAD') != design['tracer']['path'] or limits_missing_capture_design(design):
            raise ValueError('capture tool differs from pre-device freeze')
    elif any(full_env.get(k) for k in ('LD_PRELOAD','LD_AUDIT','CUDA_INJECTION64_PATH','CUDA_INJECTION32_PATH','NVBIT_TOOL','NVBIT_TOOL_PATH','CUDA_PROFILE','CUDA_ENABLE_PROFILING','CUDA_MPS_PIPE_DIRECTORY','CUDA_MPS_ACTIVE_THREAD_PERCENTAGE')) or 'nvbit' in (env.get('LD_LIBRARY_PATH') or '').lower():
        raise ValueError('timing process loads instrumentation/profiling/MPS')
    return lease


def limits_missing_capture_design(design):
    # Tool identity is checked by bytes, not by importing a GPU library for info.
    return not original(design['tracer']).is_file()


def validate_sample(bundle, case, sample, mode):
    exact(sample, SAMPLE_FIELDS, 'complete raw hardware sample')
    if sample['schema'] != 'arch-explorer.sim30-host-sample.v1' or sample['kind'] != 'hardware' or sample['mode'] != mode or sample['status'] != 'returned' or sample['correct'] is not True:
        raise ValueError('synthetic/failed/wrong-mode sample cannot become hardware timing')
    expected = {'registry_sha256':bundle.sha256,'case':case['id'],'family':case['family'],'partition':case['partition'],'steps':case['steps'],'function':case['function'],'cubin_sha256':bundle.document['cubin']['sha256'],'text_sha256':case['text_sha256'],'input_sha256':case['input']['sha256'],'auxiliary_sha256':case['auxiliary']['sha256'],'expected_sha256':case['expected']['sha256'],'grid':case['grid'],'block':case['block'],'dynamic_shared':0,'stream':0}
    if any(not same_json(sample[k], v) for k,v in expected.items()):
        raise ValueError('raw function/cubin/input/partition/geometry binding differs')
    integer(sample['pid'], 1, 2**31, 'sample PID')
    integer(sample['sample_index'], 0, 2 if mode == 'timing' else 0, 'sample index')
    for key in ('launch_return','synchronize_return','hidden_warmup_launches','helper_launches'):
        if type(sample[key]) is not int or sample[key] != 0: raise ValueError('failed/extra/hidden launch')
    if sample['all_function_handles_live'] is not True or sample['hardware_core_mhz'] is not None or sample['clock_state'] != 'unknown' or sample['cache_state'] != 'uncontrolled; sample order retained':
        raise ValueError('unsupported resource/physical clock/cache claim')
    if mode == 'structure':
        if sample['unit'] != 'not_timed' or any(sample[k] is not None for k in ('elapsed_ns','event_elapsed_ms','event_elapsed_ms_float32_bits','start_event','end_event')):
            raise ValueError('instrumented structure cannot be consumed as timing')
        return None
    if mode != 'timing' or sample['unit'] != 'ns' or not sample['start_event'] or not sample['end_event'] or sample['start_event'] == sample['end_event']:
        raise ValueError('raw event unit/actual handles differ')
    bits = integer(sample['event_elapsed_ms_float32_bits'], 0, 2**32-1, 'original float32 event bits')
    ms = struct.unpack('<f', struct.pack('<I', bits))[0]
    if not math.isfinite(ms) or ms <= 0 or type(sample['event_elapsed_ms']) not in (int,float) or sample['event_elapsed_ms'] != ms or type(sample['elapsed_ns']) not in (int,float) or sample['elapsed_ns'] != float(ms)*1e6:
        raise ValueError('nonfinite/zero event or ms/bits/ns conversion mismatch')
    return sample['elapsed_ns']


def case_files(output, cid, index, supervisor):
    root = Path(output).resolve(); d = root/cid; p = 'sample-'+str(index)
    return dict(sample=d/(p+'.json'), output=d/(p+'.actual.bin'), started=d/(p+'.started.json'), allocation=d/'allocation.json', launch=d/'launch.json', preflight=root/'preflight.json', invocation=root/'invocation.json', command=root/'command.json', environment=root/'environment.json', maps_before=root/'maps-before-cuInit.txt', maps_after=root/'maps-before-close.txt', closed=root/'closed.json', supervisor=Path(supervisor).resolve(), **({'event':d/(p+'.event.json')} if (d/(p+'.event.json')).exists() else {}))


def validate_loader_environment(mode, launched, observed, maps, tracer):
    """Keep launch and in-process observations distinct, with mapped-byte proof.

    NVBit initialization can clear LD_PRELOAD for child tools. Only a capture
    process with the frozen tracer mapped before/after execution may report that
    one removal. Timing has no such allowance and cannot hide instrumentation.
    """
    exact(launched, ENV_KEYS, 'supervisor loader variables')
    exact(observed, ENV_KEYS, 'in-process loader variables')
    differences={k for k in ENV_KEYS if not same_json(launched[k],observed[k])}
    if not differences:return
    if mode!='structure' or differences!={'LD_PRELOAD'} or launched['LD_PRELOAD']!=tracer['path'] or observed['LD_PRELOAD'] is not None:
        raise ValueError('actual command/environment differs from supervisor')
    tool=original(tracer)
    if len(maps)!=2 or any(not any(line.split()[-1:]==[str(tool)] for line in text.splitlines()) for text in maps):
        raise ValueError('cleared capture preload requires exact frozen mapped tracer before and after')


def validate_case_attempt(bundle, case, files, mode):
    sample, _ = json_file(files['sample']); process, _ = json_file(files['supervisor'])
    lease = validate_process(bundle, process); elapsed = validate_sample(bundle, case, sample, mode)
    expected_files = case_files(process['output'], case['id'], sample['sample_index'], files['supervisor'])
    for role,p in expected_files.items():
        if role not in files or Path(files[role]).resolve() != p: raise ValueError('original role outside actual host output: '+role)
    pre, _ = json_file(files['preflight']); inv, _ = json_file(files['invocation'])
    resources = validate_preflight(bundle, pre)
    exact(inv, identity_fields(bundle,INV_FIELDS), 'actual host invocation')
    check_identity(bundle,inv,'host-invocation')
    for row in (sample,pre,inv):
        if row['pid'] != process['pid']: raise ValueError('actual host PID differs')
    if sample['context'] != pre['context'] or sample['freeze_sha256'] != process['freeze_sha256'] or sample['lease_sha256'] != process['lease']['sha256']:
        raise ValueError('raw sample process/context/freeze/lease mismatch')
    if inv['registry_sha256'] != bundle.sha256 or inv['host_sha256'] != process['host']['sha256'] or inv['mode'] != mode or inv['lease_file'] != process['lease']['path'] or inv['lease_sha256'] != sample['lease_sha256'] or inv['freeze_sha256'] != sample['freeze_sha256'] or inv['common_lock'] != process['lock']['path'] or inv['lock_fd'] != process['lock']['fd']:
        raise ValueError('actual host invocation binding differs')
    if inv['parent_pid'] != process['supervisor_pid'] or pre['parent_pid'] != process['supervisor_pid']:
        raise ValueError('actual parent supervisor differs')
    for row in (inv,pre):
        if not process['started_realtime_ns'] <= row['host_realtime_ns'] <= process['ended_realtime_ns']: raise ValueError('host timestamp outside supervisor interval')
    if not process['started_monotonic_ns'] <= inv['monotonic_ns'] <= pre['host_monotonic_ns'] <= process['ended_monotonic_ns']: raise ValueError('live preflight timing order differs')
    if mode == 'structure' and (not pre['limits']['managed_memory'] or not pre['limits']['concurrent_managed_access'] or not 0 < pre['driver_branch'] <= 595):
        raise ValueError('NVBit managed memory/driver compatibility blocked')
    if case['id'] not in process['budget_account']['case_ids'] or (mode == 'structure') != (process['phase'] == 'structure'): raise ValueError('case not scheduled in actual process phase')
    cmd, _ = json_file(files['command']); exact(cmd, {'schema','argv','cwd'}, 'actual host command')
    env, _ = json_file(files['environment']); exact(env, {'schema','scope','values'}, 'actual loader environment')
    if cmd['schema'] != 'arch-explorer.sim30-host-command.v1' or cmd['argv'] != process['command'] or not Path(cmd['cwd']).is_absolute() or env['schema'] != 'arch-explorer.sim30-loader-environment.v1': raise ValueError('actual command/environment differs from supervisor')
    design,_=json_file(original(process['design']))
    map_texts=[Path(files[k]).read_text() for k in ('maps_before','maps_after')]
    validate_loader_environment(mode,process['environment'],env['values'],map_texts,design['tracer'])
    for role in ('maps_before','maps_after'):
        maps = Path(files[role]).read_text()
        if not maps.strip() or 'libcuda.so' not in maps: raise ValueError('actual loaded CUDA library maps absent')
        if mode == 'timing' and any(x in maps.lower() for x in ('nvbit','sim30_capture','sim28_capture','libcupti','nsight')): raise ValueError('timing maps contain instrumentation')
    closed, _ = json_file(files['closed'])
    exact(closed, {'schema','status','application_launches','planned_application_launches','unstarted_application_launches','all_module_context_handles_destroyed','retry_allowed'}, 'actual host close')
    quota = process['budget_account']['launch_quota']
    if not same_json(closed, {'schema':'arch-explorer.sim30-host-close.v1','status':'returned','application_launches':quota,'planned_application_launches':quota,'unstarted_application_launches':0,'all_module_context_handles_destroyed':True,'retry_allowed':False}): raise ValueError('host launch denominator/cleanup differs')
    actual = Path(files['output']).read_bytes(); main, aux, expected = bundle.inputs(case)
    if actual != expected or file_digest(files['output']) != sample['output_sha256'] or oracle(case,main,aux) != actual: raise ValueError('actual full output differs from independent frozen numeric oracle')
    allocation, _ = json_file(files['allocation']); launch, _ = json_file(files['launch'])
    exact(allocation, ALLOCATION, 'actual allocation record')
    if allocation['schema'] != 'arch-explorer.sim30-allocation.v1' or launch != dict(allocation,schema='arch-explorer.sim30-launch.v1') or any(not same_json(allocation[k], sample[k]) for k in BASE): raise ValueError('actual allocation/launch sample identity differs')
    d = Path(files['sample']).parent; count = 1 if mode == 'structure' else 3
    expected_names={'input.bin','auxiliary.bin','expected.bin','output.initial.bin','allocation.json','launch.json'}
    expected_names|={f'sample-{i}.'+tail for i in range(count) for tail in ('started.json','actual.bin','json')}
    expected_names|=({f'sample-{i}.event.json' for i in range(count)} if mode=='timing' else {'records.bin','static.json','static.bin','capture.json'})
    if {p.name for p in d.iterdir()}!=expected_names:raise ValueError('missing/extra original case files or duplicate/extra samples')
    paths = {'input_file':d/'input.bin','auxiliary_file':d/'auxiliary.bin','expected_file':d/'expected.bin','output_initial_file':d/'output.initial.bin'}
    if any(allocation[k] != str(v) for k,v in paths.items()) or allocation['actual_output_files'] != [str(d/(f'sample-{i}.actual.bin')) for i in range(count)] or allocation['correct'] is not None or allocation['output_status'] != 'awaiting_registered_launch': raise ValueError('actual consumed input/output path contract differs')
    if any(p.read_bytes() != v for p,v in zip(paths.values(),(main,aux,expected,b'\xa5'*len(expected)))): raise ValueError('actual consumed input/expected bytes differ')
    started, _ = json_file(files['started'])
    exact(started, BASE | EVENTS | {'schema','sample_index','host_realtime_ns','timed','status'}, 'actual launch started record')
    if any(not same_json(started[k],sample[k]) for k in BASE | EVENTS | {'sample_index'}) or started['schema'] != 'arch-explorer.sim30-started.v1' or started['timed'] != (mode=='timing') or started['status'] != 'about_to_attempt_launch' or not pre['host_realtime_ns'] <= started['host_realtime_ns'] <= process['ended_realtime_ns']: raise ValueError('actual launch start differs')
    if mode == 'timing':
        if 'event' not in files: raise ValueError('original event API record missing')
        event, _ = json_file(files['event'])
        exact(event, BASE | EVENTS | {'schema','sample_index','event_api_return','event_elapsed_ms_float32_bits','event_elapsed_ms'}, 'raw event API record')
        if any(not same_json(event[k],sample[k]) for k in BASE | EVENTS | {'sample_index','event_elapsed_ms_float32_bits','event_elapsed_ms'}) or event['schema'] != 'arch-explorer.sim30-event-raw.v1' or type(event['event_api_return']) is not int or event['event_api_return'] != 0: raise ValueError('original event API/sample differ')
    return {'kind':'hardware','status':'verified','elapsed_ns':elapsed,'sample':sample,'resource':resources[case['id']],'allocation':allocation,'process':process,'lease':lease,'hardware_accuracy':'not validated'}


def attempt_reference(bundle, cid, output, index, supervisor):
    """Serializable originals, never a free-form promoted hardware data row."""
    from arch_explorer.components.accelsim.microexperiment import ref
    files = case_files(output,cid,index,supervisor)
    return {'schema':'arch-explorer.sim30-observation-originals.v1','case':cid,'mode':'timing','files':{k:ref(v) for k,v in files.items()}}


def load_attempt(bundle, row):
    exact(row, {'schema','case','mode','files'}, 'original observation references')
    if row['schema'] != 'arch-explorer.sim30-observation-originals.v1' or row['mode'] != 'timing' or row['case'] not in bundle.cases:
        raise ValueError('only actual registered timing originals accepted')
    exact(row['files'],{'sample','output','started','allocation','launch','preflight','invocation','command','environment','maps_before','maps_after','closed','supervisor','event'},'complete original timing roles')
    return validate_case_attempt(bundle,bundle.cases[row['case']],{k:original(v) for k,v in row['files'].items()},'timing')
