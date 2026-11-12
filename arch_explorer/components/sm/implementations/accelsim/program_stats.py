# Concrete Accel-Sim implementation; owned by the component provider.
"""One complete ordered program's native cumulative counters."""
import math
import re


def parse_program_stats(stdout, stderr, expected, core_mhz):
    if not expected or not math.isfinite(core_mhz) or core_mhz <= 0:
        raise ValueError('complete program and positive native core clock required')
    combined = stdout+'\n'+stderr
    if re.search(r'^(?:GPGPU-Sim[^\n]*(?:DEADLOCK|break due to reaching)|.*Assertion .*failed|.*Segmentation fault|Aborted|deadlock detected)', combined, re.I|re.M):
        raise ValueError('native program failed, truncated or reached its cycle limit')
    if any(stdout.count(marker) != 1 for marker in ('GPGPU-Sim: *** exit detected ***', 'GPGPU-Sim: *** simulation thread exiting ***')):
        raise ValueError('native program completion markers absent or duplicated')
    launches = re.findall(r'^launching kernel name: (.+?) uid: (\d+) cuda_stream_id: (\d+)\s*$', stdout, re.M)
    names = re.findall(r'^kernel_name = (.*?)\s*$', stdout, re.M)
    uids = re.findall(r'^kernel_launch_uid = (\d+)\s*$', stdout, re.M)
    count = len(expected)
    if names != [e['name'] for e in expected] or len(launches) != count or len(uids) != count:
        raise ValueError('native program kernel set/order differs from complete trace list')
    if any(n != e['name'] or stream != '0' or uid != uids[i] for i, ((n, uid, stream), e) in enumerate(zip(launches, expected))) or len(set(uids)) != count:
        raise ValueError('native program launch identity/stream differs')
    metrics = {}
    for key in ('gpu_sim_cycle', 'gpu_tot_sim_cycle', 'gpu_sim_insn', 'gpu_tot_sim_insn', 'gpu_tot_issued_cta', 'gpgpu_n_tot_w_icount'):
        rows = re.findall(r'^'+key+r'\s*=\s*(\d+)\s*$', stdout, re.M)
        if len(rows) != count: raise ValueError('native program missing/duplicate statistic '+key)
        metrics[key] = list(map(int, rows))
    cumulative = {'cycles':0, 'instructions':0, 'ctas':0, 'warps':0}
    kernels = []
    for i, e in enumerate(expected):
        cycles = metrics['gpu_sim_cycle'][i]
        cumulative['cycles'] += cycles
        cumulative['instructions'] += e['thread_instructions']
        cumulative['ctas'] += e['ctas']; cumulative['warps'] += e['warp_instructions']
        if (cycles <= 0 or metrics['gpu_tot_sim_cycle'][i] != cumulative['cycles']
            or metrics['gpu_sim_insn'][i] != e['thread_instructions']
            or metrics['gpu_tot_sim_insn'][i] != cumulative['instructions']
            or metrics['gpu_tot_issued_cta'][i] != cumulative['ctas']
            or metrics['gpgpu_n_tot_w_icount'][i] != cumulative['warps']):
            raise ValueError('incomplete native retirement/CTA/cumulative accounting at '+e['name'])
        kernels.append({'name':e['name'],'uid':int(uids[i]),'cycles':cycles,
                        'cumulative_cycles':cumulative['cycles'],'thread_instructions':e['thread_instructions'],
                        'ctas':e['ctas'],'warp_instructions':e['warp_instructions']})
    periods = re.findall(r'^GPGPU-Sim uArch: clock periods: ([0-9.eE+-]+):', stdout, re.M)
    if len(periods) != 1 or abs(float(periods[0])-1/(core_mhz*1e6)) > 1e-19:
        raise ValueError('native program clock disagrees with compiled ADF')
    return {'kernels':kernels,'kernel_count':count,'gpu_tot_sim_cycle':cumulative['cycles'],
            'gpu_tot_sim_insn':cumulative['instructions'],'gpu_tot_issued_cta':cumulative['ctas'],
            'gpgpu_n_tot_w_icount':cumulative['warps'],'core_mhz':core_mhz,'clock_domain':'core',
            'latency_ns':cumulative['cycles']*1000/core_mhz,
            'formula':'final gpu_tot_sim_cycle * 1000 / core_clock_MHz',
            'scope':'complete ordered stream in one native process, including configured launch latency; host/copies excluded',
            'completion':'all expected kernels, EXIT closure and exact instruction/issued-CTA counters, normal no-limit termination'}
