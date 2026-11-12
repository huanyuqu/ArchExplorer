# Concrete Accel-Sim implementation; owned by the component provider.
"""Completion and time accounting for one actual native kernel run."""
import re


def parse_stats(stdout, stderr, expected, core_mhz):
    combined = stdout + '\n' + stderr
    if re.search(r'^(?:GPGPU-Sim[^\n]*(?:DEADLOCK|break due to reaching)|.*Assertion .*failed|.*Segmentation fault|Aborted|deadlock detected)', combined, re.I | re.M):
        raise ValueError('accelsim native run failed/truncated/deadlocked')
    if stdout.count('GPGPU-Sim: *** exit detected ***') != 1 or stdout.count('GPGPU-Sim: *** simulation thread exiting ***') != 1:
        raise ValueError('accelsim missing/duplicate native completion markers')
    launches = re.findall(r'^launching kernel name: (.+?) uid: (\d+) cuda_stream_id: (\d+)\s*$', stdout, re.M)
    names = re.findall(r'^kernel_name = (.*?)\s*$', stdout, re.M)
    uids = re.findall(r'^kernel_launch_uid = (.*?)\s*$', stdout, re.M)
    if len(launches) != 1 or names != [expected['name']] or uids != [launches[0][1]] or launches[0][0] != expected['name'] or launches[0][2] != '0':
        raise ValueError('accelsim missing/duplicate/unexpected kernel or stream statistics')
    metrics = {}
    for key in ['gpu_sim_cycle', 'gpu_tot_sim_cycle', 'gpu_sim_insn', 'gpu_tot_sim_insn', 'gpu_tot_issued_cta']:
        rows = re.findall(rf'^{key}\s*=\s*(\d+)\s*$', stdout, re.M)
        if len(rows) != 1:
            raise ValueError(f'accelsim missing/duplicate native statistic {key}')
        metrics[key] = int(rows[0])
    if metrics['gpu_tot_sim_cycle'] <= 0 or metrics['gpu_sim_cycle'] != metrics['gpu_tot_sim_cycle']:
        raise ValueError('accelsim invalid single-kernel versus cumulative cycle accounting')
    if metrics['gpu_sim_insn'] != expected['thread_instructions'] or metrics['gpu_tot_sim_insn'] != expected['thread_instructions'] or metrics['gpu_tot_issued_cta'] != expected['ctas']:
        raise ValueError('accelsim incomplete retired instructions or issued CTA count')
    if 'warp_instructions' in expected:
        rows = re.findall(r'^gpgpu_n_tot_w_icount\s*=\s*(\d+)\s*$', stdout, re.M)
        if len(rows) != 1 or int(rows[0]) != expected['warp_instructions']:
            raise ValueError('accelsim incomplete/duplicate retired warp instructions')
        metrics['gpgpu_n_tot_w_icount'] = int(rows[0])
    periods = re.findall(r'^GPGPU-Sim uArch: clock periods: ([0-9.eE+-]+):', stdout, re.M)
    if len(periods) != 1 or abs(float(periods[0]) - 1 / (core_mhz * 1_000_000)) > 1e-19:
        raise ValueError('accelsim native core clock disagrees with compiled ADF')
    # gpu-sim.cc:1340-1354 converts MHz to Hz; :1713-1721 prints the
    # cumulative simulated *core* cycles. Never sum cumulative counters.
    return {**metrics, 'kernel': names[0], 'uid': int(uids[0]),
            'completion': 'inferred from complete trace EXIT closure, exact retired instruction counts, issued CTA count, normal no-limit termination',
            'cta_counter_semantics': 'gpu_tot_issued_cta counts issued CTAs; no directly printed completed-CTA counter',
            'latency_ns': metrics['gpu_tot_sim_cycle'] * 1000 / core_mhz,
            'clock_domain': 'core', 'core_mhz': core_mhz,
            'formula': 'final gpu_tot_sim_cycle * 1000 / core_clock_MHz',
            'scope': 'single-kernel device core cycles including configured native launch latency; no host/copies'}
