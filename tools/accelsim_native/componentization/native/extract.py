"""Deterministic SIM38 extraction from the pinned typed-v1 native source.

Only a new private checkout may be passed. The original cycle body is retained
for the oracle. Component phase bodies are selected from that same body, with
the input SHA checked before any write. Generated files are build artifacts;
no upstream full source is distributed by this package.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import shutil
from peek import patch_peek

EXPECTED = {
    'gpu-sim.cc': 'fd528a420be6bb93199e3c5ecbf48c9f57dc1e5e331c8f017477aee6d2be00e1',
    'gpu-sim.h': 'c4f52ce5f4aacd8506f4d08cbab8240aa6e4349738ce72728e2f04fc6b351476',
    'l2cache.cc': '1b2b0ebca49a2f5943c9350a637d849961da96159a147a70b887434bd7f37ff5',
    'l2cache.h': '724164b9a9e97b408781cccf8efccd94c56c3cf79667941b2ea8a3105cfb7890',
    'mem_fetch.cc': '9fa9075185675555905d669d8cd3aee04a797e7986b7a992d935ee75a8e1eec8',
    'dram.h': 'a7281038aa2020ad324697152b3f418eac6f8e737a2f637631244ab17e6f525c',
}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def replace_once(text, old, new):
    if text.count(old) != 1:
        raise ValueError(f'expected one patch anchor: {old[:100]!r}')
    return text.replace(old, new, 1)


def block(text, marker):
    """Return offsets and body of the brace block following a unique marker."""
    if text.count(marker) != 1:
        raise ValueError(f'non-unique extraction anchor: {marker}')
    start = text.index('{', text.index(marker) + len(marker))
    depth = 0
    tokens = re.finditer(r'//[^\n]*|/\*.*?\*/|"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'|[{}]',
                         text[start:], re.S)
    for item in tokens:
        token = item.group()
        if token == '{':
            depth += 1
        elif token == '}':
            depth -= 1
            if not depth:
                end = start + item.end()
                return start, end, text[start + 1:end - 1]
    raise ValueError('unterminated native block')


def extract(root: Path, component_source: Path, runtime: Path):
    gp = root / 'gpgpusim/src/gpgpu-sim'
    ac = root / 'accelsim/gpu-simulator'
    source = {}
    for name, sha in EXPECTED.items():
        data = (gp / name).read_bytes()
        if digest(data) != sha:
            raise ValueError(f'pinned typed-v1 source mismatch: {name}')
        source[name] = data.decode()
    shader = (gp / 'shader.cc').read_text()
    sh = (gp / 'shader.h').read_text()
    cc = source['gpu-sim.cc']
    _, _, cycle = block(cc, 'void gpgpu_sim::cycle()')
    original_cycle_sha = digest(cycle.encode())
    # Every extraction is from the original body. No guessed model formulas.
    reply_start = cycle.index('    // pop from L2 to interconnect')
    reply_if = cycle[reply_start:cycle.index('  partiton_replys_in_parallel +=')]
    _, _, reply = block(reply_if, 'for (unsigned i = 0; i < m_memory_config->m_n_mem_sub_partition; i++)')
    _, _, dram_if = block(cycle, 'if (clock_mask & DRAM)')
    _, _, dram = block(dram_if, 'for (unsigned i = 0; i < m_memory_config->m_n_mem; i++)')
    cache_start = cycle.index('    m_power_stats->pwr_mem_stat->l2_cache_stats')
    cache_if = cycle[cache_start:cycle.index('  partiton_reqs_in_parallel +=')]
    _, _, cache = block(cache_if, 'for (unsigned i = 0; i < m_memory_config->m_n_mem_sub_partition; i++)')
    # accept() already performs the destination queue write on this path.
    cache = replace_once(cache, '        m_memory_sub_partition[i]->push(mf, gpu_sim_cycle + gpu_tot_sim_cycle);', '')
    cache_end = cycle[cycle.index('  partiton_reqs_in_parallel +='):cycle.index('  if (clock_mask & ICNT) {\n    icnt_transfer();')]
    _, _, core = block(cycle, 'if (clock_mask & CORE)')
    a, b, sm = block(core[:core.index('    float temp =')], 'for (unsigned i = 0; i < m_shader_config->n_simt_clusters; i++)')
    # That first loop is the actual per-cluster advance/statistics. Remaining
    # core tail remains byte-identical, including CTA issue, latency and flush.
    core_tail = core[b:]
    phases = {
        'ae_l2_reply(unsigned i, unsigned& partiton_replys_in_parallel_per_cycle)': reply,
        'ae_dram(unsigned i)': dram,
        'ae_l2_cache(unsigned i, unsigned& partiton_reqs_in_parallel_per_cycle)': cache,
        'ae_l2_cache_end(unsigned partiton_reqs_in_parallel_per_cycle)': cache_end,
        'ae_sm(unsigned i)': sm,
        'ae_core_tail()': core_tail,
    }
    generated = cc[:cc.index('#include')] + '\n// Generated from pinned native cycle; see EXTRACTION.json.\n'
    for signature, body in phases.items():
        generated += f'void gpgpu_sim::{signature} {{\n{body}\n}}\n'
    generated += '''void gpgpu_sim::ae_l2_cache_begin() {
  m_power_stats->pwr_mem_stat->l2_cache_stats[CURRENT_STAT_IDX].clear();
}
void gpgpu_sim::ae_core_begin() {
  m_power_stats->pwr_mem_stat->core_cache_stats[CURRENT_STAT_IDX].clear();
}
void gpgpu_sim::ae_l2_reply_end(unsigned replies) {
  partiton_replys_in_parallel += replies;
}
'''
    # Keep the complete original body in place, adding a counter only.
    cc = replace_once(cc, 'void gpgpu_sim::cycle() {',
                      'void gpgpu_sim::cycle() {\n  ae_accelsim::original_cycle();')
    cc += '\n#include "ae_component/phases.inc"\n'
    cc = '#include "ae_component/hooks.hpp"\n' + cc
    header = source['gpu-sim.h']
    header = 'namespace ae_accelsim { class Bridge; }\n' + header
    header = replace_once(header, '  friend class gpgpu_sim;', '  friend class gpgpu_sim;\n  friend class ae_accelsim::Bridge;')
    decl = '\n'.join('  void ' + sig + ';' for sig in phases)
    header = replace_once(header, '  void init();\n  void cycle();', '''  void init();
  void cycle();
  friend class ae_accelsim::Bridge;
  void ae_l2_cache_begin();
  void ae_core_begin();
  void ae_l2_reply_end(unsigned replies);
''' + decl)
    # L2->DRAM acceptance owns the latency queue and native credit update.
    lcc = source['l2cache.cc']
    marker = '      dram_delay_t d;'
    start = lcc.index(marker, lcc.index('void memory_partition_unit::dram_cycle()'))
    stop = lcc.index('      break;', start)
    enqueue = lcc[start:stop]
    # The simple and detailed paths share exactly this admission operation.
    if lcc.count(enqueue) != 2:
        raise ValueError('DRAM detailed admission anchor changed')
    _, _, simple = block(lcc, 'void memory_partition_unit::simple_dram_model_cycle()')
    simple_enqueue_start = simple.index(marker)
    simple_enqueue_end = simple.index('      processed++;', simple_enqueue_start)
    if simple[simple_enqueue_start:simple_enqueue_end] != enqueue:
        raise ValueError('simple/detailed DRAM admission diverged')
    old = '      m_sub_partition[spid]->L2_dram_queue_pop();'
    # Wrap the native pop/enqueue in both original functions. On component
    # path route/accept writes the DRAM latency queue; pop only after success.
    for function in ('simple_dram_model_cycle()', 'dram_cycle()'):
        a, b, body = block(lcc, 'void memory_partition_unit::' + function)
        body = replace_once(body, old, '''      if (ae_accelsim::components()) {
        if (!ae_accelsim::dram_request(this, spid, mf)) continue;
        m_sub_partition[spid]->L2_dram_queue_pop();
      } else {
''' + '        ae_accelsim::dram_request(this, spid, mf);\n' + old)
        body = replace_once(body, enqueue, enqueue + '      }\n')
        lcc = lcc[:a + 1] + body + lcc[b - 1:]
    lcc += '\nvoid memory_partition_unit::ae_accept_dram(unsigned spid, mem_fetch* mf) {\n' + enqueue + '}\n'
    lcc = replace_once(lcc, '  m_dram_L2_queue->push(mf);\n}', '''  if (ae_accelsim::components()) {
    ae_accelsim::dram_reply(this, mf);
  } else {
    ae_accelsim::dram_reply(this, mf);
    m_dram_L2_queue->push(mf);
  }
}''')
    lcc = '#include "ae_component/hooks.hpp"\n' + lcc
    lch = 'namespace ae_accelsim { class Bridge; }\n' + source['l2cache.h']
    for name in ('memory_partition_unit', 'memory_sub_partition'):
        lch = replace_once(lch, 'class ' + name + ' {\n public:',
                           'class ' + name + ' {\n  friend class ae_accelsim::Bridge;\n public:')
    lch = replace_once(lch, '  void dram_cycle();',
                       '  void dram_cycle();\n  void ae_accept_dram(unsigned spid, mem_fetch* mf);')
    # SM destination admission is the original receive code, moved as a unit.
    sm_start = shader.index('    assert(mf->get_tpc() == m_cluster_id);', shader.index('void simt_core_cluster::icnt_cycle()'))
    sm_end_marker = '    m_stats->n_mem_to_simt[m_cluster_id] += mf->get_num_flits(false);'
    sm_end = shader.index(sm_end_marker, sm_start) + len(sm_end_marker)
    accept = shader[sm_start:sm_end]
    shader = shader[:sm_start] + '    if (ae_accelsim::components()) return;\n' + accept + shader[sm_end:]
    shader += '\nvoid simt_core_cluster::ae_accept_reply(mem_fetch* mf) {\n' + accept + '\n}\n'
    shader = '#include "ae_component/hooks.hpp"\n' + shader
    sh = 'namespace ae_accelsim { class Bridge; }\n' + sh
    sh = replace_once(sh, 'class simt_core_cluster {\n public:',
                      'class simt_core_cluster {\n  friend class ae_accelsim::Bridge;\n public:\n  void ae_accept_reply(mem_fetch* mf);')
    # Destruction is the native resource release, never a guessed timestamp.
    mfcc = '#include "ae_component/hooks.hpp"\n' + source['mem_fetch.cc']
    mfcc = replace_once(mfcc, 'mem_fetch::~mem_fetch() { m_status = MEM_FETCH_DELETED; }',
                       'mem_fetch::~mem_fetch() { ae_accelsim::destroyed(m_request_uid); m_status = MEM_FETCH_DELETED; }')
    drh = 'namespace ae_accelsim { class Bridge; }\n' + source['dram.h']
    drh = replace_once(drh, 'class dram_t {\n public:',
                       'class dram_t {\n  friend class ae_accelsim::Bridge;\n public:')
    for name, text in {'gpu-sim.cc':cc, 'gpu-sim.h':header, 'l2cache.cc':lcc,
                       'l2cache.h':lch, 'shader.cc':shader, 'shader.h':sh,
                       'mem_fetch.cc':mfcc, 'dram.h':drh}.items():
        (gp / name).write_text(text)
    inc = gp / 'ae_component'
    inc.mkdir()
    (inc / 'phases.inc').write_text(generated)
    shutil.copy2(component_source / 'hooks.hpp', inc / 'hooks.hpp')
    shutil.copy2(component_source / 'adapter.cc', gp / 'ae_component.cc')
    shutil.copy2(runtime, inc / 'runtime.hpp')
    # This is the only trace frontend dispatch change. Oracle still calls the
    # original cycle; component mode calls the generic Runtime tick.
    asc = (ac / 'accel-sim.cc').read_text()
    asc = '#include "gpgpu-sim/ae_component/hooks.hpp"\n' + asc
    asc = replace_once(asc, '      m_gpgpu_sim->cycle();', '''      if (ae_accelsim::components()) ae_accelsim::step(m_gpgpu_sim);
      else m_gpgpu_sim->cycle();''')
    asc = replace_once(asc, '      cleanup(finished_kernel_uid);',
                       '      cleanup(finished_kernel_uid);\n      ae_accelsim::kernel(m_gpgpu_sim, finished_kernel_uid);')
    (ac / 'accel-sim.cc').write_text(asc)
    main = (ac / 'main.cc').read_text()
    main = '#include "gpgpu-sim/ae_component/hooks.hpp"\n' + main
    main = replace_once(main, '  accel_sim_framework accel_sim(argc, argv);',
                        '  ae_accelsim::configure(argc, argv);\n  accel_sim_framework accel_sim(argc, argv);\n  ae_accelsim::attach(accel_sim.get_gpgpu_sim());')
    main = replace_once(main, '  accel_sim.simulation_loop();', '''  if (ae_accelsim::memory_source()) ae_accelsim::run_memory_source(accel_sim.get_gpgpu_sim());
  else accel_sim.simulation_loop();
  ae_accelsim::finish(accel_sim.get_gpgpu_sim());''')
    (ac / 'main.cc').write_text(main)
    peek_outputs=patch_peek(root/'gpgpusim/src')
    receipt = {'schema':'arch-explorer.accelsim-native-extraction.v1',
               'checked_inputs':EXPECTED, 'original_cycle_body_sha256':original_cycle_sha,
               'phases':{sig:digest(body.encode()) for sig,body in phases.items()},
               'runtime_sha256':digest(runtime.read_bytes()), 'peek_outputs':peek_outputs,
               'generated':{str(f.relative_to(root)):digest(f.read_bytes())
                            for f in [gp/n for n in ['gpu-sim.cc','gpu-sim.h','l2cache.cc','l2cache.h','shader.cc','shader.h','mem_fetch.cc','dram.h','ae_component.cc','ae_component/phases.inc','ae_component/hooks.hpp','ae_component/runtime.hpp']] + [ac/'main.cc',ac/'accel-sim.cc']}}
    (root/'EXTRACTION.json').write_text(json.dumps(receipt,indent=2)+'\n')
    return receipt
