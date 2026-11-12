"""Pinned minimum identity/config services for the component-only image.

This transforms a NEW typed-v1 checkout only. The frozen original engine is
never patched. Each component owns its actual mechanisms; no GPU host or
service locator is substituted here.
"""
import hashlib
from pathlib import Path

from extract import block, replace_once


EXPECTED = {
    'libcuda/gpgpu_context.h': '55672d17661f77d71e548bb11b7bc2b1bbd6e01488737dac3ef5946a01500e97',
    'src/gpgpu-sim/addrdec.cc': 'de15bbe5c7a6e18827da26d743cfa5bce6a002d9a8a4342d23794b9104cedb8e',
    'src/gpgpu-sim/addrdec.h': '2cb34e959b8949113e21ba9e26cd272696be8a5004013f8d91678bab9627fe6f',
    'src/gpgpu-sim/gpu-sim.h': 'c4f52ce5f4aacd8506f4d08cbab8240aa6e4349738ce72728e2f04fc6b351476',
    'src/gpgpu-sim/delayqueue.h': '804128a400d925d06bdd10aa2e1961f308f47f52dec5e1ca1bca9351fd9cdd4d',
    'src/cuda-sim/memory.cc': '88d44b45da3188031cec3f2c9364cf11b4d976a2f86d8c1c814e9d3f4dae4666',
}


def parameter_memory(text):
    # Kernel parameter byte storage remains native. GPU debugger watchpoints
    # are not a component service; refuse registration before data mutation.
    marker='  if (!m_watchpoints.empty())'
    a,b,body=block(text,marker)
    text=text[:text.index(marker)]+text[b:]
    signature='const ptx_instruction *pI) {'
    text=replace_once(text,signature,signature+'\n  if (!m_watchpoints.empty()) throw std::invalid_argument("GPU debugger watchpoints are not standalone parameter-memory services");')
    marker='void memory_space_impl<BSIZE>::set_watch('
    a,b,body=block(text,marker)
    text=text[:a+1]+'\n  throw std::invalid_argument("GPU debugger watchpoints are not standalone parameter-memory services");\n'+text[b-1:]
    return '#include <stdexcept>\n'+text


def patch_core(root: Path):
    root = Path(root)
    source = {}
    for name, expected in EXPECTED.items():
        raw = (root/name).read_bytes()
        if hashlib.sha256(raw).hexdigest() != expected:
            raise ValueError('standalone core input differs: '+name)
        source[name] = raw.decode()

    name = 'libcuda/gpgpu_context.h'
    text = source[name]
    start, end, original = block(text, '  gpgpu_context()')
    # Preserve only per-instance low-level identities. None of the original
    # CUDA API, parser, GPU thread, functional simulator or device runtime is
    # allocated. ptx_stats is a local optional sink, disabled until requested.
    identities = original[:original.index('    api = new cuda_runtime_api(this);')]
    body = '''
    if (!component_only)
      throw std::logic_error("component image forbids the GPU/CUDA environment constructor");
''' + identities + '''
    g_filename = nullptr;
    api = nullptr;
    ptxinfo = nullptr;
    ptx_parser = nullptr;
    the_gpgpusim = nullptr;
    func_sim = nullptr;
    device_runtime = nullptr;
    stats = instruction_statistics ? new ptx_stats(this) : nullptr;
    if (stats) stats->enable_ptx_file_line_stats = false;
'''
    text = text[:start+1]+body+text[end-1:]
    text = replace_once(text, '  gpgpu_context()',
                        '  explicit gpgpu_context(bool component_only = false, bool instruction_statistics = false)')
    text = replace_once(text, '  // global list',
                        '  ~gpgpu_context() { delete stats; }\n'
                        '  unsigned long long ae_kernel_launch_latency = 0;\n'
                        '  unsigned long long ae_tb_launch_latency = 0;\n'
                        '  // Per-instance low-level identity fields (no GPU environment).')
    source[name] = '#include <stdexcept>\n'+text

    name = 'src/gpgpu-sim/gpu-sim.h'
    text = '#include "ae_component/services.hpp"\n'+source[name]
    text = replace_once(text, '  memory_config(gpgpu_context *ctx) {', '''  ae_accelsim::standalone::RequestIds* ae_request_ids = nullptr;
  ae_accelsim::standalone::TransactionObserver* ae_transaction_observer = nullptr;
  memory_config(gpgpu_context *ctx) {''')
    text = replace_once(text, '  gpgpu_sim_config(gpgpu_context *ctx)', '''  int ae_ptx_instruction_classification = 0, ae_sim_mode = 0;
  int ae_kernel_launch_latency = 0, ae_tb_launch_latency = 0;
  bool ae_cdp_enabled = false;
  gpgpu_sim_config(gpgpu_context *ctx)''')
    text = replace_once(text, '  friend class sst_gpgpu_sim;',
                        '  friend class sst_gpgpu_sim;\n  friend struct ae_accelsim::standalone::ConfigurationAccess;')
    text = replace_once(text, '''    n_chiplet = shader_config->n_chiplet;
    chiplet_interleave = shader_config->chiplet_interleave;
    m_n_sub_partition_per_chiplet = m_n_mem_sub_partition / n_chiplet;
    n_simt_clusters_per_chiplet = shader_config->n_simt_clusters_per_chiplet;''', '''    // No SM configuration is required for a single-chiplet memory domain.
    n_chiplet = shader_config ? shader_config->n_chiplet : 1;
    chiplet_interleave = shader_config ? shader_config->chiplet_interleave : CHIPLET_INTERLEAVED;
    m_n_sub_partition_per_chiplet = m_n_mem_sub_partition / n_chiplet;
    n_simt_clusters_per_chiplet = shader_config ? shader_config->n_simt_clusters_per_chiplet : 1;''')
    source[name] = text

    name = 'src/gpgpu-sim/delayqueue.h'
    text = '#include <vector>\n'+source[name]
    text = replace_once(text, 'class fifo_pipeline {\n public:', '''class fifo_pipeline {
 public:
  // Delay bubbles are not physical payloads. These queries never pop,
  // advance or mutate queue length/statistics/credits.
  bool ae_has_payload() const {
    for (const fifo_data<T>* node = m_head; node; node = node->m_next)
      if (node->m_data) return true;
    return false;
  }
  std::vector<T*> ae_payloads() const {
    std::vector<T*> result;
    for (const fifo_data<T>* node = m_head; node; node = node->m_next)
      if (node->m_data) result.push_back(node->m_data);
    return result;
  }''')
    source[name] = text

    name = 'src/gpgpu-sim/addrdec.cc'
    text = replace_once(source[name], '  unsigned n_chiplet = m_shader_config->n_chiplet;',
                        '  unsigned n_chiplet = m_shader_config ? m_shader_config->n_chiplet : 1;')
    text = replace_once(text, '  return m_shader_config->n_chiplet;',
                        '  return m_shader_config ? m_shader_config->n_chiplet : 1;')
    text = replace_once(text, 'rand() % (m_n_channel * m_n_sub_partition_in_channel)',
                        'ae_partition_random.next() % (m_n_channel * m_n_sub_partition_in_channel)')
    text = replace_once(text, '  if (memory_partition_indexing == RANDOM) srand(1);',
                        '  if (memory_partition_indexing == RANDOM) { address_random_interleaving.clear(); ae_partition_random.seed(1); }')
    source[name] = text

    name = 'src/gpgpu-sim/addrdec.h'
    text = '#include "ae_component/partition_random.hpp"\n#include "../tr1_hash_map.h"\n'+source[name]
    text = replace_once(text, '  const shader_core_config *m_shader_config;',
                        '  const shader_core_config *m_shader_config;\n'
                        '  mutable tr1_hash_map<new_addr_type,unsigned> address_random_interleaving;\n'
                        '  mutable ae_accelsim::standalone::PartitionRandom ae_partition_random;')
    source[name] = text

    name = 'src/cuda-sim/memory.cc'
    source[name] = parameter_memory(source[name])

    outputs = {}
    for name, text in source.items():
        (root/name).write_text(text)
        outputs[name] = hashlib.sha256(text.encode()).hexdigest()
    return {'schema':'arch-explorer.standalone-core-extraction.v1',
            'inputs':EXPECTED,'outputs':outputs,
            'context_services':['instance identities','disabled local ptx statistics sink'],
            'forbidden_context_services':['CUDA API','PTX parser','GPU simulator context','functional simulator','device runtime'],
            'memory_configuration':'shader_config optional only for supported single-chiplet memory domain'}
