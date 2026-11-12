"""Extract independent trace SMs from the pinned typed-v1 source.

Only generated private native trees are changed. The oracle input stays in a
separate tree/image. All inputs/anchors are checked before any output is written.
The transforms substitute explicit services at host dependencies; instruction,
warp, scoreboard, pipeline, L1/shared and CTA algorithms remain upstream code.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
import re


EXPECTED = {
    'gpgpusim/src/gpgpu-sim/shader.h': '4fb4f943fc41f56deccd0246384d00bdfb0bc581ebb55428c6cfc8da55f52d9d',
    'gpgpusim/src/gpgpu-sim/shader.cc': 'a24a5b236c7fac574a8679696b307d1ef908f36d1db525311b9797529f9aeb57',
    'gpgpusim/src/gpgpu-sim/scoreboard.h': 'cd274cbc6ea7fb1e6516f31425db6e28aa0397efd2df7d484e4a60e5ede417a7',
    'gpgpusim/src/gpgpu-sim/scoreboard.cc': '7e63d414e8ece195f9644f4f96ce22aff4bdc0ea7d131fe02911c6f6eeb894ca',
    'gpgpusim/src/abstract_hardware_model.h': '50b428094c66500138244dd216bd360611fec807d652725e83c6975c1a66bb75',
    'gpgpusim/src/abstract_hardware_model.cc': 'a2f03d709f1351c87b294e106e245f7f128a6a1a3dbe50036f1b3e5201a01327',
    'accelsim/gpu-simulator/trace-driven/trace_driven.h': '1e1b736924e573393ad9164b930a26c27f17e886cf63d0d128baa68882bd6821',
    'accelsim/gpu-simulator/trace-driven/trace_driven.cc': '46c9ad39c20ae68e7bfc0dde35554ed040fe0ef056f7d40771d558154d91135c',
    'gpgpusim/src/cuda-sim/cuda-sim.h': '56050541fc7763647d4a46159ca3e23874eda46462518103e901f63194db035d',
    # Read-only source of shader methods upstream placed in the host TU.
    'gpgpusim/src/gpgpu-sim/gpu-sim.cc': 'fd528a420be6bb93199e3c5ecbf48c9f57dc1e5e331c8f017477aee6d2be00e1',
}

NS = 'ae_accelsim::standalone::'
SERVICE_ARGS = (f'const {NS}CycleView& cycles, {NS}ProgramDispatch& program, '
                f'{NS}ShaderCounters& counters, {NS}ShaderTransport& transport, '
                f'{NS}CacheServices& caches, {NS}ShaderOptions options')
SERVICE_VALUES = 'cycles, program, counters, transport, caches, options'
MEMBER_VALUES = 'm_cycles, m_program, m_counters, m_transport, m_cache, m_options'
MEMBERS = (f'  const {NS}CycleView& m_cycles;\n'
           f'  {NS}ProgramDispatch& m_program;\n'
           f'  {NS}ShaderCounters& m_counters;\n'
           f'  {NS}ShaderTransport& m_transport;\n'
           f'  {NS}CacheServices& m_cache;\n'
           f'  const {NS}ShaderOptions m_options;')
INITIALIZERS = ('m_cycles(cycles), m_program(program), m_counters(counters), '
                'm_transport(transport), m_cache(caches), m_options(options)')


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def once(text: str, before: str, after: str) -> str:
    if text.count(before) != 1:
        raise ValueError(f'SM expected one anchor: {before[:100]!r}')
    return text.replace(before, after, 1)


def sub(text: str, pattern: str, replacement: str, count: int | None = None) -> str:
    result, actual = re.subn(pattern, replacement, text, flags=re.S)
    if not actual or (count is not None and actual != count):
        raise ValueError(f'SM anchor count {actual}, expected {count}: {pattern}')
    return result


def braces(text: str, marker: str) -> tuple[int, int]:
    if text.count(marker) != 1:
        raise ValueError(f'SM expected unique block: {marker}')
    marker_end = text.index(marker) + len(marker)
    start = marker_end - 1 if marker.endswith('{') else text.index('{', marker_end)
    depth = 0
    # Comments/string literals do not affect the native C++ brace nesting.
    for token in re.finditer(r'//[^\n]*|/\*.*?\*/|"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'|[{}]',
                             text[start:], re.S):
        if token[0] == '{':
            depth += 1
        elif token[0] == '}':
            depth -= 1
            if not depth:
                return start, start + token.end()
    raise ValueError(f'SM unterminated block: {marker}')


def body(text: str, marker: str, replacement: str) -> str:
    a, b = braces(text, marker)
    return text[:a + 1] + '\n' + replacement + '\n' + text[b - 1:]


def remove_checkpoint_branches(text: str, expected_count: int = 2) -> str:
    matches = list(re.finditer(r'if \(m_gpu->resume_option == 1', text))
    if len(matches) != expected_count:
        raise ValueError(f'SM expected {expected_count} functional CTA checkpoint branches')
    for match in reversed(matches):
        a = text.index('{', match.end())
        depth = 0
        end = None
        for token in re.finditer(r'//[^\n]*|/\*.*?\*/|"(?:\\.|[^"\\])*"|[{}]', text[a:], re.S):
            if token[0] == '{':
                depth += 1
            elif token[0] == '}':
                depth -= 1
                if not depth:
                    end = a + token.end()
                    break
        if end is None:
            raise ValueError('unterminated SM checkpoint branch')
        text = text[:match.start()] + text[end:]
    return text


def remove_class(text: str, name: str) -> str:
    marker = f'class {name} :'
    if marker not in text:
        marker = f'class {name} {{'
        a, b = braces(text, marker[:-1])
    else:
        a, b = braces(text, marker)
    start = text.index(marker)
    end = b
    while text[end].isspace():
        end += 1
    if text[end] != ';':
        raise ValueError(f'SM class terminator: {name}')
    return text[:start] + text[end + 1:]


def definitions(text: str, class_names: str) -> list[tuple[int, int, str]]:
    result = []
    pattern = r'(?m)^[^\n;{}#]*\b(?:' + class_names + r')::[^;{}\n]*\('
    for match in re.finditer(pattern, text):
        a, b = braces(text, match[0])
        result.append((match.start(), b, text[match.start():b]))
    return result


def remove_definitions(text: str, names: str) -> str:
    for start, end, _ in reversed(definitions(text, names)):
        text = text[:start] + text[end:]
    return text


def class_insert(text: str, name: str, insert: str) -> str:
    matches = list(re.finditer(r'\bclass ' + re.escape(name) + r'\b[^;{]*\{', text))
    if len(matches) != 1:
        raise ValueError(f'SM class definition is not unique: {name}')
    a, b = braces(text, matches[0][0])
    access = re.search(r'\bpublic\s*:', text[a:b])
    if not access:
        raise ValueError(f'SM class has no public section: {name}')
    pos = a + access.end()
    return text[:pos] + '\n' + insert + text[pos:]


def service_access(text: str) -> str:
    # Longest spelling first. Apply by origin rather than an omnibus Context.
    for prefix in ('m_core->get_gpu()->', 'm_shader->get_gpu()->', 'm_gpu->'):
        owner = ('m_core->' if prefix.startswith('m_core') else
                 'm_shader->' if prefix.startswith('m_shader') else '')
        clock = owner + 'cycles().' if owner else 'm_cycles.'
        context = owner + 'context()' if owner else 'm_context'
        # Upstream wraps many of these expressions across physical lines.
        for field, replacement in (
            ('gpu_sim_cycle', clock + 'current'),
            ('gpu_tot_sim_cycle', clock + 'completed'),
            ('global_cycle()', clock + 'absolute()'),
            ('gpgpu_ctx', '&' + context),
            ('gpu_sim_insn_last_update_sid', (owner + 'shader_counters().' if owner else 'm_counters.') + 'last_instruction_shader'),
            ('gpu_sim_insn_last_update', (owner + 'shader_counters().' if owner else 'm_counters.') + 'last_instruction_cycle'),
            ('gpu_sim_insn', (owner + 'shader_counters().' if owner else 'm_counters.') + 'instructions'),
            ('gpu_stall_core2icnt', (owner + 'shader_counters().' if owner else 'm_counters.') + 'core_to_icnt_stalls'),
            ('gpu_stall_icnt2core', (owner + 'shader_counters().' if owner else 'm_counters.') + 'icnt_to_core_stalls'),
            ('get_config().get_core_freq()', (owner + 'shader_options().' if owner else 'm_options.') + 'core_frequency_hz'),
            ('get_config().flush_l1()', (owner + 'shader_options().' if owner else 'm_options.') + 'flush_l1'),
            ('get_config().g_power_simulation_enabled', 'false'),
            ('select_kernel()', (owner + 'program_dispatch().' if owner else 'm_program.') + 'select_kernel()'),
            ('kernel_more_cta_left', (owner + 'program_dispatch().' if owner else 'm_program.') + 'kernel_more_cta_left'),
            ('inc_completed_cta()', (owner + 'program_dispatch().' if owner else 'm_program.') + 'inc_completed_cta()'),
            ('set_kernel_done', (owner + 'program_dispatch().' if owner else 'm_program.') + 'set_kernel_done'),
        ):
            text = text.replace(prefix + field, replacement)
    return text


def patch_abstract(header: str, source: str) -> tuple[str, str]:
    header = '#include "ae_component/shader_services.hpp"\n' + header
    header = once(header, 'simt_stack(unsigned wid, unsigned warpSize, class gpgpu_sim *gpu);',
                  f'simt_stack(unsigned wid, unsigned warpSize, const {NS}CycleView& cycles, gpgpu_context& context);')
    header = header.replace('  class gpgpu_sim *m_gpu;',
                            f'  const {NS}CycleView& m_cycles;\n  gpgpu_context& m_context;')
    header = once(header, 'core_t(gpgpu_sim *gpu, kernel_info_t *kernel, unsigned warp_size,',
                  f'core_t(const {NS}CycleView& cycles, gpgpu_context& context, kernel_info_t *kernel, unsigned warp_size,')
    header = once(header, ': m_gpu(gpu),', ': m_cycles(cycles), m_context(context),')
    header = once(header, '  class gpgpu_sim *get_gpu() { return m_gpu; }',
                  f'  const {NS}CycleView& cycles() const {{ return m_cycles; }}\n'
                  '  gpgpu_context& context() const { return m_context; }')
    header = once(header, 'virtual ~core_t() { free(m_thread); }',
                  'virtual ~core_t() { deleteSIMTStack(); free(m_thread); }')
    # Register-set copies are non-owning during vector growth; final shader
    # explicitly releases its surviving unique register containers once.
    header = class_insert(header, 'register_set',
                          '  void ae_release_owned_registers() {\n'
                          '    for (auto* reg : regs) delete reg;\n'
                          '    regs.clear();\n  }')
    source = '#include "ae_component/shader_services.hpp"\n#include <stdexcept>\n' + source
    source = once(source, 'simt_stack::simt_stack(unsigned wid, unsigned warpSize, class gpgpu_sim *gpu) {',
                  f'simt_stack::simt_stack(unsigned wid, unsigned warpSize, const {NS}CycleView& cycles, gpgpu_context& context)\n'
                  '    : m_cycles(cycles), m_context(context) {')
    source = once(source, '  m_gpu = gpu;\n', '')
    source = once(source, 'new simt_stack(i, warp_size, m_gpu)',
                  'new simt_stack(i, warp_size, m_cycles, m_context)')
    source = source.replace('entry->gpgpu_ctx->device_runtime->g_kernel_launch_latency',
                            'entry->gpgpu_ctx->ae_kernel_launch_latency')
    source = source.replace('entry->gpgpu_ctx->device_runtime->g_TB_launch_latency',
                            'entry->gpgpu_ctx->ae_tb_launch_latency')
    source = body(source, 'void kernel_info_t::notify_parent_finished()',
                  '  if (m_parent_kernel) throw std::logic_error("standalone trace does not execute CDP");')
    for marker in ('CUstream_st *kernel_info_t::create_stream_cta(',
                   'CUstream_st *kernel_info_t::get_default_stream_cta('):
        source = body(source, marker,
                      '  throw std::logic_error("standalone trace does not create CUDA streams");')
    source = body(source, 'void kernel_info_t::destroy_cta_streams()',
                  '  if (!m_cta_streams.empty()) std::terminate();')
    for marker in ('warp_inst_t core_t::getExecuteWarp(',
                   'void core_t::execute_warp_inst_t(',
                   'void core_t::updateSIMTStack('):
        source = body(source, marker,
                      '  throw std::logic_error("standalone trace has no PTX functional executor");')
    source = source.replace('m_gpu->gpgpu_ctx->func_sim->ptx_print_insn(stack_entry.m_pc, fout);',
                            'fprintf(fout, "trace pc=0x%llx", static_cast<unsigned long long>(stack_entry.m_pc));')
    source = source.replace('m_gpu->gpgpu_ctx->stats->', 'm_context.stats->')
    source = service_access(source)
    return header, source


def patch_shader_header(text: str) -> str:
    text = '#include "ae_component/shader_services.hpp"\n' + text
    for name in ('exec_shader_core_ctx', 'sst_simt_core_cluster',
                 'exec_simt_core_cluster', 'sst_memory_interface'):
        text = remove_class(text, name)
    text = once(text, 'shader_core_ctx(class gpgpu_sim *gpu, class simt_core_cluster *cluster,',
                f'shader_core_ctx({SERVICE_ARGS}, class simt_core_cluster *cluster,')
    text = once(text, 'simt_core_cluster(class gpgpu_sim *gpu, unsigned cluster_id,',
                f'simt_core_cluster({SERVICE_ARGS}, unsigned cluster_id,')
    la, lb = braces(text, 'class ldst_unit :')
    part = once(text[la:lb], '  gpgpu_sim *m_gpu;', f'  {NS}CacheServices& m_cache;')
    text = text[:la] + part + text[lb:]
    ca, cb = braces(text, 'class simt_core_cluster {')
    part = once(text[ca:cb], '  gpgpu_sim *m_gpu;', MEMBERS)
    text = text[:ca] + part + text[cb:]
    text = text.replace('  gpgpu_sim *get_gpu() { return m_gpu; }', '')
    text = text.replace('  gpgpu_sim *get_gpu() const { return m_gpu; }', '')
    text = once(text, 'unsigned sid, unsigned tpc, gpgpu_sim *gpu);',
                f'unsigned sid, unsigned tpc, {NS}CacheServices& caches);')
    accessors = (f'  {NS}CacheServices& cache_services() const {{ return m_cache; }}\n'
                 f'  {NS}ProgramDispatch& program_dispatch() const {{ return m_program; }}\n'
                 f'  {NS}ShaderCounters& shader_counters() const {{ return m_counters; }}\n'
                 f'  const {NS}ShaderOptions& shader_options() const {{ return m_options; }}')
    text = once(text, 'class shader_core_ctx : public core_t {\n public:',
                'class shader_core_ctx : public core_t {\n public:\n  virtual ~shader_core_ctx();\n' + accessors)
    text = once(text, '  class simt_core_cluster *m_cluster;',
                '\n'.join(MEMBERS.splitlines()[1:]) + '\n  class simt_core_cluster *m_cluster;')
    text = class_insert(text, 'simt_core_cluster', '  virtual ~simt_core_cluster();')
    text = once(text, '  bool response_queue_full() {',
                '  bool response_queue_empty() const { return m_response_fifo.empty(); }\n'
                '  bool response_queue_full() {')
    text = once(text, '  ~simd_function_unit() { delete m_dispatch_reg; }',
                '  virtual ~simd_function_unit() { delete m_dispatch_reg; }')
    text = once(text, 'class pipelined_simd_unit : public simd_function_unit {\n public:',
                'class pipelined_simd_unit : public simd_function_unit {\n public:\n  ~pipelined_simd_unit() override;')
    text = once(text, 'class ldst_unit : public pipelined_simd_unit {\n public:',
                'class ldst_unit : public pipelined_simd_unit {\n public:\n  ~ldst_unit() override;')
    text = class_insert(text, 'shd_warp_t', '  virtual ~shd_warp_t() = default;')
    text = class_insert(text, 'opndcoll_base_t', '  virtual ~opndcoll_base_t() = default;')
    text = class_insert(text, 'opndcoll_simple_t',
                        '  opndcoll_simple_t() : m_swap_buffer(nullptr) {}\n'
                        '  ~opndcoll_simple_t() override { delete m_swap_buffer; }')
    text = class_insert(text, 'opndcoll_rfu_t',
                        '  ~opndcoll_rfu_t() override {\n'
                        '    for (auto& set : m_cus)\n'
                        '      for (auto& collector : set.second) collector.ae_release();\n  }')
    text = class_insert(text, 'collector_unit_t',
                        '  void ae_release() {\n'
                        '    delete m_warp; m_warp = nullptr;\n'
                        '    delete[] m_src_op; m_src_op = nullptr;\n  }')
    text = class_insert(text, 'arbiter_t',
                        '  ~arbiter_t() {\n'
                        '    if (_request) for (unsigned i = 0; i < m_num_banks; ++i) delete[] _request[i];\n'
                        '    delete[] _request; delete[] _inmatch; delete[] _outmatch;\n'
                        '    delete[] m_queue; delete[] m_allocated_bank; delete[] m_allocator_rr_head;\n  }')
    text = once(text, '_request = new int *[m_num_banks];',
                '_request = new int *[m_num_banks]();')
    ca, cb = braces(text, 'class simt_core_cluster {')
    part = once(text[ca:cb], '  shader_core_ctx **m_core;', '  shader_core_ctx **m_core = nullptr;')
    text = text[:ca] + part + text[cb:]
    ca, cb = braces(text, 'class shader_core_ctx : public core_t {')
    part = text[ca:cb]
    for field in ('m_threadState', 'm_icnt', 'm_mem_fetch_allocator', 'm_L1I',
                  'm_scoreboard', 'm_operand_collector', 'm_ldst_unit'):
        part = sub(part, r'(\*' + field + r')\s*;', r'\1 = nullptr;', 1)
    text = text[:ca] + part + text[cb:]
    text = once(text, 'class shader_core_config : public core_config {\n public:',
                'class shader_core_config : public core_config {\n private:\n'
                f'  {NS}InstructionTiming ae_instruction_timing;\n public:\n'
                f'  void set_instruction_timing({NS}InstructionTiming timing) {{ ae_instruction_timing = timing; }}')
    text = sub(text, r',\s*gpgpu_t \*gpu\) = 0;', ') = 0;', 1)
    return text


def patch_shader_source(text: str, host: str) -> tuple[str, dict]:
    extracted = definitions(host, 'shader_core_ctx')
    if len(extracted) != 7:
        raise ValueError(f'SM expected 7 trace shader methods in host TU, got {len(extracted)}')
    provenance = {part[:part.index('(')].strip(): sha(part.encode())
                  for _, _, part in extracted}
    text += '\n// Shader-owned methods extracted without the GPU host TU.\n'
    text += '\n\n'.join(part for _, _, part in extracted)
    text = remove_definitions(text, 'gpgpu_sim|exec_shader_core_ctx|exec_simt_core_cluster|sst_simt_core_cluster')
    text = '#include "ae_component/memory_services.hpp"\n#include <stdexcept>\n' + text
    text = once(text, '#include "shader.h"', '#include "shader.h"\n' + CORE_ISSUE_TRACE)
    # Trace init_warps explicitly calls this base routine before loading trace
    # PCs. Keep its actual mask/stack/warp/counter initialization. next_pc()
    # reads the native nullable thread slot (the original no-PC marker in trace
    # mode); init_traces subsequently supplies the first real SASS PC.
    a, b = braces(text, 'void shader_core_ctx::init_warps(')
    initialize = remove_checkpoint_branches(text[a + 1:b - 1], expected_count=1)
    initialize = once(initialize, '  unsigned kernel_id = kernel.get_uid();\n', '')
    initialize = once(initialize, '    unsigned warp_per_cta = cta_size / m_config->warp_size;\n', '')
    text = text[:a + 1] + initialize + text[b - 1:]
    text = remove_checkpoint_branches(text)
    text = once(text, '  checkpoint *g_checkpoint = new checkpoint();\n', '')
    text = once(text, '  symbol_table *symtab = kernel_func_info->get_symtab();\n', '')
    text = once(text, '  function_info *kernel_func_info = kernel.entry();\n', '')
    text = once(text, 'free_cta_hw_id, warp_id,\n        m_cluster->get_gpu());',
                'free_cta_hw_id, warp_id);')
    text = once(text, 'shader_core_ctx::shader_core_ctx(class gpgpu_sim *gpu,',
                f'shader_core_ctx::shader_core_ctx({SERVICE_ARGS},')
    text = once(text, ': core_t(gpu, NULL, config->warp_size, config->n_thread_per_shader),',
                ': core_t(cycles, *config->gpgpu_ctx, NULL, config->warp_size, config->n_thread_per_shader),\n'
                '      m_program(program), m_counters(counters), m_transport(transport), m_cache(caches), m_options(options),')
    text = once(text, '''  if (get_gpu()->get_config().g_power_simulation_enabled) {
    scaling_coeffs = get_gpu()->get_scaling_coeffs();
  }''', '  scaling_coeffs = nullptr;')
    text = sub(text, r'(?:m_core->|m_shader->)?get_gpu\(\)->get_config\(\).g_power_simulation_enabled', 'false')
    # Perfect/SST routes are separate execution contracts, never hidden hosts.
    start = text.index('  if (m_memory_config->SST_mode) {')
    stop = text.index('  m_mem_fetch_allocator =', start)
    text = text[:start] + '  m_icnt = new shader_memory_interface(this, m_cluster);\n' + text[stop:]
    text = once(text, 'IN_L1I_MISS_QUEUE, OTHER_GPU_CACHE, m_gpu);',
                'IN_L1I_MISS_QUEUE, OTHER_GPU_CACHE, m_cache);')
    text = once(text, 'new Scoreboard(m_sid, m_config->max_warps_per_shader, m_gpu)',
                'new Scoreboard(m_sid, m_config->max_warps_per_shader, m_cycles)')
    text = once(text, 'm_memory_config, m_stats, m_sid, m_tpc, m_gpu);',
                'm_memory_config, m_stats, m_sid, m_tpc, m_cache);')
    text = once(text, 'IN_L1C_MISS_QUEUE, OTHER_GPU_CACHE, m_gpu);',
                'IN_L1C_MISS_QUEUE, OTHER_GPU_CACHE, m_cache);')
    text = once(text, 'unsigned sid, unsigned tpc, gpgpu_sim *gpu)',
                f'unsigned sid, unsigned tpc, {NS}CacheServices& caches)')
    text = once(text, '      m_gpu(gpu) {', '      m_cache(caches) {')
    text = once(text, 'IN_L1D_MISS_QUEUE, core->get_gpu(), L1_GPU_CACHE);',
                'IN_L1D_MISS_QUEUE, m_cache, L1_GPU_CACHE);')
    text = once(text, '''      if (m_memory_config->SST_mode &&
          (static_cast<sst_memory_interface *>(m_icnt)->full(
              size, inst.is_store() || inst.isatomic(), access.get_type()))) {
        // SST need mf type here
        // Cast it to sst_memory_interface pointer first as this full() method
        // is not a virtual method in parent class
        stall_cond = ICNT_RC_FAIL;
        break;
      } else if (!m_memory_config->SST_mode &&
                 (m_icnt->full(size, inst.is_store() || inst.isatomic()))) {''',
                '''      if (m_memory_config->SST_mode) {
        throw std::invalid_argument("standalone trace SM does not support SST");
      } else if (m_icnt->full(size, inst.is_store() || inst.isatomic())) {''')
    # Alternative externally supplied L1 constructor uses the same narrow cache
    # service, with no implicit uninitialized host pointer.
    text = once(text, '      m_L1D(new_l1d_cache),\n      m_next_wb(config) {',
                '      m_L1D(new_l1d_cache),\n      m_next_wb(config),\n      m_cache(core->cache_services()) {')
    text = once(text, 'simt_core_cluster::simt_core_cluster(class gpgpu_sim *gpu, unsigned cluster_id,',
                f'simt_core_cluster::simt_core_cluster({SERVICE_ARGS}, unsigned cluster_id,')
    a, _ = braces(text, 'simt_core_cluster::simt_core_cluster(')
    text = text[:a] + ': ' + INITIALIZERS + ' ' + text[a:]
    text = once(text, '  m_gpu = gpu;\n', '')
    text = once(text, '!::icnt_has_buffer(m_cluster_id, request_size)',
                '!m_transport.can_send(m_cluster_id, request_size)')
    text = sub(text, r'::icnt_push\(m_cluster_id,\s*m_config->mem2device\(destination\),\s*\(void\s*\*\)mf,',
               'm_transport.send(m_cluster_id, m_config->mem2device(destination), mf,', 2)
    text = sub(text, r'\(mem_fetch\s*\*\)\s*::icnt_pop\(m_cluster_id\)',
               'm_transport.pop_reply(m_cluster_id)', 1)
    text = text.replace('m_config->gpgpu_ctx->func_sim->ptx_print_insn(pc, fout);',
                        'fprintf(fout, "trace op=%u", static_cast<unsigned>(op));')
    for old, field in (('int', 'integer'), ('fp', 'floating'), ('dp', 'dual'),
                       ('sfu', 'special'), ('tensor', 'tensor')):
        text = text.replace('gpgpu_ctx->func_sim->opcode_latency_' + old,
                            'ae_instruction_timing.' + field)
    a, _ = braces(text, 'void shader_core_config::set_pipeline_latency()')
    text = text[:a + 1] + '''
  if (!ae_instruction_timing.integer || !ae_instruction_timing.floating ||
      !ae_instruction_timing.dual || !ae_instruction_timing.special ||
      !ae_instruction_timing.tensor)
    throw std::invalid_argument("standalone shader needs explicit instruction timing before init");
''' + text[a + 1:]
    # Debug instruction labels must not fetch PTX from a functional GPU.
    text = sub(text, r'm_shader->m_config->gpgpu_ctx->func_sim\s*->ptx_get_insn_str\(\s*[^()]+?\s*\)',
               'std::string("trace")')
    text = sub(text, r'm_shader->m_config->gpgpu_ctx->func_sim\s*->cdp_latency\[[^\]]+\]',
               'ae_unsupported_cdp_latency()', 3)
    text = ('static unsigned ae_unsupported_cdp_latency() {\n'
            '  throw std::logic_error("standalone trace does not execute CDP");\n}\n' + text)
    text = '#include <stdexcept>\n' + text
    text = service_access(text)
    text += DESTRUCTORS
    return text, provenance


CORE_ISSUE_TRACE = r'''
// shader_trace.h omits this macro when tracing is off and its tracing-on
// spelling observes a GPU host. The standalone core uses its injected clock.
#undef CORE_ISSUE_DPRINTF
#if TRACING_ON
#define CORE_ISSUE_DPRINTF(...)                                           \
  do {                                                                  \
    if (SHADER_DTRACE(CORE_ISSUE)) {                                      \
      printf(CORE_ISSUE_PRINT_STR,                                       \
             static_cast<unsigned long long>(m_cycles.absolute()),       \
             Trace::trace_streams_str[Trace::CORE_ISSUE], get_sid());      \
      printf(__VA_ARGS__);                                              \
    }                                                                   \
  } while (0)
#else
#define CORE_ISSUE_DPRINTF(...) do {} while (0)
#endif
'''


DESTRUCTORS = r'''
// Independent ownership: transaction pointers belong to the allocation arena.
simt_core_cluster::~simt_core_cluster() {
  if (m_core) for (unsigned i = 0; i < m_config->n_simt_cores_per_cluster; ++i) delete m_core[i];
  delete[] m_core;
}
shader_core_ctx::~shader_core_ctx() {
  for (auto* scheduler : schedulers) delete scheduler;
  for (auto* unit : m_fu) delete unit;
  for (auto* warp : m_warp) delete warp;
  for (auto* bus : m_result_bus) delete bus;
  for (auto& registers : m_pipeline_reg) registers.ae_release_owned_registers();
  delete m_operand_collector;
  delete m_scoreboard;
  delete m_L1I;
  delete m_mem_fetch_allocator;
  delete m_icnt;
  free(m_threadState);
}
pipelined_simd_unit::~pipelined_simd_unit() {
  for (unsigned i = 0; i < m_pipeline_depth; ++i) delete m_pipeline_reg[i];
  delete[] m_pipeline_reg;
}
ldst_unit::~ldst_unit() {
  delete m_L1T;
  delete m_L1C;
  delete m_L1D;
}
'''


def patch_trace(header: str, source: str) -> tuple[str, str]:
    header = remove_class(header, 'trace_gpgpu_sim')
    source = remove_definitions(source, 'trace_gpgpu_sim')
    header = header.replace('class gpgpu_sim *gpu,', SERVICE_ARGS + ',')
    header = once(header, 'simt_core_cluster(gpu, cluster_id, config, mem_config, stats, mstats)',
                  'simt_core_cluster(' + SERVICE_VALUES + ', cluster_id, config, mem_config, stats, mstats)')
    header = once(header, 'shader_core_ctx(gpu, cluster, shader_id, tpc_id, config, mem_config,\n                        stats)',
                  'shader_core_ctx(' + SERVICE_VALUES + ', cluster, shader_id, tpc_id, config, mem_config, stats)')
    source = once(source, 'new trace_shader_core_ctx(m_gpu, this, sid, m_cluster_id,',
                  'new trace_shader_core_ctx(' + MEMBER_VALUES + ', this, sid, m_cluster_id,')
    source = once(source, 'new shader_core_ctx *[m_config->n_simt_cores_per_cluster];',
                  'new shader_core_ctx *[m_config->n_simt_cores_per_cluster]();')
    header = class_insert(header, 'trace_shd_warp_t',
                          '  ~trace_shd_warp_t() override { delete m_stream; }')
    header = sub(header, r',\s*gpgpu_t \*gpu\);', ');', 1)
    source = sub(source, r',\s*gpgpu_t \*gpu\) {', ') {', 1)
    return header, source


def patch_functional_header(text: str) -> str:
    # ISA/statistic declarations include this type even though standalone
    # trace assemblies never execute a functional GPU core. Delete the old
    # constructor instead of introducing a GPU-shaped core_t overload.
    marker = '  functionalCoreSim(kernel_info_t *kernel, gpgpu_sim *g, unsigned warp_size)'
    _, end = braces(text, marker)
    start = text.index(marker)
    return text[:start] + marker + ' = delete;' + text[end:]


def patch_shader(root: Path) -> dict:
    root = Path(root).resolve(strict=True)
    original = {}
    for name, expected in EXPECTED.items():
        path = root / name
        if path.is_symlink() or not path.resolve().is_relative_to(root):
            raise ValueError(f'SM input escapes generated tree: {name}')
        data = path.read_bytes()
        if sha(data) != expected:
            raise ValueError(f'SM pinned typed-v1 input mismatch: {name}')
        original[name] = data.decode()
    out = dict(original)
    ah = 'gpgpusim/src/abstract_hardware_model.h'
    ac = 'gpgpusim/src/abstract_hardware_model.cc'
    sh = 'gpgpusim/src/gpgpu-sim/shader.h'
    sc = 'gpgpusim/src/gpgpu-sim/shader.cc'
    bh = 'gpgpusim/src/gpgpu-sim/scoreboard.h'
    bc = 'gpgpusim/src/gpgpu-sim/scoreboard.cc'
    th = 'accelsim/gpu-simulator/trace-driven/trace_driven.h'
    tc = 'accelsim/gpu-simulator/trace-driven/trace_driven.cc'
    fh = 'gpgpusim/src/cuda-sim/cuda-sim.h'
    out[ah], out[ac] = patch_abstract(out[ah], out[ac])
    out[sh] = patch_shader_header(out[sh])
    out[sc], extracted = patch_shader_source(out[sc], original['gpgpusim/src/gpgpu-sim/gpu-sim.cc'])
    out[th], out[tc] = patch_trace(out[th], out[tc])
    out[fh] = patch_functional_header(out[fh])
    out[bh] = '#include "ae_component/services.hpp"\n' + out[bh]
    out[bh] = once(out[bh], 'class gpgpu_t *gpu', f'const {NS}CycleView& cycles')
    out[bh] = once(out[bh], '  class gpgpu_t *m_gpu;', f'  const {NS}CycleView& m_cycles;')
    out[bc] = once(out[bc], 'class gpgpu_t* gpu)\n    : longopregs() {',
                   f'const {NS}CycleView& cycles)\n    : longopregs(), m_cycles(cycles) {{')
    out[bc] = once(out[bc], '  m_gpu = gpu;\n', '')
    out[bc] = service_access(out[bc])
    for name in (sh, sc, bh, bc, th, tc):
        leftovers = [(m.start(), m[0]) for m in re.finditer(
            r'\bm_gpu\b|\bget_gpu\s*\(|\b(?:gpgpu_sim|trace_gpgpu_sim|exec_gpgpu_sim)\s*\*|::icnt_(?:push|pop|has_buffer)|\bfunc_sim\b|\bsst_memory_interface\b', out[name])]
        if leftovers:
            raise ValueError(f'SM host dependency remains in {name}: {leftovers[:8]}')
    if out['gpgpusim/src/gpgpu-sim/gpu-sim.cc'] != original['gpgpusim/src/gpgpu-sim/gpu-sim.cc']:
        raise AssertionError('SM transformer cannot write the oracle host TU')
    outputs = {name: value for name, value in out.items() if value != original[name]}
    report = {'schema': 'arch-explorer.standalone-shader-extraction.v1',
              'inputs': dict(EXPECTED), 'extracted_shader_methods': extracted,
              'outputs': {name: sha(value.encode()) for name, value in outputs.items()}}
    for name, value in outputs.items():
        (root / name).write_text(value)
    return report
