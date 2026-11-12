#!/usr/bin/env python3
"""Extract independent native memory components from the pinned typed-v1 tree.

The original cache, arbitration, scheduler and DRAM command bodies remain the
only mechanisms. This transform changes their explicit dependencies, ownership,
observation and transport boundaries. Run on a fresh tree after patch_core,
before the SM/ICNT transforms; never run the former SIM38 Bridge extractor.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil

EXPECTED = {
    "dram.cc": "0e427702bd3d73bcbb52d7317d9a511d82761324b13748d77ef41d6f1f10d3f5",
    "dram.h": "a7281038aa2020ad324697152b3f418eac6f8e737a2f637631244ab17e6f525c",
    "dram_sched.cc": "2a0280247c5558541dc3896f86bc596549dd83b71630717cc40756248fee3e73",
    "dram_sched.h": "72d34ecf65ca16501d967d830ead8d0634a4a0353850ef4fe396c10dfce61a06",
    "l2cache.cc": "1b2b0ebca49a2f5943c9350a637d849961da96159a147a70b887434bd7f37ff5",
    "l2cache.h": "724164b9a9e97b408781cccf8efccd94c56c3cf79667941b2ea8a3105cfb7890",
    "mem_fetch.cc": "9fa9075185675555905d669d8cd3aee04a797e7986b7a992d935ee75a8e1eec8",
    "mem_fetch.h": "bd6342af1b1651b922c9e99c0edcebc6415ee75ad0d1c86b1c4a7e60f48d63d1",
    "mem_latency_stat.cc": "bd04d9b1aa377457f512d52e414a54e0208573cb13df4be97a32e70781a29045",
    "mem_latency_stat.h": "c5e069b5a3580c50a624608933b1736fa0841418e183d5798a1f7b03a8658951",
    "gpu-cache.cc": "0a8bded03d50b69049f65c4cfd59f277b160a2d8165672d66c18ec2cda86a8e6",
    "gpu-cache.h": "bbb0039de60ceffb21ce4a7e8e7cf20393f87c9cb0569374bfae45cb2110831d",
}


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def replace(text: str, old: str, new: str, count: int = 1) -> str:
    if text.count(old) != count:
        raise ValueError(f"native anchor count {text.count(old)} != {count}: {old[:100]!r}")
    return text.replace(old, new)


def regex(text: str, pattern: str, new: str, count: int = 1) -> str:
    text, actual = re.subn(pattern, new, text, flags=re.S)
    if actual != count:
        raise ValueError(f"native pattern count {actual} != {count}: {pattern[:100]!r}")
    return text


def span(text: str, marker: str) -> tuple[int, int, int]:
    """Find one C++ function, balancing braces outside comments/literals."""
    if text.count(marker) != 1:
        raise ValueError(f"function marker is not unique: {marker!r}")
    start = text.index(marker)
    begin = text.index("{", start)
    depth, state, escape, i = 0, "code", False, begin
    while i < len(text):
        ch, pair = text[i], text[i:i + 2]
        if state in ("string", "char"):
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == ('"' if state == "string" else "'"):
                state = "code"
        elif state == "line":
            if ch == "\n":
                state = "code"
        elif state == "block":
            if pair == "*/":
                state = "code"
                i += 1
        elif pair == "//":
            state = "line"
            i += 1
        elif pair == "/*":
            state = "block"
            i += 1
        elif ch in ('"', "'"):
            state = "string" if ch == '"' else "char"
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return start, begin, i + 1
        i += 1
    raise ValueError(f"unbalanced native function: {marker}")


def function(text: str, marker: str, new: str) -> str:
    a, _, b = span(text, marker)
    return text[:a] + new + text[b:]


def body_prefix(text: str, marker: str, prefix: str) -> str:
    _, b, _ = span(text, marker)
    return text[:b + 1] + "\n" + prefix + text[b + 1:]


def include(text: str, header: str) -> str:
    match = re.search(r'^#include[^\n]*\n', text, re.M)
    if match is None:
        raise ValueError("native include anchor missing")
    directive = f"#include <{header}>\n" if header == "set" else f'#include "{header}"\n'
    return text[:match.start()] + directive + text[match.start():]


def clock(text: str, owner: str) -> str:
    # Preserve current-vs-absolute meanings, including scheduler timestamps.
    for a, b in (("gpu_sim_cycle", "gpu_tot_sim_cycle"),
                 ("gpu_tot_sim_cycle", "gpu_sim_cycle")):
        text = re.sub(r"m_gpu->" + a + r"\s*\+\s*m_gpu->" + b,
                      owner + ".absolute()", text)
    text = text.replace("m_gpu->global_cycle()", owner + ".absolute()")
    text = text.replace("m_gpu->gpu_sim_cycle", owner + ".current")
    text = text.replace("m_gpu->gpu_tot_sim_cycle", owner + ".completed")
    return text


def patch_cache(h: str, cc: str) -> tuple[str, str]:
    h = include(h, "ae_component/memory_services.hpp")
    # The pinned cache_config exposes other read-only geometry but not its
    # cache type. Keep the field protected and expose only this typed query.
    h = replace(h, "  enum mshr_config_t get_mshr_type() const { return m_mshr_type; }",
                "  enum mshr_config_t get_mshr_type() const { return m_mshr_type; }\n  enum cache_type get_cache_type() const { return m_cache_type; }")
    # Every cache constructor takes the same narrow dependency, including the
    # custom-tag variants used by component-local SM implementations.
    h = re.sub(r"(?:class )?gpgpu_sim \*gpu", "const ae_accelsim::standalone::CacheServices& services", h)
    h = re.sub(r"(?:class )?gpgpu_sim \*m_gpu;", "", h)
    h = re.sub(r"\bgpu\b(?=\s*[,\)])", "services", h)
    h = replace(h, "    m_gpu = gpu;", "", 1)
    h = replace(h, "  cache_gpu_level m_level;", "  cache_gpu_level m_level;\n  const ae_accelsim::standalone::CacheServices& m_services;")
    h = replace(h, "enum mem_fetch_status status, tag_array *new_tag_array)",
                "enum mem_fetch_status status, tag_array *new_tag_array,\n                 const ae_accelsim::standalone::CacheServices& services)", 2)
    h = replace(h, "        m_tag_array(new_tag_array),", "        m_tag_array(new_tag_array),\n        m_level(L1_GPU_CACHE), m_services(services),")
    h = replace(h, "                       new_tag_array) {", "                       new_tag_array, services) {", 2)
    h = replace(h, "  bool busy() const { return false; }", "  bool busy() const { return false; }\n  bool ae_empty() const { return m_data.empty() && pending_lines.empty() && m_current_response.empty(); }")
    h = replace(h, "  virtual ~baseline_cache() { delete m_tag_array; }", "  virtual ~baseline_cache() { delete m_tag_array; }\n  bool ae_drained() const { return m_miss_queue.empty() && m_mshrs.ae_empty() && m_extra_mf_fields.empty(); }")

    cc = re.sub(r"(?:class )?gpgpu_sim \*gpu", "const ae_accelsim::standalone::CacheServices& services", cc)
    cc = replace(cc, "      m_gpu(gpu),", "      m_services(services),")
    cc = replace(cc, "                     gpu) {", "                     services) {")
    cc = replace(cc, "  m_gpu = gpu;", "")
    # Cache counters remain in m_stats; no hidden GPU counter registry.
    a, b, c = span(cc, "data_cache::data_cache(")
    ctor = cc[a:c]
    cut = ctor.index("  m_gpu->perf_counters")
    cc = cc[:a] + ctor[:cut] + "}\n" + cc[c:]
    for marker in ("void baseline_cache::inc_aggregated_stats(",
                   "void baseline_cache::inc_aggregated_fail_stats(",
                   "void baseline_cache::inc_aggregated_stats_pw("):
        cc = body_prefix(cc, marker, "  if (!m_services.aggregate) return;\n")
    cc = cc.replace("m_gpu->aggregated_l1_stats", "(*m_services.aggregate)")
    cc = cc.replace("m_gpu->aggregated_l2_stats", "(*m_services.aggregate)")
    cc = cc.replace("m_gpu->gpgpu_ctx", "&m_services.context")
    cc = clock(cc, "m_services.clock")
    return h, cc


def patch_memfetch(h: str, cc: str) -> tuple[str, str]:
    cc = include(cc, "ae_component/services.hpp")
    cc = replace(cc, "unsigned mem_fetch::sm_next_mf_request_uid = 1;\nstd::vector<void *> mem_fetch::s_free_list;", "")
    cc = function(cc, "void *mem_fetch::operator new(", "void *mem_fetch::operator new(size_t size) { return ::operator new(size); }")
    cc = function(cc, "void mem_fetch::operator delete(", "void mem_fetch::operator delete(void *ptr) noexcept { ::operator delete(ptr); }")
    cc = replace(cc, "  m_request_uid = sm_next_mf_request_uid++;", "  if (!config->ae_request_ids) throw std::invalid_argument(\"missing per-instance request identity service\");\n  m_request_uid = config->ae_request_ids->next();", 2)
    # Both constructors publish ownership only after the last original field
    # initialization (including sector-parent address overrides). Internal
    # requests must be owned even when they never reach a Runtime wire. A
    # registration failure propagates before the caller can enqueue this MF.
    cc = replace(cc, "    m_raw_addr.sub_partition = m_original_mf->get_tlx_addr().sub_partition;\n  }\n}",
                 "    m_raw_addr.sub_partition = m_original_mf->get_tlx_addr().sub_partition;\n  }\n  if (config->ae_transaction_observer)\n    config->ae_transaction_observer->allocated(m_request_uid, this);\n}", 2)
    cc = function(cc, "mem_fetch::~mem_fetch()", "mem_fetch::~mem_fetch() {\n  m_status = MEM_FETCH_DELETED;\n  if (m_mem_config->ae_transaction_observer)\n    m_mem_config->ae_transaction_observer->destroyed(m_request_uid);\n}")
    h = regex(h, r"\s*static unsigned sm_next_mf_request_uid;", "")
    h = regex(h, r"\s*static std::vector<void \*> s_free_list;", "")
    return h, cc


def patch_dram(h: str, cc: str, sh: str, sc: str) -> tuple[str, str, str, str]:
    h = include(h, "ae_component/memory_services.hpp")
    h = re.sub(r"class gpgpu_sim \*gpu", "const ae_accelsim::standalone::CycleView& clock", h)
    h = replace(h, "  class gpgpu_sim *m_gpu;", "  const ae_accelsim::standalone::CycleView& m_clock;", 2)
    h = replace(h, "class dram_t {\n public:", "class dram_t {\n  friend struct ae_accelsim::standalone::MemoryNativeAccess;\n public:\n  ~dram_t();")
    cc = include(cc, "../intersim2/stats.hpp")
    cc = include(cc, "set")
    cc = re.sub(r"(?:class )?gpgpu_sim \*gpu", "const ae_accelsim::standalone::CycleView& clock", cc)
    # Both constructors bind clocks before executing the original algorithm.
    cc = replace(cc, "const ae_accelsim::standalone::CycleView& clock) {", "const ae_accelsim::standalone::CycleView& clock) : m_clock(clock) {", 2)
    cc = replace(cc, "  m_gpu = gpu;", "", 2)
    cc = cc.replace("m_memory_partition_unit->get_mgpu()", "m_clock")
    cc = clock(cc, "m_clock")
    cc += """
// Independent object lifetime. Requests belong to the assembly arena; normal
// teardown occurs after physical drain, and never ticks/flushes from a dtor.
dram_t::~dram_t() {
  // Destroy native timing-request wrappers once even when unwinding a failed
  // attempt. The arena retains the underlying mem_fetch until components are
  // gone; cancellation never marks a request physically complete.
  std::set<dram_req_t*> pending;
  for (auto* p : mrqq->ae_payloads()) pending.insert(p);
  for (auto* p : rwq->ae_payloads()) pending.insert(p);
  for (unsigned b = 0; b < m_config->nbk; ++b)
    if (bk[b]->mrq) pending.insert(bk[b]->mrq);
  if (m_frfcfs_scheduler) m_frfcfs_scheduler->ae_collect(pending);
  delete m_frfcfs_scheduler;
  delete rwq; delete mrqq; delete returnq;
  for (auto* p : pending) delete p;
  free(bk[0]); free(bk); free(bkgrp[0]); free(bkgrp);
  delete mrqq_Dist;
}
"""
    sh = include(sh, "set")
    sh = replace(sh, "  frfcfs_scheduler(const memory_config *config, dram_t *dm,", "  ~frfcfs_scheduler();\n  void ae_collect(std::set<dram_req_t*>&) const;\n  frfcfs_scheduler(const memory_config *config, dram_t *dm,")
    sc = sc.replace("m_dram->m_gpu->gpu_sim_cycle", "m_dram->m_clock.current")
    sc = clock(sc, "m_clock")
    sc += """
void frfcfs_scheduler::ae_collect(std::set<dram_req_t*>& output) const {
  for (unsigned b = 0; b < m_config->nbk; ++b) {
    for (auto* p : m_queue[b]) output.insert(p);
    if (m_config->seperate_write_queue_enabled)
      for (auto* p : m_write_queue[b]) output.insert(p);
  }
}
frfcfs_scheduler::~frfcfs_scheduler() {
  delete[] m_queue; delete[] m_bins; delete[] m_last_row;
  delete[] curr_row_service_time; delete[] row_service_timestamp;
  if (m_config->seperate_write_queue_enabled) {
    delete[] m_write_queue; delete[] m_write_bins; delete[] m_last_write_row;
  }
}
"""
    return h, cc, sh, sc


def patch_statistics(h: str, cc: str) -> tuple[str, str]:
    h = include(h, "ae_component/memory_services.hpp")
    h = regex(h, r"  memory_stats_t\(unsigned n_shader,.*?const class gpgpu_sim \*gpu\);", "  memory_stats_t(unsigned n_shader, unsigned warps_per_endpoint,\n                 const memory_config *mem_config,\n                 const ae_accelsim::standalone::CycleView& clock,\n                 ae_accelsim::standalone::MemoryStatsObserver* observer = nullptr);\n  ~memory_stats_t();")
    h = replace(h, "  const shader_core_config *m_shader_config;", "")
    h = replace(h, "  const class gpgpu_sim *m_gpu;", "  const ae_accelsim::standalone::CycleView& m_clock;\n  ae_accelsim::standalone::MemoryStatsObserver* m_observer;")
    cc = regex(cc, r"memory_stats_t::memory_stats_t\(unsigned n_shader,.*?const class gpgpu_sim \*gpu\)", "memory_stats_t::memory_stats_t(unsigned n_shader, unsigned warps_per_endpoint,\n    const memory_config *mem_config,\n    const ae_accelsim::standalone::CycleView& clock,\n    ae_accelsim::standalone::MemoryStatsObserver* observer)")
    cc = replace(cc, "    : LRC_subpartition_num_icnt_to_lrc_sectors(", "    : m_clock(clock), m_observer(observer), LRC_subpartition_num_icnt_to_lrc_sectors(")
    cc = replace(cc, "  assert(shader_config->m_valid);", "  if (!n_shader) throw std::invalid_argument(\"memory statistics need endpoint slots\");")
    cc = regex(cc, r"shader_config->n_thread_per_shader\s*/\s*shader_config->warp_size", "warps_per_endpoint")
    cc = replace(cc, "  m_gpu = gpu;", "")
    cc = clock(cc, "m_clock")
    cc = replace(cc, "  shader_mem_lat_log(mf->get_sid(), mf_latency);", "  if (m_observer) m_observer->latency(mf->get_sid(), mf_latency);")
    cc = replace(cc, "        shader_mem_acc_log(mf->get_sid(), dram_id, bank, 'w');", "        if (m_observer) m_observer->access(mf->get_sid(), dram_id, bank, true);")
    cc = replace(cc, "      bankreads[mf->get_sid()][dram_id][bank]++;\n      shader_mem_acc_log(mf->get_sid(), dram_id, bank, 'r');", "      if (mf->get_sid() < m_n_shader) {\n        bankreads[mf->get_sid()][dram_id][bank]++;\n        if (m_observer) m_observer->access(mf->get_sid(), dram_id, bank, false);\n      }")
    cc = replace(cc, "  if (mf->get_pc() != (unsigned)-1)\n    m_gpu->gpgpu_ctx->stats->ptx_file_line_stats_add_dram_traffic(\n        mf->get_pc(), mf->get_data_size());", "  if (m_observer && mf->get_pc() != (unsigned)-1)\n    m_observer->traffic(mf->get_pc(), mf->get_data_size());")
    # All native counter arrays are per-instance. Complete their missing native
    # lifetime; this does not change collection or reset any observed values.
    cc += """
memory_stats_t::~memory_stats_t() {
  const unsigned channels = m_memory_config->m_n_mem;
  for (unsigned e = 0; e < m_n_shader; ++e) {
    for (unsigned c = 0; c < channels; ++c) {
      free(bankreads[e][c]); free(bankwrites[e][c]);
    }
    free(bankreads[e]); free(bankwrites[e]);
  }
  free(bankreads); free(bankwrites);
  for (unsigned c = 0; c < channels; ++c) {
    free(concurrent_row_access[c]); free(num_activates[c]); free(row_access[c]);
    free(max_conc_access2samerow[c]); free(max_servicetime2samerow[c]);
    free(totalbankreads[c]); free(totalbankwrites[c]); free(totalbankaccesses[c]);
    free(mf_total_lat_table[c]); free(mf_max_lat_table[c]);
  }
  free(concurrent_row_access); free(num_activates); free(row_access);
  free(max_conc_access2samerow); free(max_servicetime2samerow);
  free(totalbankreads); free(totalbankwrites); free(totalbankaccesses);
  free(mf_total_lat_table); free(mf_max_lat_table);
  for (unsigned t = 0; t < NUM_MEM_ACCESS_TYPE; ++t) {
    for (unsigned c = 0; c < channels; ++c) free(mem_access_type_stats[t][c]);
    free(mem_access_type_stats[t]);
  }
  free(mem_access_type_stats); free(num_MCBs_accessed); free(position_of_mrq_chosen);
  free(L2_cbtoL2length); free(L2_cbtoL2writelength); free(L2_L2tocblength);
  free(L2_dramtoL2length); free(L2_dramtoL2writelength); free(L2_L2todramlength);
}
"""
    return h, cc


def patch_l2(h: str, cc: str) -> tuple[str, str]:
    h = include(h, "ae_component/memory_services.hpp")
    # Native cache policy remains unchanged; memory geometry is explicit.
    h = replace(h, "           class gpgpu_sim *gpu, enum cache_gpu_level level,", "           const ae_accelsim::standalone::CacheServices& services, enum cache_gpu_level level,")
    h = replace(h, "           class memory_stats_t *stats)", "           class memory_stats_t *stats, const memory_config* memory)")
    h = replace(h, "                   L2_WR_ALLOC_R, L2_WRBK_ACC, gpu, level) {", "                   L2_WR_ALLOC_R, L2_WRBK_ACC, services, level) {\n    m_ae_memory_config = memory;")
    h = replace(h, "  mem_fetch_interface *m_chiplet_port;", "  const memory_config* m_ae_memory_config;\n  mem_fetch_interface *m_chiplet_port;")
    part_sig = """  memory_partition_unit(unsigned partition_id, const memory_config *config,
                        memory_stats_t* stats,
                        ae_accelsim::standalone::MemoryServices& services,
                        const std::vector<memory_sub_partition*>& subpartitions);"""
    sub_sig = """  memory_sub_partition(unsigned sub_partition_id, const memory_config *config,
                       memory_stats_t* stats,
                       ae_accelsim::standalone::MemoryServices& services);"""
    h = regex(h, r"  memory_partition_unit\(unsigned partition_id,.*?&reply_1_to_0\);", part_sig)
    h = regex(h, r"  memory_sub_partition\(unsigned sub_partition_id,.*?&reply_1_to_0\);", sub_sig)
    h = replace(h, "  class gpgpu_sim *get_mgpu() const { return m_gpu; }", "  unsigned get_id() const { return m_id; }\n  bool ae_can_accept(mem_fetch* mf) const;\n  bool ae_route(mem_fetch* mf, bool reply);")
    h = replace(h, "  class gpgpu_sim *m_gpu;", "  ae_accelsim::standalone::MemoryServices& m_services;", 2)
    h = replace(h, "class memory_partition_unit {\n public:", "class memory_partition_unit {\n  friend struct ae_accelsim::standalone::MemoryNativeAccess;\n public:")
    h = replace(h, "class memory_sub_partition {\n public:", "class memory_sub_partition {\n  friend class memory_partition_unit;\n  friend struct ae_accelsim::standalone::MemoryNativeAccess;\n  memory_partition_unit* m_ae_partition = nullptr;\n public:")
    h = replace(h, "    return m_unit->m_chiplet_icnt.to_peer_request()->full();",
                "    if (m_unit->m_chiplet_disabled) return true;\n    if (!m_unit->m_chiplet_icnt.to_peer_request())\n      throw std::logic_error(\"unconnected chiplet request port\");\n    return m_unit->m_chiplet_icnt.to_peer_request()->full();")
    cc = include(cc, "ae_component/memory_services.hpp")
    cc = function(cc, "memory_partition_unit::memory_partition_unit(", """memory_partition_unit::memory_partition_unit(
    unsigned partition_id, const memory_config* config, memory_stats_t* stats,
    ae_accelsim::standalone::MemoryServices& services,
    const std::vector<memory_sub_partition*>& subpartitions)
    : m_id(partition_id), m_config(config), m_stats(stats),
      m_arbitration_metadata(config), m_services(services) {
  if (config->n_chiplet != 1 || config->lrc_enabled ||
      config->m_address_mapping.get_n_chiplet_partition() != 1)
    throw std::invalid_argument("standalone memory requires explicit single-chiplet, LRC-off topology");
  if (partition_id >= config->m_n_mem ||
      subpartitions.size() != config->m_n_sub_partition_per_memory_channel)
    throw std::invalid_argument("DRAM requires its exact explicitly connected L2 set");
  m_chiplet_id = 0;
  for (unsigned p = 0; p < subpartitions.size(); ++p) {
    if (!subpartitions[p] || subpartitions[p]->get_id() !=
        m_id * config->m_n_sub_partition_per_memory_channel + p ||
        subpartitions[p]->m_config != config || subpartitions[p]->m_stats != stats ||
        &subpartitions[p]->m_services != &services || subpartitions[p]->m_ae_partition)
      throw std::invalid_argument("DRAM L2 connections are missing, duplicated or out of order");
  }
  m_sub_partition = new memory_sub_partition*[subpartitions.size()];
  for (unsigned p = 0; p < subpartitions.size(); ++p)
    m_sub_partition[p] = subpartitions[p];
  try { m_dram = new dram_t(m_id, m_config, m_stats, this, services.cache.clock); }
  catch (...) { delete[] m_sub_partition; throw; }
  for (auto* sub : subpartitions) sub->m_ae_partition = this;
}""")
    cc = function(cc, "memory_partition_unit::~memory_partition_unit()", """memory_partition_unit::~memory_partition_unit() {
  delete m_dram;
  // Connections borrow L2 objects. Each L2 factory owns exactly its own node.
  for (unsigned p = 0; p < m_config->m_n_sub_partition_per_memory_channel; ++p)
    m_sub_partition[p]->m_ae_partition = nullptr;
  delete[] m_sub_partition;
}""")
    a, begin, end = span(cc, "memory_sub_partition::memory_sub_partition(")
    ctor = cc[a:end]
    ctor = regex(ctor, r"memory_sub_partition::memory_sub_partition\(.*?\) \{", """memory_sub_partition::memory_sub_partition(
    unsigned sub_partition_id, const memory_config* config, memory_stats_t* stats,
    ae_accelsim::standalone::MemoryServices& services) : m_services(services) {""")
    ctor = replace(ctor, "  m_gpu = gpu;", "  m_L2cache = nullptr;\n  if (config->n_chiplet != 1 || config->lrc_enabled ||\n      config->m_address_mapping.get_n_chiplet_partition() != 1)\n    throw std::invalid_argument(\"standalone L2 requires explicit single-chiplet, LRC-off topology\");")
    first = ctor.index("  uint32_t local_sub_partition_id")
    last = ctor.index("  assert(m_id < m_config->m_n_mem_sub_partition);")
    ctor = ctor[:first] + "  m_chiplet_disabled = true;\n\n" + ctor[last:]
    ctor = replace(ctor, "IN_PARTITION_L2_MISS_QUEUE, gpu, L2_GPU_CACHE,", "IN_PARTITION_L2_MISS_QUEUE, services.cache, L2_GPU_CACHE,")
    ctor = replace(ctor, "                             m_id, m_chiplet_id, stats);", "                             m_id, m_chiplet_id, stats, config);")
    cc = cc[:a] + ctor + cc[end:]
    cc = replace(cc, "  delete m_L2interface;", "  delete m_L2interface;\n  delete m_chiplet_interface;\n  delete m_mf_allocator;")
    cc = function(cc, "void gpgpu_sim::print_dram_stats(", "")
    cc = clock(cc, "m_services.cache.clock")
    cc = cc.replace("m_unit->m_gpu->gpu_sim_cycle + m_unit->m_gpu->gpu_tot_sim_cycle", "m_unit->m_services.cache.clock.absolute()")
    cc = cc.replace("m_gpu->getMemoryConfig()", "m_ae_memory_config")
    cc = cc.replace("m_gpu->getShaderCoreConfig()->n_chiplet", "m_config->n_chiplet")
    # A single-chiplet component deliberately has no peer queues. Passive
    # peer probes report absence, while an attempted unconnected send fails
    # before mutating its MF. Connected peer mechanisms retain their bodies.
    for marker, disabled, port in (
        ("mem_fetch *memory_sub_partition::get_chiplet_req()", "nullptr", "from_peer_request"),
        ("bool memory_sub_partition::push_chiplet_reply(", "false", "to_peer_reply"),
        ("bool memory_sub_partition::handle_chiplet_reply()", "false", "from_peer_reply"),
        ("void memory_sub_partition::forward_write_to_peer_chiplet(", "", "to_peer_request"),
    ):
        cc = body_prefix(cc, marker,
                         f"  if (m_chiplet_disabled) return{(' ' + disabled) if disabled else ''};\n"
                         f"  if (!m_chiplet_icnt.{port}())\n"
                         f'    throw std::logic_error("unconnected chiplet {port} port");\n')
    cc = body_prefix(cc, "void Chipletinterface::push(",
                     "  if (m_unit->m_chiplet_disabled || !m_unit->m_chiplet_icnt.to_peer_request())\n"
                     '    throw std::logic_error("unconnected chiplet request port");\n')
    cc = replace(cc, "          !mf->is_write() || !m_chiplet_icnt.to_peer_request()->full();",
                 "          m_chiplet_disabled || !mf->is_write() ||\n          !m_chiplet_interface->full(mf->size(), mf->get_is_write());")
    # l2_cache owns CacheServices through its base, not MemoryServices.
    a, _, end = span(cc, "void l2_cache::cycle()")
    cc = cc[:a] + cc[a:end].replace("m_services.cache.clock", "m_services.clock") + cc[end:]
    # The native arbitration, credit checks and queue order are kept verbatim.
    # Refusal occurs before pop, delay insertion or credit mutation.
    cc = replace(cc, "      m_sub_partition[spid]->L2_dram_queue_pop();", "      if (!ae_route(mf, false)) continue;\n      m_sub_partition[spid]->L2_dram_queue_pop();", 2)
    cc = replace(cc, "if (!m_sub_partition[dest_spid]->dram_L2_queue_full()) {", "if (!m_sub_partition[dest_spid]->dram_L2_queue_full() &&\n            ae_route(mf_return, true)) {", 2)
    cc += """
bool memory_partition_unit::ae_can_accept(mem_fetch* mf) const {
  if (!mf || mf->get_tlx_addr().chip != m_id) return false;
  const int p = global_sub_partition_id_to_local_id(mf->get_sub_partition_id());
  if (p < 0 || static_cast<unsigned>(p) >= m_config->m_n_sub_partition_per_memory_channel)
    return false;
  return !m_sub_partition[p]->L2_dram_queue_empty() &&
         m_sub_partition[p]->L2_dram_queue_top() == mf &&
         !m_sub_partition[p]->dram_L2_queue_full() &&
         m_arbitration_metadata.has_credits(p) && !m_dram->full(mf->is_write());
}

bool memory_partition_unit::ae_route(mem_fetch* mf, bool reply) {
  if (!m_services.routes) return true;
  return m_services.routes->transfer(mf->get_sub_partition_id(), m_id, reply,
                                     m_services.requests.describe(mf));
}
"""
    return h, cc


def transform(original: dict[str, str]) -> dict[str, str]:
    out = dict(original)
    out["gpu-cache.h"], out["gpu-cache.cc"] = patch_cache(out["gpu-cache.h"], out["gpu-cache.cc"])
    out["mem_fetch.h"], out["mem_fetch.cc"] = patch_memfetch(out["mem_fetch.h"], out["mem_fetch.cc"])
    out["dram.h"], out["dram.cc"], out["dram_sched.h"], out["dram_sched.cc"] = patch_dram(
        out["dram.h"], out["dram.cc"], out["dram_sched.h"], out["dram_sched.cc"])
    out["mem_latency_stat.h"], out["mem_latency_stat.cc"] = patch_statistics(out["mem_latency_stat.h"], out["mem_latency_stat.cc"])
    out["l2cache.h"], out["l2cache.cc"] = patch_l2(out["l2cache.h"], out["l2cache.cc"])
    for name, text in out.items():
        if re.search(r"\bm_gpu\b|\bshader_config\b|gpgpu_sim\s*\*|ae_accelsim::Bridge", text):
            raise ValueError(f"unremoved GPU host dependency in {name}")
    return out


def patch_memory(gpgpusim_root: Path | str) -> dict:
    root = Path(gpgpusim_root).resolve()
    native = root / "src/gpgpu-sim"
    original = {}
    # Validate every owned input before writing even the first output.
    for name, expected in EXPECTED.items():
        data = (native / name).read_bytes()
        if sha(data) != expected:
            raise ValueError(f"standalone memory input mismatch: {name}: {sha(data)} != {expected}")
        original[name] = data.decode("utf-8")
    out = transform(original)
    from arch_explorer.components._support.accelsim.native_resources import native_files
    component = native_files()
    facade = ("memory_api.hpp", "memory_services.hpp", "l2.cc", "dram.cc", "memory_statistics.cc")
    for name in facade:
        if name not in component:
            raise ValueError(f"standalone memory facade missing: {name}")
    result = {"schema": "arch-explorer.standalone-memory-transform.v1", "files": {}}
    for name, text in out.items():
        data = text.encode("utf-8")
        (native / name).write_bytes(data)
        result["files"][name] = {"before": EXPECTED[name], "after": sha(data)}
    (native / "ae_component").mkdir(exist_ok=True)
    for name in facade:
        shutil.copy2(component[name], native / "ae_component" / name)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("gpgpusim_root", type=Path)
    args = parser.parse_args()
    print(json.dumps(patch_memory(args.gpgpusim_root), indent=2, sort_keys=True))
