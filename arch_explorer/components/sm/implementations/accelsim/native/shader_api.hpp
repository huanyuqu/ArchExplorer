// SPDX-License-Identifier: Apache-2.0
#ifndef ARCH_EXPLORER_ACCELSIM_SHADER_API_HPP
#define ARCH_EXPLORER_ACCELSIM_SHADER_API_HPP

#include <memory>
#include "shader_services.hpp"

class shader_core_config;
class memory_config;
class shader_core_stats;
class memory_stats_t;
class trace_simt_core_cluster;

namespace ae_accelsim { namespace standalone {

// The returned object owns the actual native trace cluster, SM pipeline,
// scoreboard, L1 caches and shared-memory machinery. Injected services/configs
// remain assembly-owned and must outlive it. There is no GPU host argument.
class ShaderCluster final {
 public:
  ~ShaderCluster();
  ShaderCluster(const ShaderCluster&) = delete;
  ShaderCluster& operator=(const ShaderCluster&) = delete;
  void core_cycle();
  void receive_cycle();
  unsigned issue_block2core();
  void reinit();
  void cache_flush();
  void cache_invalidate();
  bool busy() const;
  bool drained() const;
  unsigned active_ctas() const;
  unsigned active_sms() const;
  trace_simt_core_cluster& native();
  const trace_simt_core_cluster& native() const;

 private:
  friend std::unique_ptr<ShaderCluster> make_shader_cluster(
      unsigned, const shader_core_config&, const memory_config&,
      shader_core_stats&, memory_stats_t&, const CycleView&, ProgramDispatch&,
      ShaderCounters&, ShaderTransport&, CacheServices&, ShaderOptions);
  explicit ShaderCluster(std::unique_ptr<trace_simt_core_cluster> cluster);
  std::unique_ptr<trace_simt_core_cluster> cluster_;
};

std::unique_ptr<ShaderCluster> make_shader_cluster(
    unsigned cluster_id, const shader_core_config& config,
    const memory_config& memory, shader_core_stats& stats,
    memory_stats_t& memory_stats, const CycleView& cycles,
    ProgramDispatch& program, ShaderCounters& counters,
    ShaderTransport& transport, CacheServices& caches, ShaderOptions options);

}}  // namespace ae_accelsim::standalone
#endif
