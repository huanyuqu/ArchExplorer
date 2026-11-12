// SPDX-License-Identifier: Apache-2.0
#include "shader_api.hpp"
#include "memory_services.hpp"
#include "gpgpu_context.h"
#include "gpgpu-sim/gpu-sim.h"
#include "trace_driven.h"
#include <stdexcept>
#include <utility>

namespace ae_accelsim { namespace standalone {

ShaderCluster::ShaderCluster(std::unique_ptr<trace_simt_core_cluster> cluster)
    : cluster_(std::move(cluster)) {}
ShaderCluster::~ShaderCluster() = default;
void ShaderCluster::core_cycle() { cluster_->core_cycle(); }
void ShaderCluster::receive_cycle() { cluster_->icnt_cycle(); }
unsigned ShaderCluster::issue_block2core() { return cluster_->issue_block2core(); }
void ShaderCluster::reinit() { cluster_->reinit(); }
void ShaderCluster::cache_flush() { cluster_->cache_flush(); }
void ShaderCluster::cache_invalidate() { cluster_->cache_invalidate(); }
bool ShaderCluster::busy() const { return cluster_->get_not_completed() != 0; }
bool ShaderCluster::drained() const {
  return !busy() && cluster_->response_queue_empty();
}
unsigned ShaderCluster::active_ctas() const { return cluster_->get_n_active_cta(); }
unsigned ShaderCluster::active_sms() const { return cluster_->get_n_active_sms(); }
trace_simt_core_cluster& ShaderCluster::native() { return *cluster_; }
const trace_simt_core_cluster& ShaderCluster::native() const { return *cluster_; }

std::unique_ptr<ShaderCluster> make_shader_cluster(
    unsigned cluster_id, const shader_core_config& config,
    const memory_config& memory, shader_core_stats& stats,
    memory_stats_t& memory_stats, const CycleView& cycles,
    ProgramDispatch& program, ShaderCounters& counters,
    ShaderTransport& transport, CacheServices& caches, ShaderOptions options) {
  if (!config.n_simt_cores_per_cluster || cluster_id >= config.n_simt_clusters ||
      !config.gpgpu_ctx || !options.core_frequency_hz || config.gpgpu_perfect_mem ||
      memory.SST_mode || &caches.clock != &cycles || &caches.context != config.gpgpu_ctx)
    throw std::invalid_argument("standalone SM requires a valid trace configuration and transport");
  if (!config.gpgpu_ctx->stats)
    throw std::invalid_argument(
        "standalone SM requires instruction statistics (gpgpu_context(true, true))");
  std::unique_ptr<trace_simt_core_cluster> cluster(new trace_simt_core_cluster(
      cycles, program, counters, transport, caches, options, cluster_id,
      &config, &memory, &stats, &memory_stats));
  return std::unique_ptr<ShaderCluster>(new ShaderCluster(std::move(cluster)));
}

}}  // namespace ae_accelsim::standalone
