// SPDX-License-Identifier: Apache-2.0
#include "memory_api.hpp"
#include "memory_services.hpp"
#include "../gpu-sim.h"
#include "../l2cache.h"
#include "../dram.h"
#include "../dram_sched.h"
#include "../mem_fetch.h"
#include "../mem_latency_stat.h"
#include <stdexcept>

namespace ae_accelsim { namespace standalone {
namespace memory_factory_detail {
void validate(const memory_config& config, const MemoryServices& services) {
  if (!config.m_valid || !config.ae_request_ids ||
      !config.ae_transaction_observer ||
      config.gpgpu_ctx != &services.cache.context)
    throw std::invalid_argument("memory factory requires valid instance-owned configuration, IDs, observer and access context");
  if (config.n_chiplet != 1 || config.lrc_enabled ||
      config.m_address_mapping.get_n_chiplet_partition() != 1)
    throw std::invalid_argument("standalone memory supports explicit single-chiplet/LRC-off connections");
}
mem_fetch* request(MemoryServices& services, const ae_dynamic::Transaction& tx) {
  auto* mf = services.requests.resolve(tx.id);
  if (!mf || mf->get_addr() != tx.address || mf->get_data_size() != tx.bytes)
    throw std::invalid_argument("memory transaction does not match its arena request");
  return mf;
}
}

std::unique_ptr<memory_stats_t> make_memory_statistics(
    unsigned endpoints, unsigned warps_per_endpoint, const memory_config& config,
    const CycleView& clock, MemoryStatsObserver* observer) {
  if (!config.m_valid || !endpoints)
    throw std::invalid_argument("memory statistics require valid memory geometry and endpoint slots");
  return std::unique_ptr<memory_stats_t>(new memory_stats_t(
      endpoints, warps_per_endpoint, &config, clock, observer));
}

}}  // namespace ae_accelsim::standalone
