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
void validate(const memory_config&, const MemoryServices&);
mem_fetch* request(MemoryServices&, const ae_dynamic::Transaction&);
}
using memory_factory_detail::validate;
using memory_factory_detail::request;

struct DramComponent::Impl {
  MemoryServices& services;
  std::unique_ptr<memory_partition_unit> native;
  Impl(unsigned id, const memory_config& config, memory_stats_t& stats,
       MemoryServices& services, const std::vector<memory_sub_partition*>& peers)
      : services(services),
        native(new memory_partition_unit(id, &config, &stats, services, peers)) {}
};

DramComponent::DramComponent(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}

DramComponent::~DramComponent() = default;

std::unique_ptr<DramComponent> make_dram(
    unsigned id, const memory_config& config, memory_stats_t& stats,
    MemoryServices& services, const std::vector<L2Component*>& peers) {
  validate(config, services);
  if (stats.m_memory_config != &config)
    throw std::invalid_argument("DRAM statistics belongs to another configuration");
  std::vector<memory_sub_partition*> native;
  for (auto* peer : peers) {
    if (!peer) throw std::invalid_argument("null L2 connection");
    native.push_back(&MemoryNativeAccess::l2(*peer));
  }
  return std::unique_ptr<DramComponent>(new DramComponent(
      std::unique_ptr<DramComponent::Impl>(new DramComponent::Impl(id, config, stats, services, native))));
}

unsigned DramComponent::id() const { return impl_->native->get_id(); }

void DramComponent::tick() {
  auto& n = *impl_->native;
  if (MemoryNativeAccess::simple(n)) n.simple_dram_model_cycle();
  else n.dram_cycle();
}

bool DramComponent::can_accept(const ae_dynamic::Transaction& tx) const {
  return impl_->native->ae_can_accept(request(impl_->services, tx));
}

bool DramComponent::drained() const { return MemoryNativeAccess::drained(*impl_->native); }

DramMetrics DramComponent::metrics() const { return MemoryNativeAccess::metrics(*impl_->native); }

memory_partition_unit& MemoryNativeAccess::dram(DramComponent& node) { return *node.impl_->native; }

bool MemoryNativeAccess::simple(const memory_partition_unit& node) {
  return node.m_config->simple_dram_model;
}

DramMetrics MemoryNativeAccess::metrics(const memory_partition_unit& n) {
  DramMetrics out;
  auto& d = *n.m_dram;
  unsigned cmd, activity, nop, act, pre, rd, wr, wb, requests;
  d.set_dram_power_stats(cmd, activity, nop, act, pre, rd, wr, wb, requests);
  out.commands = cmd; out.activity = activity; out.nop = nop;
  out.act = act; out.pre = pre; out.reads = rd; out.writes = wr;
  out.writebacks = wb; out.requests = requests;
  out.delay = n.m_dram_latency_queue.size();
  out.controller = d.mrqq->ae_payloads().size();
  out.read_write = d.rwq->ae_payloads().size();
  out.replies = d.returnq->ae_payloads().size();
  if (d.m_frfcfs_scheduler) {
    out.scheduled_reads = d.m_frfcfs_scheduler->num_pending();
    out.scheduled_writes = d.m_frfcfs_scheduler->num_write_pending();
  }
  for (unsigned b = 0; b < d.m_config->nbk; ++b)
    out.bank_requests += d.bk[b]->mrq != nullptr;
  return out;
}

bool MemoryNativeAccess::drained(const memory_partition_unit& n) {
  auto& d = *n.m_dram;
  if (!n.m_dram_latency_queue.empty() || d.mrqq->ae_has_payload() ||
      d.rwq->ae_has_payload() || d.returnq->ae_has_payload()) return false;
  if (d.m_frfcfs_scheduler && (d.m_frfcfs_scheduler->num_pending() ||
                              d.m_frfcfs_scheduler->num_write_pending())) return false;
  for (unsigned b = 0; b < d.m_config->nbk; ++b)
    if (d.bk[b]->mrq) return false;
  return true;
}

}}  // namespace ae_accelsim::standalone
