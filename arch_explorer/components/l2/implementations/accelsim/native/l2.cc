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

struct L2Component::Impl {
  const memory_config& config;
  MemoryServices& services;
  std::unique_ptr<memory_sub_partition> native;
  Impl(unsigned id, const memory_config& config, memory_stats_t& stats,
       MemoryServices& services) : config(config), services(services),
      native(new memory_sub_partition(id, &config, &stats, services)) {}
};

L2Component::L2Component(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}

L2Component::~L2Component() = default;

std::unique_ptr<L2Component> make_l2(
    unsigned id, const memory_config& config, memory_stats_t& stats,
    MemoryServices& services) {
  validate(config, services);
  if (id >= config.m_n_mem_sub_partition || stats.m_memory_config != &config)
    throw std::invalid_argument("L2 ID or statistics belongs to another configuration");
  return std::unique_ptr<L2Component>(new L2Component(
      std::unique_ptr<L2Component::Impl>(new L2Component::Impl(id, config, stats, services))));
}

unsigned L2Component::id() const { return impl_->native->get_id(); }

bool L2Component::can_accept(const ae_dynamic::Transaction& tx) const {
  auto* mf = request(impl_->services, tx);
  return MemoryNativeAccess::can_accept(*impl_->native, mf);
}

bool L2Component::accept(const ae_dynamic::Transaction& tx) {
  if (!can_accept(tx)) return false;
  impl_->native->push(request(impl_->services, tx), impl_->services.cache.clock.absolute());
  return true;
}

bool L2Component::response_cycle() {
  return impl_->native->top() != nullptr;
}

bool L2Component::peek_response(ae_dynamic::Transaction& tx) const {
  // Native top() may retire a writeback. Keep a public peek strictly read-only.
  auto* mf = MemoryNativeAccess::peek_response(*impl_->native);
  if (!mf) return false;
  tx = impl_->services.requests.describe(mf);
  return true;
}

bool L2Component::pop_response(std::uint64_t expected_id) {
  ae_dynamic::Transaction tx;
  if (!peek_response(tx) || tx.id != expected_id) return false;
  impl_->native->pop();
  return true;
}

bool L2Component::peek_dram_request(ae_dynamic::Transaction& tx) const {
  auto* mf = impl_->native->L2_dram_queue_top();
  if (!mf) return false;
  tx = impl_->services.requests.describe(mf);
  return true;
}

void L2Component::tick() {
  impl_->native->cache_cycle(impl_->services.cache.clock.absolute());
}

bool L2Component::drained() const { return MemoryNativeAccess::drained(*impl_->native); }

L2Metrics L2Component::metrics() const { return MemoryNativeAccess::metrics(*impl_->native); }

memory_sub_partition& MemoryNativeAccess::l2(L2Component& node) { return *node.impl_->native; }

mem_fetch* MemoryNativeAccess::peek_response(const memory_sub_partition& node) {
  return node.m_L2_icnt_queue->top();
}

L2Metrics MemoryNativeAccess::metrics(const memory_sub_partition& n) {
  L2Metrics out;
  if (n.m_L2cache) {
    cache_sub_stats css;
    css.clear();
    n.m_L2cache->get_sub_stats(css);
    out.accesses = css.accesses; out.misses = css.misses;
    out.pending_hits = css.pending_hits; out.reservation_fails = css.res_fails;
    out.port_available_cycles = css.port_available_cycles;
    out.data_port_busy_cycles = css.data_port_busy_cycles;
    out.fill_port_busy_cycles = css.fill_port_busy_cycles;
    cache_request_status hits[] = {HIT};
    mem_access_type accesses[NUM_MEM_ACCESS_TYPE];
    for (unsigned t = 0; t < NUM_MEM_ACCESS_TYPE; ++t)
      accesses[t] = static_cast<mem_access_type>(t);
    out.hits = n.m_L2cache->get_stats().get_stats(
        accesses, NUM_MEM_ACCESS_TYPE, hits, 1);
  }
  out.tracked = n.m_request_tracker.size(); out.rop = n.m_rop.size();
  out.ingress = n.m_icnt_L2_queue->get_length();
  out.dram_requests = n.m_L2_dram_queue->get_length();
  out.dram_replies = n.m_dram_L2_queue->get_length();
  out.responses = n.m_L2_icnt_queue->get_length();
  return out;
}

bool MemoryNativeAccess::drained(const memory_sub_partition& n) {
  return n.m_request_tracker.empty() && n.m_rop.empty() &&
      n.m_icnt_L2_queue->empty() && n.m_L2_dram_queue->empty() &&
      n.m_dram_L2_queue->empty() && n.m_L2_icnt_queue->empty() &&
      (!n.m_L2cache || n.m_L2cache->ae_drained());
}

bool MemoryNativeAccess::can_accept(const memory_sub_partition& n, mem_fetch* mf) {
  if (!mf || mf->get_sub_partition_id() != n.m_id) return false;
  unsigned packets = 1;
  if (n.m_config->m_L2_config.get_cache_type() == SECTOR) {
    // Exactly the native breakdown shape, without constructing any children.
    const auto& mask = mf->get_access_sector_mask();
    if (mf->get_data_size() == SECTOR_SIZE && mask.count() == 1) packets = 1;
    else if (mf->get_data_size() == MAX_MEMORY_ACCESS_SIZE) packets = SECTOR_CHUNCK_SIZE;
    else if (mf->get_data_size() == 64 && (mask.all() || mask.none())) packets = 2;
    else packets = mask.count();
    if (!packets) throw std::invalid_argument("memory request has no active sectors");
  }
  // Native is_avilable_size is historically named inversely: true means full.
  return !n.m_icnt_L2_queue->is_avilable_size(packets);
}

}}  // namespace ae_accelsim::standalone
