// SPDX-License-Identifier: Apache-2.0
#ifndef ARCH_EXPLORER_ACCELSIM_MEMORY_API_HPP
#define ARCH_EXPLORER_ACCELSIM_MEMORY_API_HPP

#include "runtime.hpp"
#include <cstdint>
#include <memory>
#include <vector>

class memory_config;
class memory_stats_t;

namespace ae_accelsim { namespace standalone {
struct MemoryServices;
struct MemoryNativeAccess;
struct CycleView;
class MemoryStatsObserver;

struct L2Metrics {
  std::uint64_t accesses = 0, misses = 0, hits = 0;
  std::uint64_t pending_hits = 0, reservation_fails = 0;
  std::uint64_t port_available_cycles = 0, data_port_busy_cycles = 0;
  std::uint64_t fill_port_busy_cycles = 0;
  std::uint64_t tracked = 0, ingress = 0, dram_requests = 0;
  std::uint64_t dram_replies = 0, responses = 0, rop = 0;
};
struct DramMetrics {
  std::uint64_t commands = 0, activity = 0, nop = 0, act = 0, pre = 0;
  std::uint64_t reads = 0, writes = 0, writebacks = 0, requests = 0;
  std::uint64_t delay = 0, controller = 0, read_write = 0, replies = 0;
  std::uint64_t scheduled_reads = 0, scheduled_writes = 0, bank_requests = 0;
};

// Statistics depend on endpoint counts, not shader/kernel configuration. The
// caller owns the returned object through destruction after all components.
std::unique_ptr<memory_stats_t> make_memory_statistics(
    unsigned endpoints, unsigned warps_per_endpoint, const memory_config&,
    const CycleView&, MemoryStatsObserver* = nullptr);

class L2Component {
 public:
  ~L2Component();
  L2Component(const L2Component&) = delete;
  L2Component& operator=(const L2Component&) = delete;
  unsigned id() const;
  bool can_accept(const ae_dynamic::Transaction&) const;
  bool accept(const ae_dynamic::Transaction&);
  // Call once in the scheduled reply phase. Retires native writeback heads;
  // true means that native top() returned a response in this phase.
  bool response_cycle();
  bool peek_response(ae_dynamic::Transaction&) const;
  bool pop_response(std::uint64_t expected_id);
  bool peek_dram_request(ae_dynamic::Transaction&) const;
  void tick();
  bool drained() const;
  L2Metrics metrics() const;
 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
  explicit L2Component(std::unique_ptr<Impl>);
  friend struct MemoryNativeAccess;
  friend std::unique_ptr<L2Component> make_l2(
      unsigned, const memory_config&, memory_stats_t&, MemoryServices&);
};

class DramComponent {
 public:
  ~DramComponent();
  DramComponent(const DramComponent&) = delete;
  DramComponent& operator=(const DramComponent&) = delete;
  unsigned id() const;
  // A read-only admission predicate for a request at its linked L2 head.
  // Admission is committed by tick's unchanged native arbitration algorithm.
  bool can_accept(const ae_dynamic::Transaction&) const;
  void tick();
  bool drained() const;
  DramMetrics metrics() const;
 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
  explicit DramComponent(std::unique_ptr<Impl>);
  friend struct MemoryNativeAccess;
  friend std::unique_ptr<DramComponent> make_dram(
      unsigned, const memory_config&, memory_stats_t&, MemoryServices&,
      const std::vector<L2Component*>&);
};

std::unique_ptr<L2Component> make_l2(
    unsigned subpartition, const memory_config&, memory_stats_t&,
    MemoryServices&);
// Exactly the partition's configured subpartitions in native local order;
// borrowed explicitly. Destroy this DRAM component before these L2 objects.
std::unique_ptr<DramComponent> make_dram(
    unsigned partition, const memory_config&, memory_stats_t&, MemoryServices&,
    const std::vector<L2Component*>& subpartitions);

}}  // namespace ae_accelsim::standalone
#endif
