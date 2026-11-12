// SPDX-License-Identifier: Apache-2.0
#ifndef ARCH_EXPLORER_ACCELSIM_MEMORY_SYSTEM_HPP
#define ARCH_EXPLORER_ACCELSIM_MEMORY_SYSTEM_HPP
#include "memory_api.hpp"
#include <memory>
#include <string>

namespace ae_accelsim { namespace standalone {
// Convenience assembly of the same public L2/DRAM factories used by the GPU.
// No shader, program, warp or CTA services are constructed by this assembly.
struct MemoryOptions {
  unsigned channels=2, subpartitions_per_channel=1;
  unsigned cache_sets=4, cache_ways=1;
  bool simple_dram=false;
  unsigned dram_latency=30, rop_latency=2;
  unsigned controller_queue=32;
  double cache_period=1.0, dram_period=2.0;
};
struct MemorySnapshot {
  std::uint64_t cycles=0, submitted=0, received=0, rejected=0;
  std::uint64_t physical=0, consumer=0, resource=0, pending=0;
  bool drained=false;
  std::vector<L2Metrics> l2;
  std::vector<DramMetrics> dram;
};
class MemorySystem {
 public:
  explicit MemorySystem(const MemoryOptions& = {});
  ~MemorySystem();
  MemorySystem(const MemorySystem&)=delete;
  MemorySystem& operator=(const MemorySystem&)=delete;
  // A logical, aligned 32..128B request contained in one 128B line. The
  // source emits real 32B native sectors and returns one correlated receipt
  // after all sectors reply. The handle is the first sector's instance ID.
  std::uint64_t submit(std::uint64_t address,unsigned bytes,bool write);
  void tick();
  // Passive FIFO of completed logical requests, not submission-order service.
  // id/address/bytes match submit, parent_id=0, byte masks are merged and
  // transfer_bytes sums the actual sector replies. Never mark Runtime stages
  // here: the internal arena owns each real sector's three-stage lifecycle.
  bool receive(ae_dynamic::Transaction&);
  bool drained() const;
  MemorySnapshot snapshot() const;
 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};
}}  // namespace
#endif
