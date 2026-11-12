// SPDX-License-Identifier: Apache-2.0
#ifndef ARCH_EXPLORER_ACCELSIM_MEMORY_SERVICES_HPP
#define ARCH_EXPLORER_ACCELSIM_MEMORY_SERVICES_HPP

#include "services.hpp"
#include "runtime.hpp"

class gpgpu_context;
class cache_stats;
class mem_fetch;
class memory_sub_partition;
class memory_partition_unit;
class memory_stats_t;

namespace ae_accelsim { namespace standalone {

// This is a component-local dependency, never a simulation/GPU facade. The
// context supplies only mem_access_t's access-ID namespace. The aggregate is
// optional; each native cache continues to keep its complete own cache_stats.
struct CacheServices {
  const CycleView& clock;
  gpgpu_context& context;
  cache_stats* aggregate = nullptr;
};

// Private native bridge. Public ports carry only Transaction/ID. The assembly
// owns the arena and validates that both directions refer to its own domain.
class MemoryRequestAccess {
 public:
  virtual ~MemoryRequestAccess() = default;
  virtual mem_fetch* resolve(std::uint64_t id) = 0;
  virtual ae_dynamic::Transaction describe(mem_fetch*) = 0;
};

// A synchronous, side-effect-free refusal boundary. On success the original
// native transfer runs exactly once, after this call. The Runtime endpoint
// acknowledges admission; it must not also push/pop the native queues.
// subpartition and partition identify actual explicitly connected nodes.
class MemoryRoutes {
 public:
  virtual ~MemoryRoutes() = default;
  virtual bool transfer(unsigned subpartition, unsigned partition, bool reply,
                        const ae_dynamic::Transaction&) = 0;
};

// Optional observation sinks replace AerialVision/PTX global statistics. They
// do not drive queues or clocks, and must be scoped to the same assembly.
class MemoryStatsObserver {
 public:
  virtual ~MemoryStatsObserver() = default;
  virtual void latency(unsigned endpoint, unsigned latency) {}
  virtual void access(unsigned endpoint, unsigned partition, unsigned bank,
                      bool write) {}
  virtual void traffic(std::uint64_t pc, unsigned bytes) {}
};

struct MemoryServices {
  CacheServices cache;
  MemoryRequestAccess& requests;
  MemoryRoutes* routes = nullptr;
  MemoryStatsObserver* statistics = nullptr;
};

class L2Component;
class DramComponent;
struct L2Metrics;
struct DramMetrics;
// Implementation-only access for adapters gathering the native metrics schema.
// No pointer returned here may cross a public Runtime connection.
struct MemoryNativeAccess {
  static memory_sub_partition& l2(L2Component&);
  static memory_partition_unit& dram(DramComponent&);
  static L2Metrics metrics(const memory_sub_partition&);
  static DramMetrics metrics(const memory_partition_unit&);
  static bool drained(const memory_sub_partition&);
  static bool drained(const memory_partition_unit&);
  static mem_fetch* peek_response(const memory_sub_partition&);
  static bool simple(const memory_partition_unit&);
  static bool can_accept(const memory_sub_partition&, mem_fetch*);
};

}}  // namespace ae_accelsim::standalone
#endif
