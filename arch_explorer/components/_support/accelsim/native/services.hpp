// SPDX-License-Identifier: Apache-2.0
#ifndef ARCH_EXPLORER_ACCELSIM_COMPONENT_SERVICES_HPP
#define ARCH_EXPLORER_ACCELSIM_COMPONENT_SERVICES_HPP

#include <cstdint>
#include <limits>
#include <stdexcept>

class mem_fetch;

namespace ae_accelsim { namespace standalone {
struct ConfigurationAccess;

// The assembly owns these counters. Components only observe them. A core
// phase increments current after the native core tail; a kernel boundary
// moves current into completed. Memory-only assemblies can choose the same
// reference clock without creating an instruction/CTA execution service.
struct CycleView {
  std::uint64_t current = 0;
  std::uint64_t completed = 0;
  std::uint64_t absolute() const { return completed + current; }
};

// Per assembly, never a process-global request sequence. Distinct Runtime
// instances have independent namespaces. No clock/allocator/GPU singleton.
class RequestIds {
 public:
  unsigned next() {
    if (next_ == std::numeric_limits<unsigned>::max())
      throw std::overflow_error("native request identity exhausted");
    return next_++;
  }
 private:
  unsigned next_ = 1;
};

// An allocation domain observes destruction of its own native transaction.
// This cannot tick a component or throw from a native destructor. The owner
// records protocol faults and propagates them at its next explicit boundary.
class TransactionObserver {
 public:
  virtual ~TransactionObserver() = default;
  // Component-domain allocation ownership only; never a public Runtime wire.
  // Register after construction fields are initialized. Failure propagates
  // from the constructor before any queue can retain this object.
  virtual void allocated(unsigned request_id, mem_fetch*) = 0;
  virtual void destroyed(unsigned request_id) noexcept = 0;
};

}}  // namespace ae_accelsim::standalone
#endif
