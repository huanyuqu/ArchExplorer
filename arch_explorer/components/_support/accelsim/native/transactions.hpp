// SPDX-License-Identifier: Apache-2.0
#ifndef ARCH_EXPLORER_ACCELSIM_NATIVE_TRANSACTIONS_HPP
#define ARCH_EXPLORER_ACCELSIM_NATIVE_TRANSACTIONS_HPP

#include "memory_services.hpp"
#include <cstdint>
#include <map>
#include <set>
#include <string>

class memory_config;

namespace ae_accelsim { namespace standalone {

// Implementation-only native payload domain. Runtime ports never expose its
// pointers. Its lifetime covers all components, configuration and contexts.
class NativeRequests final : public MemoryRequestAccess, public TransactionObserver {
 public:
  struct Life {
    std::string owner;
    bool physical = false, consumer = false, resource = false, cancelled = false;
  };
  NativeRequests() = default;
  ~NativeRequests();
  NativeRequests(const NativeRequests&) = delete;
  NativeRequests& operator=(const NativeRequests&) = delete;
  void bind(ae_dynamic::Runtime&);
  mem_fetch* resolve(std::uint64_t id) override;
  ae_dynamic::Transaction describe(mem_fetch*) override;
  void allocated(unsigned id, mem_fetch*) override;
  void destroyed(unsigned id) noexcept override;
  static void validate_source_request(std::uint64_t address, unsigned bytes);
  ae_dynamic::Transaction create(std::uint64_t address, unsigned bytes, bool write,
      const memory_config&, gpgpu_context&, const CycleView&);
  bool route(const ae_dynamic::Transaction&, bool physical_complete = false);
  void physical(std::uint64_t id);
  void complete_destroyed();
  void check() const;
  void cancel() noexcept;
  std::uint64_t pending() const;
  std::uint64_t native_deleted() const { return native_deleted_; }
  std::uint64_t allocated_count() const { return allocated_count_; }
  std::uint64_t destroyed_count() const { return destroyed_count_; }
  std::uint64_t live_count() const { return payload_.size(); }
  std::uint64_t physical_count() const { return physical_count_; }
  std::uint64_t consumer_count() const { return consumer_count_; }
  std::uint64_t resource_count() const { return resource_count_; }
  const std::map<std::uint64_t,Life>& lifetimes() const { return life_; }

 private:
  ae_dynamic::Runtime* runtime_ = nullptr;
  std::map<std::uint64_t,mem_fetch*> payload_;
  std::map<std::uint64_t,Life> life_;
  std::set<std::uint64_t> deleted_;
  unsigned route_depth_ = 0;
  bool closing_ = false;
  const char* fault_ = nullptr;
  std::uint64_t native_deleted_ = 0, physical_count_ = 0;
  std::uint64_t allocated_count_ = 0, destroyed_count_ = 0;
  std::uint64_t consumer_count_ = 0, resource_count_ = 0;
};

}}  // namespace ae_accelsim::standalone
#endif
