// SPDX-License-Identifier: Apache-2.0
#ifndef ARCH_EXPLORER_ACCELSIM_MEMORY_LINKS_HPP
#define ARCH_EXPLORER_ACCELSIM_MEMORY_LINKS_HPP

#include "memory_api.hpp"
#include "memory_services.hpp"
#include "transactions.hpp"
#include <array>
#include <map>
#include <string>

namespace ae_accelsim { namespace standalone {

using NativeEdge = std::array<std::string,4>;
struct EdgeCounts {
  std::uint64_t accepted=0,rejected=0,logical_bytes=0,transfer_bytes=0;
  std::uint64_t oracle_observed_bytes=0;
};

// Explicitly connected memory ports, not an assembly/service locator. These
// links own no native component and cannot advance one from an admission.
class MemoryLinks final : public MemoryRoutes {
 public:
  explicit MemoryLinks(NativeRequests& requests):requests_(requests) {}
  void connect(L2Component&,DramComponent&,NativeEdge request,NativeEdge reply);
  bool transfer(unsigned subpartition,unsigned partition,bool reply,
                const ae_dynamic::Transaction&) override;
  bool native_transfer(const ae_dynamic::Transaction&,bool reply) const;
  bool forward(ae_dynamic::Transaction,NativeEdge,bool physical_complete=false,
               std::uint64_t oracle_observed_bytes=UINT64_MAX);
  void declare(const NativeEdge& edge) { counters_.emplace(edge,EdgeCounts{}); }
  const std::map<NativeEdge,EdgeCounts>& counters() const { return counters_; }
 private:
  struct Link {L2Component* l2;DramComponent* dram;NativeEdge request,reply;};
  NativeRequests& requests_;
  std::map<unsigned,Link> links_;
  std::map<NativeEdge,EdgeCounts> counters_;
  const ae_dynamic::Transaction* active_=nullptr;
  bool replying_=false;
};

class L2Endpoint final : public ae_dynamic::Component {
 public:
  L2Endpoint(L2Component& component,MemoryLinks& links):component_(component),links_(links) {}
  bool can_accept(const ae_dynamic::Transaction&,const ae_dynamic::Context&) const override;
  bool accept(const ae_dynamic::Transaction&,const ae_dynamic::Context&) override;
  bool drained() const override { return component_.drained(); }
 private:
  L2Component& component_;
  MemoryLinks& links_;
};

class DramEndpoint final : public ae_dynamic::Component {
 public:
  DramEndpoint(DramComponent& component,MemoryLinks& links):component_(component),links_(links) {}
  bool can_accept(const ae_dynamic::Transaction&,const ae_dynamic::Context&) const override;
  bool accept(const ae_dynamic::Transaction&,const ae_dynamic::Context&) override;
  bool drained() const override { return component_.drained(); }
 private:
  DramComponent& component_;
  MemoryLinks& links_;
};

}}  // namespace ae_accelsim::standalone
#endif
