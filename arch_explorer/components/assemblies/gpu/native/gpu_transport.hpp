// SPDX-License-Identifier: Apache-2.0
#ifndef ARCH_EXPLORER_ACCELSIM_GPU_TRANSPORT_HPP
#define ARCH_EXPLORER_ACCELSIM_GPU_TRANSPORT_HPP
#include "icnt_api.hpp"
#include "memory_links.hpp"
#include "shader_api.hpp"
#include "shader_services.hpp"
#include <set>
#include <memory>
namespace ae_accelsim { namespace standalone {
class GpuTransport final : public ShaderTransport, public ae_dynamic::Component {
 public:
  GpuTransport(Icnt& network,NativeRequests& requests,MemoryLinks& links,unsigned clusters)
      :network_(network),requests_(requests),links_(links),clusters_(clusters) {}
  bool can_send(unsigned cluster,unsigned bytes) const override;
  void send(unsigned cluster,unsigned memory_device,mem_fetch*,unsigned bytes) override;
  mem_fetch* pop_reply(unsigned cluster) override;
  bool can_accept(const ae_dynamic::Transaction&,const ae_dynamic::Context&) const override;
  bool accept(const ae_dynamic::Transaction&,const ae_dynamic::Context&) override;
  bool drained() const override {return !network_.busy();}
  void bind(unsigned cluster);
  bool receiving(unsigned cluster,const ae_dynamic::Transaction&) const;
  void receive_l2(L2Component&,std::uint64_t& stalls);
  void send_l2(L2Component&,std::uint64_t& stalls);
 private:
  Icnt& network_;
  NativeRequests& requests_;
  MemoryLinks& links_;
  unsigned clusters_;
  std::set<unsigned> bound_;
  const ae_dynamic::Transaction* receiving_=nullptr;
  unsigned receiving_cluster_=0;
  std::pair<unsigned,unsigned> endpoints(const ae_dynamic::Transaction&) const;
};
class ShaderEndpoint final : public ae_dynamic::Component {
 public:
  ShaderEndpoint(unsigned id,ShaderCluster& node,GpuTransport& transport):id_(id),node_(node),transport_(transport) {}
  bool can_accept(const ae_dynamic::Transaction& tx,const ae_dynamic::Context&) const override {
    return transport_.receiving(id_,tx);
  }
  bool accept(const ae_dynamic::Transaction& tx,const ae_dynamic::Context& context) override {
    return can_accept(tx,context);
  }
  bool drained() const override {return node_.drained();}
 private:
  unsigned id_;
  ShaderCluster& node_;
  GpuTransport& transport_;
};
}}
#endif
