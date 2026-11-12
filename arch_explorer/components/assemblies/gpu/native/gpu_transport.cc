// SPDX-License-Identifier: Apache-2.0
#include "gpu_transport.hpp"
#include "memory_services.hpp"
#include "../gpu-sim.h"
#include "../l2cache.h"
#include "../mem_fetch.h"
#include <stdexcept>
namespace ae_accelsim { namespace standalone {
using ae_dynamic::Transaction;
namespace {
IcntPacketType packet_type(mem_fetch& mf) {
  if(mf.is_reply())return mf.get_is_write()?IcntPacketType::WriteReply:IcntPacketType::ReadReply;
  return mf.get_is_write()?IcntPacketType::WriteRequest:IcntPacketType::ReadRequest;
}
}
void GpuTransport::bind(unsigned id) {
  if(id>=clusters_ || !bound_.insert(id).second)
    throw std::invalid_argument("duplicate/out-of-range SM transport endpoint");
}
bool GpuTransport::can_send(unsigned cluster,unsigned bytes) const {
  if(!bound_.count(cluster))throw std::invalid_argument("unbound native SM endpoint");
  return network_.has_buffer(cluster,bytes);
}
std::pair<unsigned,unsigned> GpuTransport::endpoints(const Transaction& tx) const {
  auto* mf=requests_.resolve(tx.id);
  return mf->is_reply()?std::make_pair(clusters_+mf->get_sub_partition_id(),mf->get_tpc()):
                        std::make_pair(mf->get_tpc(),clusters_+mf->get_sub_partition_id());
}
bool GpuTransport::can_accept(const Transaction& tx,const ae_dynamic::Context&) const {
  const auto pair=endpoints(tx);return network_.has_buffer(pair.first,tx.transfer_bytes);
}
bool GpuTransport::accept(const Transaction& tx,const ae_dynamic::Context& context) {
  if(!can_accept(tx,context))return false;
  const auto pair=endpoints(tx);
  return network_.push(pair.first,pair.second,tx,packet_type(*requests_.resolve(tx.id)));
}
void GpuTransport::send(unsigned cluster,unsigned memory_device,mem_fetch* mf,unsigned bytes) {
  if(cluster!=mf->get_tpc() || memory_device!=clusters_+mf->get_sub_partition_id())
    throw std::invalid_argument("SM transport destination disagrees with native address routing");
  auto tx=requests_.describe(mf);tx.transfer_bytes=bytes;
  if(!links_.forward(tx,{"sm"+std::to_string(cluster),"request","icnt","request"+std::to_string(cluster)}))
    throw std::logic_error("native SM injection lost previously checked credit");
}
bool GpuTransport::receiving(unsigned id,const Transaction& tx) const {
  return receiving_ && receiving_cluster_==id && tx.id==receiving_->id;
}
mem_fetch* GpuTransport::pop_reply(unsigned cluster) {
  const auto* head=network_.peek(cluster);if(!head)return nullptr;
  auto tx=*head;
  receiving_=&tx;receiving_cluster_=cluster;
  bool accepted;
  try {accepted=links_.forward(tx,{"icnt","reply"+std::to_string(cluster),"sm"+std::to_string(cluster),"reply"});}
  catch(...){receiving_=nullptr;throw;}
  receiving_=nullptr;if(!accepted)return nullptr;
  Transaction actual;
  if(!network_.pop(cluster,actual) || actual.id!=tx.id)
    throw std::logic_error("ICNT reply head changed after synchronous admission");
  return requests_.resolve(tx.id);
}
void GpuTransport::receive_l2(L2Component& node,std::uint64_t& stalls) {
  const auto output=clusters_+node.id();
  // Preserve the original GPU's worst-case sector admission rule. The raw
  // L2 factory also supports callers with an exact packet-size preflight.
  if(MemoryNativeAccess::l2(node).full(SECTOR_CHUNCK_SIZE)) {
    if(network_.peek(output))++stalls;
  }else if(const auto* head=network_.peek(output)) {
    auto tx=*head;
    // Preserve the frozen SIM38 egress observer in a separate diagnostic
    // view. Its formula used reply lengths even at request ejection. The
    // actual Runtime descriptor and ICNT counters keep the real packet size.
    auto* mf=requests_.resolve(tx.id);
    const auto observed=mf->get_is_write()?mf->get_ctrl_size():mf->size();
    if(!links_.forward(tx,{"icnt","request_mem"+std::to_string(node.id()),"l2"+std::to_string(node.id()),"request"},false,observed))
      throw std::logic_error("credited L2 input rejected its native head");
    Transaction actual;
    if(!network_.pop(output,actual) || actual.id!=tx.id)
      throw std::logic_error("ICNT request head changed after synchronous admission");
  }
  node.tick();requests_.complete_destroyed();
}
void GpuTransport::send_l2(L2Component& node,std::uint64_t& stalls) {
  if(!node.response_cycle()){requests_.complete_destroyed();return;}
  Transaction tx;if(!node.peek_response(tx))throw std::logic_error("native L2 response disappeared");
  if(links_.forward(tx,{"l2"+std::to_string(node.id()),"reply","icnt","reply_mem"+std::to_string(node.id())},true)) {
    if(!node.pop_response(tx.id))throw std::logic_error("accepted L2 response did not pop");
  }else ++stalls;
  requests_.complete_destroyed();
}
}}
