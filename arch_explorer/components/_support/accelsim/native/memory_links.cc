// SPDX-License-Identifier: Apache-2.0
#include "memory_links.hpp"
#include "../mem_fetch.h"
#include <stdexcept>

namespace ae_accelsim { namespace standalone {
using ae_dynamic::Transaction;

void MemoryLinks::connect(L2Component& l2,DramComponent& dram,NativeEdge request,NativeEdge reply) {
  if(links_.count(l2.id()) || request[0].empty() || request[2].empty() ||
     request[0]!=reply[2] || request[2]!=reply[0])
    throw std::invalid_argument("duplicate/mismatched explicit memory connection");
  links_.emplace(l2.id(),Link{&l2,&dram,request,reply});
  counters_[request];counters_[reply];
}

bool MemoryLinks::forward(Transaction tx,NativeEdge edge,bool complete,std::uint64_t observation) {
  tx.source=edge[0];tx.source_port=edge[1];
  tx.destination=edge[2];tx.destination_port=edge[3];
  const bool accepted=requests_.route(tx,complete);
  auto& count=counters_[edge];
  if(accepted) {
    ++count.accepted;count.logical_bytes+=tx.bytes;count.transfer_bytes+=tx.transfer_bytes;
    count.oracle_observed_bytes+=observation==UINT64_MAX?tx.transfer_bytes:observation;
  }else ++count.rejected;
  return accepted;
}

bool MemoryLinks::transfer(unsigned subpartition,unsigned partition,bool reply,const Transaction& value) {
  auto found=links_.find(subpartition);
  if(found==links_.end() || found->second.dram->id()!=partition || active_)
    throw std::invalid_argument("native transfer has no matching explicit memory link");
  auto tx=value;
  // The native L2/DRAM queue transfers the complete native memory fetch,
  // including its control bytes, in both directions. ICNT packet lengths
  // are selected separately by the requesting/replying network adapter.
  tx.transfer_bytes=requests_.resolve(tx.id)->size();
  const auto& edge=reply?found->second.reply:found->second.request;
  tx.source=edge[0];tx.source_port=edge[1];tx.destination=edge[2];tx.destination_port=edge[3];
  active_=&tx;replying_=reply;
  try {
    const bool accepted=forward(tx,edge,reply);
    active_=nullptr;
    return accepted;
  }catch(...) {active_=nullptr;throw;}
}

bool MemoryLinks::native_transfer(const Transaction& tx,bool reply) const {
  return active_ && replying_==reply && tx.id==active_->id && tx.parent_id==active_->parent_id &&
         tx.address==active_->address && tx.bytes==active_->bytes &&
         tx.source==active_->source && tx.source_port==active_->source_port &&
         tx.destination==active_->destination && tx.destination_port==active_->destination_port;
}

bool L2Endpoint::can_accept(const Transaction& tx,const ae_dynamic::Context&) const {
  if(tx.destination_port=="request")return component_.can_accept(tx);
  if(tx.destination_port=="dram_reply")return links_.native_transfer(tx,true);
  return false;
}
bool L2Endpoint::accept(const Transaction& tx,const ae_dynamic::Context& context) {
  if(!can_accept(tx,context))return false;
  // The original DRAM transfer following this acknowledgement writes the
  // native return queue. No second cache fill/push is performed here.
  return tx.destination_port=="dram_reply" || component_.accept(tx);
}

bool DramEndpoint::can_accept(const Transaction& tx,const ae_dynamic::Context&) const {
  return links_.native_transfer(tx,false) && component_.can_accept(tx);
}
bool DramEndpoint::accept(const Transaction& tx,const ae_dynamic::Context& context) {
  // Native arbitration already selected this L2 head/credit and performs its
  // original pop/delay/credit operation after a successful Runtime route.
  return can_accept(tx,context);
}

}}  // namespace ae_accelsim::standalone
