// SPDX-License-Identifier: Apache-2.0
#include "memory_system.hpp"
#include "memory_links.hpp"
#include "transactions.hpp"
#include "../gpu-sim.h"
#include "../mem_fetch.h"
#include "../mem_latency_stat.h"
#include "../../../libcuda/gpgpu_context.h"
#include <cmath>
#include <algorithm>
#include <deque>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>

namespace ae_accelsim { namespace standalone {
using ae_dynamic::Transaction;
using ae_dynamic::Port;
using ae_dynamic::Direction;

namespace {
class RequestSource final : public ae_dynamic::Component {
 public:
  struct Group {
    std::uint64_t address;
    unsigned bytes, remaining;
    bool write;
    Transaction reply;
  };
  NativeRequests& requests;
  MemoryLinks& links;
  const std::vector<std::unique_ptr<L2Component>>& l2;
  std::deque<Transaction> waiting,receipts;
  std::set<std::uint64_t> completed;
  std::map<std::uint64_t,std::uint64_t> fragments;
  std::map<std::uint64_t,Group> groups;
  std::uint64_t submitted=0,received=0,rejected=0;
  RequestSource(NativeRequests& r,MemoryLinks& p,
                const std::vector<std::unique_ptr<L2Component>>& nodes):requests(r),links(p),l2(nodes) {}
  bool can_accept(const Transaction& tx,const ae_dynamic::Context&) const override {
    return tx.destination_port=="reply" && fragments.count(tx.id) && !completed.count(tx.id);
  }
  bool accept(const Transaction& tx,const ae_dynamic::Context& context) override {
    if(!can_accept(tx,context))return false;
    const auto id=fragments.at(tx.id);
    auto& group=groups.at(id);
    if(tx.bytes!=32 || tx.address<group.address || tx.address-group.address>group.bytes-32 ||
       tx.operation!=(group.write?"write_ack":"read_reply") || tx.parent_id)
      throw std::logic_error("sector response differs from its source submission");
    if(group.reply.id==0) {
      group.reply=tx;group.reply.id=id;group.reply.address=group.address;
      group.reply.bytes=group.bytes;group.reply.transfer_bytes=0;
      std::fill(group.reply.byte_mask.begin(),group.reply.byte_mask.end(),0);
    }
    if(group.reply.byte_mask.size()!=tx.byte_mask.size())
      throw std::logic_error("sector response byte-mask geometry changed");
    group.reply.transfer_bytes+=tx.transfer_bytes;
    for(unsigned i=0;i<tx.byte_mask.size();++i)group.reply.byte_mask[i]|=tx.byte_mask[i];
    completed.insert(tx.id);fragments.erase(tx.id);
    if(!--group.remaining){receipts.push_back(group.reply);groups.erase(id);++received;}
    return true;
  }
  bool drained() const override { return waiting.empty() && groups.empty(); }
  void submit(std::uint64_t address,unsigned bytes,bool write,const std::vector<Transaction>& packets) {
    const auto id=packets.front().id;
    auto next=waiting;
    for(const auto& tx:packets)next.push_back(tx);
    try {
      if(!groups.emplace(id,Group{address,bytes,unsigned(packets.size()),write,{}}).second)
        throw std::logic_error("duplicate source submission identity");
      for(const auto& tx:packets)if(!fragments.emplace(tx.id,id).second)
        throw std::logic_error("duplicate source sector identity");
    } catch (...) {
      groups.erase(id);for(const auto& tx:packets)fragments.erase(tx.id);throw;
    }
    waiting.swap(next);++submitted;
  }
  void send() {
    // FIFO ownership stays at this source until the actual L2 accepts it.
    for(unsigned burst=0;burst<8 && !waiting.empty();++burst) {
      auto tx=waiting.front();
      const unsigned target=requests.resolve(tx.id)->get_sub_partition_id();
      if(target>=l2.size())throw std::logic_error("native address selects an absent L2");
      if(!links.forward(tx,{"source","request","l2_"+std::to_string(target),"request"})) {
        ++rejected;break;
      }
      waiting.pop_front();
    }
  }
  void collect() {
    for(const auto& node:l2) {
      Transaction tx;
      if(!node->response_cycle())continue;
      if(!node->peek_response(tx))continue;
      if(!links.forward(tx,{"l2_"+std::to_string(node->id()),"response","source","reply"},true))continue;
      // The native queue still owns its head throughout receiver admission.
      // Pop it first, then destroy/acknowledge its actual native allocation.
      if(!node->pop_response(tx.id))throw std::logic_error("native response head changed after accepted route");
      delete requests.resolve(tx.id);
      requests.complete_destroyed();
    }
  }
};
}

struct MemorySystem::Impl {
  gpgpu_context context{true,false};
  memory_config config{&context};
  CycleView clock;
  RequestIds ids;
  NativeRequests requests;
  ae_dynamic::Runtime runtime;
  MemoryServices services{{clock,context,nullptr},requests,nullptr,nullptr};
  std::unique_ptr<memory_stats_t> statistics;
  std::vector<std::unique_ptr<L2Component>> l2;
  std::vector<std::unique_ptr<DramComponent>> dram;
  MemoryLinks links{requests};
  RequestSource source{requests,links,l2};
  std::vector<std::unique_ptr<L2Endpoint>> l2_endpoints;
  std::vector<std::unique_ptr<DramEndpoint>> dram_endpoints;
  explicit Impl(const MemoryOptions& options) {
    auto power_two=[](unsigned n){return n && !(n&(n-1));};
    if(!power_two(options.channels) || options.channels>64 || !power_two(options.subpartitions_per_channel) ||
       options.subpartitions_per_channel>4 || !options.controller_queue || options.controller_queue>65536 ||
       !power_two(options.cache_sets) || options.cache_sets>65536 || !options.cache_ways || options.cache_ways>1024 ||
       std::uint64_t(options.cache_sets)*options.cache_ways*128>1024ULL*1024*1024 ||
       options.rop_latency>1000000 || options.dram_latency>1000000 ||
       !std::isfinite(options.cache_period) || options.cache_period<=0 ||
       !std::isfinite(options.dram_period) || options.dram_period<=0)
      throw std::invalid_argument("invalid standalone memory dimensions/periods");
    auto parser=option_parser_create();
    config.reg_options(parser);
    std::ostringstream values;
    values<<"-gpgpu_n_mem "<<options.channels
      <<";-gpgpu_n_sub_partition_per_mchannel "<<options.subpartitions_per_channel
      <<";-gpgpu_lrc_enabled 0;-gpgpu_cache:dl2_texture_only 0;-gpgpu_memory_partition_indexing 0"
      <<";-gpgpu_simple_dram_model "<<options.simple_dram
      <<";-dram_latency "<<options.dram_latency<<";-gpgpu_l2_rop_latency "<<options.rop_latency
      <<";-gpgpu_frfcfs_dram_sched_queue_size "<<options.controller_queue
      <<";-gpgpu_cache:dl2 S:"<<options.cache_sets<<":128:"<<options.cache_ways
      <<",L:B:m:L:L,A:16:4,4:0,32;";
    option_parser_delimited_string(parser,values.str().c_str()," ;");
    option_parser_destroy(parser);
    config.ae_request_ids=&ids;config.ae_transaction_observer=&requests;
    config.init(nullptr);
    services.routes=&links;
    statistics=make_memory_statistics(1,0,config,clock);
    requests.bind(runtime);
    runtime.add_component("source",source,{{"request",Direction::output},{"reply",Direction::input}});
    runtime.add_clock("cache",options.cache_period);runtime.add_clock("dram",options.dram_period);
    for(unsigned id=0;id<config.m_n_mem_sub_partition;++id) {
      l2.push_back(make_l2(id,config,*statistics,services));
      auto endpoint=std::make_unique<L2Endpoint>(*l2.back(),links);
      const std::string name="l2_"+std::to_string(id);
      runtime.add_component(name,*endpoint,{{"request",Direction::input},{"response",Direction::output},
          {"dram_request",Direction::output},{"dram_reply",Direction::input}});
      l2_endpoints.push_back(std::move(endpoint));
      runtime.connect("source","request",name,"request");runtime.connect(name,"response","source","reply");
      auto* node=l2.back().get();auto* arena=&requests;
      runtime.add_phase("cache",20,name,[node,arena](const ae_dynamic::Context&){node->tick();arena->complete_destroyed();});
    }
    for(unsigned id=0;id<config.m_n_mem;++id) {
      std::vector<L2Component*> peers;
      for(unsigned local=0;local<config.m_n_sub_partition_per_memory_channel;++local)
        peers.push_back(l2[id*config.m_n_sub_partition_per_memory_channel+local].get());
      dram.push_back(make_dram(id,config,*statistics,services,peers));
      auto endpoint=std::make_unique<DramEndpoint>(*dram.back(),links);
      const std::string name="dram_"+std::to_string(id);
      runtime.add_component(name,*endpoint,{{"request",Direction::input},{"reply",Direction::output}});
      dram_endpoints.push_back(std::move(endpoint));
      for(auto* peer:peers) {
        const std::string cache="l2_"+std::to_string(peer->id());
        runtime.connect(cache,"dram_request",name,"request");runtime.connect(name,"reply",cache,"dram_reply");
        links.connect(*peer,*dram.back(),{cache,"dram_request",name,"request"},{name,"reply",cache,"dram_reply"});
      }
      auto* node=dram.back().get();auto* arena=&requests;
      runtime.add_phase("dram",10,name,[node,arena](const ae_dynamic::Context&){node->tick();arena->complete_destroyed();});
    }
    auto* input=&source;auto* time=&clock;
    runtime.add_phase("cache",0,"source",[input](const ae_dynamic::Context&){input->collect();});
    runtime.add_phase("cache",30,"source",[input](const ae_dynamic::Context&){input->send();});
    runtime.add_phase("cache",40,"source",[time](const ae_dynamic::Context&){++time->current;});
  }
  ~Impl() { requests.cancel();dram.clear();l2.clear();statistics.reset(); }
};
MemorySystem::MemorySystem(const MemoryOptions& options):impl_(new Impl(options)) {}
MemorySystem::~MemorySystem()=default;
std::uint64_t MemorySystem::submit(std::uint64_t address,unsigned bytes,bool write) {
  auto& p=*impl_;
  p.requests.check();NativeRequests::validate_source_request(address,bytes);
  std::vector<Transaction> packets;packets.reserve(bytes/32);
  try {
    for(unsigned offset=0;offset<bytes;offset+=32)
      packets.push_back(p.requests.create(address+offset,32,write,p.config,p.context,p.clock));
    p.source.submit(address,bytes,write,packets);
  } catch (...) {
    for(const auto& tx:packets)delete p.requests.resolve(tx.id);
    p.requests.complete_destroyed();throw;
  }
  return packets.front().id;
}
void MemorySystem::tick() {impl_->runtime.tick();impl_->requests.complete_destroyed();}
bool MemorySystem::receive(Transaction& tx) {
  auto& q=impl_->source.receipts;if(q.empty())return false;
  tx=std::move(q.front());q.pop_front();return true;
}
bool MemorySystem::drained() const {return impl_->runtime.drained() && !impl_->requests.pending();}
MemorySnapshot MemorySystem::snapshot() const {
  auto& p=*impl_;MemorySnapshot out;
  out.cycles=p.clock.absolute();out.submitted=p.source.submitted;out.received=p.source.received;
  out.rejected=p.source.rejected;out.pending=p.requests.pending();out.drained=drained();
  out.physical=p.requests.physical_count();out.consumer=p.requests.consumer_count();out.resource=p.requests.resource_count();
  for(const auto& node:p.l2)out.l2.push_back(node->metrics());
  for(const auto& node:p.dram)out.dram.push_back(node->metrics());return out;
}
}}  // namespace
