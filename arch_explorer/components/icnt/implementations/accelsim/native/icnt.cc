// SPDX-License-Identifier: Apache-2.0
#include "ae_component/icnt_api.hpp"
#include "ae_standalone/interconnect_interface.hpp"
#include "ae_standalone/local_interconnect.h"
#include "ae_standalone/ae_state.hpp"
#include <limits>
#include <map>
#include <stdexcept>
#include <utility>

namespace ae_accelsim { namespace standalone {
namespace {
namespace native = ae_standalone_booksim;
using ae_dynamic::Transaction;

// Opaque only inside this implementation. Public callers never supply a host
// pointer. The envelope outlives every native flit referring to that packet.
struct Packet { Transaction transaction; unsigned output; };

class Network {
 public:
  virtual ~Network() = default;
  virtual bool buffer(unsigned input, unsigned size) const = 0;
  virtual void push(unsigned input, unsigned output, Packet*, unsigned size,
                    IcntPacketType) = 0;
  virtual Packet* peek(unsigned output) const = 0;
  virtual Packet* pop(unsigned output) = 0;
  virtual void begin_interval() = 0;
  virtual void advance() = 0;
  virtual IcntMetrics metrics() const = 0;
};

class BookSim final : public Network {
 public:
  explicit BookSim(const IcntConfig& config)
      : network_(native::InterconnectInterface::New(config.booksim_config.c_str(),state_)) {
    network_->CreateInterconnect(config.request_nodes,config.memory_nodes);
    network_->Init();
  }
  bool buffer(unsigned input, unsigned size) const override {
    return network_->HasBuffer(input,size);
  }
  void push(unsigned input, unsigned output, Packet* packet, unsigned size,
            IcntPacketType type) override {
    native::Flit::FlitType kind;
    switch(type) {
      case IcntPacketType::ReadRequest: kind=native::Flit::READ_REQUEST; break;
      case IcntPacketType::WriteRequest: kind=native::Flit::WRITE_REQUEST; break;
      case IcntPacketType::ReadReply: kind=native::Flit::READ_REPLY; break;
      case IcntPacketType::WriteReply: kind=native::Flit::WRITE_REPLY; break;
      default: throw std::invalid_argument("invalid ICNT packet type");
    }
    network_->Push(input,output,packet,size,kind);
  }
  Packet* peek(unsigned output) const override {
    return static_cast<Packet*>(network_->AePeek(output));
  }
  Packet* pop(unsigned output) override {
    return static_cast<Packet*>(network_->Pop(output));
  }
  void begin_interval() override { network_->Init(); }
  void advance() override {
    if(network_->GetIcntTime()==std::numeric_limits<int>::max())
      throw std::overflow_error("BookSim native cycle exhausted");
    network_->Advance();
  }
  IcntMetrics metrics() const override {
    IcntMetrics m;
    m.mode=1; m.flit_bytes=network_->GetFlitSize();
    m.cycles=network_->GetIcntTime(); m.native_busy=network_->Busy();
    m.allocated_flits=state_.flit_all.size();
    m.allocated_credits=state_.credit_all.size();
    m.live_flits=m.allocated_flits-state_.flit_free.size();
    m.live_credits=m.allocated_credits-state_.credit_free.size();
    return m;
  }
 private:
  native::AeBookSimState state_;
  // Declared last: native queues/destructors finish before their owned state.
  std::unique_ptr<native::InterconnectInterface> network_;
};

class Local final : public Network {
 public:
  explicit Local(const IcntConfig& config) {
    native::inct_config value;
    value.in_buffer_limit=config.local.input_buffer;
    value.out_buffer_limit=config.local.output_buffer;
    value.subnets=2;
    value.arbiter_algo=static_cast<native::Arbiteration_type>(config.local.arbitration);
    value.verbose=0; value.grant_cycles=config.local.grant_cycles;
    network_.reset(native::LocalInterconnect::New(value));
    network_->CreateInterconnect(config.request_nodes,config.memory_nodes);
    network_->Init();
  }
  bool buffer(unsigned input, unsigned) const override { return network_->AeHasBuffer(input); }
  void push(unsigned input, unsigned output, Packet* packet, unsigned size,
            IcntPacketType) override { network_->Push(input,output,packet,size); }
  Packet* peek(unsigned output) const override {
    return static_cast<Packet*>(network_->AePeek(output));
  }
  Packet* pop(unsigned output) override { return static_cast<Packet*>(network_->Pop(output)); }
  void begin_interval() override { network_->Init(); }
  void advance() override {
    if(cycles_==std::numeric_limits<std::uint64_t>::max())
      throw std::overflow_error("LocalInterconnect cycle exhausted");
    network_->Advance(); ++cycles_;
  }
  IcntMetrics metrics() const override {
    IcntMetrics m; m.mode=2; m.flit_bytes=network_->GetFlitSize();
    m.cycles=cycles_; m.native_busy=network_->Busy();
    for(const auto* router : network_->AeRouters()) {
      IcntRouterMetrics r;
      r.cycles=router->cycles; r.conflicts=router->conflicts;
      r.conflicts_util=router->conflicts_util; r.cycles_util=router->cycles_util;
      r.reqs_util=router->reqs_util; r.out_buffer_full=router->out_buffer_full;
      r.out_buffer_util=router->out_buffer_util; r.in_buffer_full=router->in_buffer_full;
      r.in_buffer_util=router->in_buffer_util; r.packets=router->packets_num;
      m.local_routers.push_back(r);
    }
    return m;
  }
 private:
  std::unique_ptr<native::LocalInterconnect> network_;
  std::uint64_t cycles_=0;
};

class Instance final : public Icnt {
 public:
  explicit Instance(const IcntConfig& config) : config_(config) {
    if(config.mode==1) network_.reset(new BookSim(config));
    else network_.reset(new Local(config));
  }
  bool has_buffer(unsigned input, std::uint64_t bytes) const override {
    check();
    return endpoint(input) && size(bytes) && network_->buffer(input,static_cast<unsigned>(bytes));
  }
  bool push(unsigned input, unsigned output, const Transaction& transaction,
            IcntPacketType type) override {
    check();
    const bool request=type==IcntPacketType::ReadRequest || type==IcntPacketType::WriteRequest;
    const bool reply=type==IcntPacketType::ReadReply || type==IcntPacketType::WriteReply;
    if(!endpoint(input) || !endpoint(output) || !(request || reply) ||
       request!=(input<config_.request_nodes) || request==(output<config_.request_nodes) ||
       !transaction.id || packets_.count(transaction.id) ||
       !has_buffer(input,transaction.transfer_bytes)) return false;
    if(accepted_==std::numeric_limits<std::uint64_t>::max() ||
       transaction.transfer_bytes>std::numeric_limits<std::uint64_t>::max()-bytes_)
      throw std::overflow_error("ICNT transaction counters exhausted");
    // Both allocations complete before native injection. A rejected probe has
    // no side effects. Native allocation errors propagate, never become false.
    std::unique_ptr<Packet> packet(new Packet{transaction,output});
    auto inserted=packets_.emplace(transaction.id,std::move(packet));
    try {
      network_->push(input,output,inserted.first->second.get(),
                     static_cast<unsigned>(transaction.transfer_bytes),type);
    } catch(...) {
      // A native exception can follow a partial flit allocation. Retain the
      // envelope until network destruction; callers must discard this instance.
      faulted_=true;
      throw;
    }
    ++accepted_; bytes_+=transaction.transfer_bytes; return true;
  }
  const Transaction* peek(unsigned output) const override {
    check(); if(!endpoint(output)) return nullptr;
    Packet* packet=network_->peek(output);
    return packet ? &packet->transaction : nullptr;
  }
  bool pop(unsigned output, Transaction& result) override {
    check(); if(!endpoint(output)) return false;
    Packet* packet=network_->peek(output);
    if(!packet) return false;
    // A failed caller copy cannot consume the native packet or advance fairness.
    Transaction copy=packet->transaction;
    result=std::move(copy);
    if(network_->pop(output)!=packet || packet->output!=output)
      throw std::logic_error("native ICNT peek/pop identity disagreement");
    const auto id=packet->transaction.id;
    packets_.erase(id); ++delivered_; return true;
  }
  void begin_interval() override { check(); network_->begin_interval(); }
  void advance() override { check(); network_->advance(); }
  bool busy() const override {
    check(); const auto m=network_->metrics();
    return !packets_.empty() || m.native_busy || m.live_flits || m.live_credits;
  }
  IcntMetrics metrics() const override {
    check(); auto m=network_->metrics();
    m.accepted=accepted_; m.delivered=delivered_; m.transfer_bytes=bytes_;
    m.pending_packets=packets_.size(); return m;
  }
 private:
  bool endpoint(unsigned value) const { return value<config_.request_nodes+config_.memory_nodes; }
  static bool size(std::uint64_t bytes) {
    return bytes>0 && bytes<=static_cast<unsigned>(std::numeric_limits<int>::max());
  }
  void check() const {
    if(faulted_) throw std::logic_error("discard ICNT after native injection exception");
  }
  const IcntConfig config_;
  std::map<std::uint64_t,std::unique_ptr<Packet>> packets_;
  std::unique_ptr<Network> network_;  // dies before its opaque packet storage
  std::uint64_t accepted_=0, delivered_=0, bytes_=0;
  bool faulted_=false;
};
}  // namespace

std::unique_ptr<Icnt> make_icnt(const IcntConfig& config) {
  if(config.mode!=1 && config.mode!=2)
    throw std::invalid_argument("ICNT mode must be BookSim(1) or LocalInterconnect(2)");
  if(!config.request_nodes || !config.memory_nodes ||
     std::uint64_t(config.request_nodes)+config.memory_nodes>4096)
    throw std::invalid_argument("ICNT requires 1..4096 total endpoints in both directions");
  if(config.mode==2 && (!config.local.input_buffer || !config.local.output_buffer ||
      config.local.input_buffer>1048576 || config.local.output_buffer>1048576 ||
      config.local.arbitration>2 || !config.local.grant_cycles ||
      config.local.grant_cycles>1048576))
    throw std::invalid_argument("invalid native LocalInterconnect configuration");
  return std::unique_ptr<Icnt>(new Instance(config));
}
}}  // namespace ae_accelsim::standalone
