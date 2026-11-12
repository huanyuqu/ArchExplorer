// SPDX-License-Identifier: Apache-2.0
// External caller: native ICNT instances, ordinary endpoints and public Runtime.
// No SM, kernel, GPU host, cache model or native queue implementation lives here.
#include "icnt_api.hpp"
#include <deque>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace example {
namespace native = ae_accelsim::standalone;
using ae_dynamic::Component;
using ae_dynamic::Context;
using ae_dynamic::Direction;
using ae_dynamic::Runtime;
using ae_dynamic::Stage;
using ae_dynamic::Transaction;

void require(bool value, const char* message) {
  if (!value) throw std::runtime_error(message);
}

bool same_payload(const Transaction& a, const Transaction& b) {
  return a.id == b.id && a.parent_id == b.parent_id &&
         a.address == b.address && a.bytes == b.bytes &&
         a.transfer_bytes == b.transfer_bytes && a.byte_mask == b.byte_mask &&
         a.operation == b.operation && a.ordering == b.ordering;
}

std::string metrics_json(const native::IcntMetrics& m) {
  std::ostringstream out;
  out << "{\"mode\":" << m.mode << ",\"flit_bytes\":" << m.flit_bytes
      << ",\"cycles\":" << m.cycles << ",\"accepted\":" << m.accepted
      << ",\"delivered\":" << m.delivered << ",\"transfer_bytes\":" << m.transfer_bytes
      << ",\"pending_packets\":" << m.pending_packets << ",\"live_flits\":" << m.live_flits
      << ",\"live_credits\":" << m.live_credits << ",\"allocated_flits\":" << m.allocated_flits
      << ",\"allocated_credits\":" << m.allocated_credits
      << ",\"native_busy\":" << (m.native_busy ? "true" : "false")
      << ",\"local_routers\":[";
  for (std::size_t i = 0; i < m.local_routers.size(); ++i) {
    const auto& r = m.local_routers[i];
    if (i) out << ',';
    out << "{\"cycles\":" << r.cycles << ",\"conflicts\":" << r.conflicts
        << ",\"conflicts_util\":" << r.conflicts_util << ",\"cycles_util\":" << r.cycles_util
        << ",\"reqs_util\":" << r.reqs_util << ",\"out_buffer_full\":" << r.out_buffer_full
        << ",\"out_buffer_util\":" << r.out_buffer_util << ",\"in_buffer_full\":" << r.in_buffer_full
        << ",\"in_buffer_util\":" << r.in_buffer_util << ",\"packets\":" << r.packets << '}';
  }
  out << "]}";
  return out.str();
}

// This is a transport workload endpoint, not a memory simulation. It echoes
// accepted requests on its own phase, with an explicit delayed receive policy.
class EchoEndpoint final : public Component {
 public:
  EchoEndpoint(Runtime& runtime, std::uint64_t ready_after)
      : runtime_(runtime), ready_after_(ready_after) {}
  bool can_accept(const Transaction& t, const Context& context) const override {
    return context.tick >= ready_after_ && replies_.size() < 1 &&
           (t.operation == "read" || t.operation == "write");
  }
  bool accept(const Transaction& t, const Context& context) override {
    if (!can_accept(t, context)) return false;
    Transaction reply = t;
    reply.operation += "_reply";
    reply.source = "echo"; reply.source_port = "reply";
    reply.destination = "icnt"; reply.destination_port = "response";
    replies_.push_back(std::move(reply));
    ++received_;
    return true;
  }
  void advance() {
    if (replies_.empty()) return;
    if (runtime_.route(replies_.front())) replies_.pop_front();
    else ++retries_;
  }
  bool drained() const override { return replies_.empty(); }
  std::uint64_t received() const { return received_; }
  std::uint64_t retries() const { return retries_; }
 private:
  Runtime& runtime_;
  const std::uint64_t ready_after_;
  std::deque<Transaction> replies_;
  std::uint64_t received_ = 0, retries_ = 0;
};

class Source final : public Component {
 public:
  Source(Runtime& runtime, unsigned count, std::uint64_t base,
         std::uint64_t reply_ready_after)
      : runtime_(runtime), count_(count), base_(base), reply_ready_after_(reply_ready_after) {}
  bool can_accept(const Transaction&, const Context& context) const override {
    return context.tick >= reply_ready_after_;
  }
  bool accept(const Transaction& t, const Context& context) override {
    if (!can_accept(t, context)) return false;
    const auto found = outstanding_.find(t.id);
    require(found != outstanding_.end(), "unknown or repeated response identity");
    auto expected = found->second;
    expected.operation += "_reply";
    require(same_payload(t, expected), "response payload/mask/correlation changed");
    receipts_.push_back(t);
    ready_.push_back(t.id);
    outstanding_.erase(found);
    return true;
  }
  void advance() {
    // ICNT transports only packets. After the reply has been popped from the
    // native network, this endpoint owns the completed round-trip transaction.
    // This rule is not reused for caches/DRAM, whose native tail is separate.
    for (auto id : ready_) {
      runtime_.mark(id, "source", Stage::consumer_ready);
      runtime_.mark(id, "source", Stage::physical_complete);
      runtime_.mark(id, "source", Stage::resource_ready);
    }
    ready_.clear();
    for (unsigned burst = 0; burst < 4 && sent_ < count_; ++burst) {
      Transaction t;
      t.id = sent_ + 1;
      t.source = "source"; t.source_port = "request";
      t.destination = "icnt"; t.destination_port = "request";
      t.operation = t.id % 3 ? "read" : "write";
      t.address = t.id < 4 ? t.id * 32 : base_ + t.id * 32;
      t.bytes = t.transfer_bytes = 32;
      t.byte_mask.assign(32, 1);
      t.ordering = "source-id";
      if (!runtime_.route(t)) { ++retries_; break; }
      require(outstanding_.emplace(t.id, t).second, "reused outstanding identity");
      ++sent_;
    }
  }
  bool drained() const override {
    return sent_ == count_ && outstanding_.empty() && ready_.empty();
  }
  std::uint64_t sent() const { return sent_; }
  std::uint64_t retries() const { return retries_; }
  const std::vector<Transaction>& receipts() const { return receipts_; }
 private:
  Runtime& runtime_;
  const unsigned count_;
  const std::uint64_t base_, reply_ready_after_;
  std::uint64_t sent_ = 0, retries_ = 0;
  std::map<std::uint64_t, Transaction> outstanding_;
  std::vector<Transaction> receipts_;
  std::vector<std::uint64_t> ready_;
};

// A caller-side port adapter only. Native make_icnt owns all queue, arbitration,
// credit and advancement mechanisms; the Runtime owns time and route delivery.
class Network final : public Component {
 public:
  Network(Runtime& runtime, const native::IcntConfig& config)
      : runtime_(runtime), native_(native::make_icnt(config)) {}
  bool can_accept(const Transaction& t, const Context&) const override {
    return native_->has_buffer(input(t), t.transfer_bytes);
  }
  bool accept(const Transaction& t, const Context&) override {
    const auto port = input(t);
    const bool write = t.operation == "write" || t.operation == "write_reply";
    const auto kind = port == 0 ? (write ? native::IcntPacketType::WriteRequest : native::IcntPacketType::ReadRequest)
                                : (write ? native::IcntPacketType::WriteReply : native::IcntPacketType::ReadReply);
    return native_->push(port, 1 - port, t, kind);
  }
  void advance() {
    native_->advance();
    for (unsigned output : {1U, 0U}) {
      while (const Transaction* head = native_->peek(output)) {
        const Transaction original = *head;
        Transaction routed = original;
        routed.source = "icnt";
        routed.source_port = output ? "to_echo" : "to_source";
        routed.destination = output ? "echo" : "source";
        routed.destination_port = output ? "request" : "reply";
        if (!runtime_.route(routed)) {
          require(native_->peek(output) && same_payload(*native_->peek(output), original),
                  "rejected delivery consumed or changed native head");
          break;
        }
        Transaction popped;
        require(native_->pop(output, popped) && same_payload(popped, original),
                "native peek/pop changed packet identity");
      }
    }
  }
  bool drained() const override { return !native_->busy(); }
  native::IcntMetrics metrics() const { return native_->metrics(); }
 private:
  static unsigned input(const Transaction& t) {
    if (t.destination_port == "request" && (t.operation == "read" || t.operation == "write")) return 0;
    if (t.destination_port == "response" && (t.operation == "read_reply" || t.operation == "write_reply")) return 1;
    throw std::invalid_argument("invalid endpoint packet operation/port");
  }
  Runtime& runtime_;
  std::unique_ptr<native::Icnt> native_;
};

struct Session {
  // Runtime/configuration outlive the registered component objects.
  Runtime runtime;
  native::IcntConfig config;
  Network network;
  EchoEndpoint echo;
  Source source;
  const unsigned request_count;
  const std::uint64_t address_base, receive_delay;
  const double period_ns;
  Session(unsigned buffers, unsigned count, std::uint64_t base,
          std::uint64_t receive_delay, double period)
      : runtime(100000), config(configuration(buffers)), network(runtime, config),
        echo(runtime, receive_delay), source(runtime, count, base, receive_delay + 32),
        request_count(count), address_base(base), receive_delay(receive_delay), period_ns(period) {
    runtime.add_component("source", source, {{"request", Direction::output}, {"reply", Direction::input}});
    runtime.add_component("icnt", network, {{"request", Direction::input}, {"response", Direction::input},
                                           {"to_echo", Direction::output}, {"to_source", Direction::output}});
    runtime.add_component("echo", echo, {{"request", Direction::input}, {"reply", Direction::output}});
    runtime.connect("source", "request", "icnt", "request");
    runtime.connect("icnt", "to_echo", "echo", "request");
    runtime.connect("echo", "reply", "icnt", "response");
    runtime.connect("icnt", "to_source", "source", "reply");
    runtime.add_clock("link", period);
    runtime.add_phase("link", 0, "icnt", [&node = network](const Context&) { node.advance(); });
    runtime.add_phase("link", 1, "echo", [&node = echo](const Context&) { node.advance(); });
    runtime.add_phase("link", 2, "source", [&node = source](const Context&) { node.advance(); });
  }
  static native::IcntConfig configuration(unsigned buffers) {
    native::IcntConfig c;
    c.mode = 2; c.request_nodes = c.memory_nodes = 1;
    c.local.input_buffer = c.local.output_buffer = buffers;
    return c;
  }
  std::string report(unsigned expected) const {
    const auto m = network.metrics();
    require(runtime.drained() && source.receipts().size() == expected && echo.received() == expected,
            "endpoint or physical network did not drain");
    require(m.accepted == 2 * expected && m.delivered == 2 * expected &&
            !m.pending_packets && !m.live_flits && !m.live_credits && !m.native_busy,
            "native packet accounting or drain differs");
    require(runtime.audit().completed == expected && source.retries() > 0 && runtime.audit().rejected > 0,
            "round-trip lifecycle or actual backpressure not exercised");
    std::ostringstream out;
    out << std::setprecision(17) << "{\"input_buffer\":" << config.local.input_buffer
        << ",\"output_buffer\":" << config.local.output_buffer
        << ",\"request_count\":" << request_count << ",\"address_base\":" << address_base
        << ",\"receive_after_tick\":" << receive_delay << ",\"clock_period_ns\":" << period_ns
        << ",\"request_bytes\":32,\"burst_limit\":4,\"tick_limit\":100000"
        << ",\"runtime_ticks\":" << runtime.audit().ticks << ",\"time_ns\":" << runtime.context().time
        << ",\"sent\":" << source.sent() << ",\"source_retries\":" << source.retries()
        << ",\"echo_retries\":" << echo.retries() << ",\"runtime_completed\":" << runtime.audit().completed
        << ",\"native\":" << metrics_json(m) << ",\"responses\":[";
    for (std::size_t i = 0; i < source.receipts().size(); ++i) {
      const auto& t = source.receipts()[i];
      if (i) out << ',';
      out << "{\"id\":" << t.id << ",\"parent_id\":" << t.parent_id
          << ",\"address\":" << t.address << ",\"bytes\":" << t.bytes
          << ",\"transfer_bytes\":" << t.transfer_bytes << ",\"operation\":\"" << t.operation
          << "\",\"ordering\":\"" << t.ordering << "\",\"byte_mask\":[";
      for (std::size_t j = 0; j < t.byte_mask.size(); ++j) {
        if (j) out << ',';
        out << unsigned(t.byte_mask[j]);
      }
      out << "]}";
    }
    out << "]}";
    return out.str();
  }
};
}  // namespace example

int main() {
  try {
    constexpr unsigned requests = 32;
    auto a = std::make_unique<example::Session>(1, requests, 0x1000, 64, 1.0);
    auto b = std::make_unique<example::Session>(2, requests, 0x9000, 4096, 1.75);
    while (!a->runtime.drained()) { a->runtime.tick(); b->runtime.tick(); }
    const auto first = a->report(requests);
    example::require(!b->runtime.drained(), "second instance finished before destruction control");
    const auto before = example::metrics_json(b->network.metrics());
    const auto before_ticks = b->runtime.audit().ticks;
    a.reset();
    const auto after = example::metrics_json(b->network.metrics());
    example::require(before == after && before_ticks == b->runtime.audit().ticks,
                     "destroying A changed B");
    bool refused = false;
    try { auto invalid = example::native::make_icnt(example::Session::configuration(0)); }
    catch (const std::invalid_argument&) { refused = true; }
    example::require(refused && before == example::metrics_json(b->network.metrics()),
                     "invalid independent construction changed B");
    while (!b->runtime.drained()) b->runtime.tick();
    const auto second = b->report(requests);
    b.reset();
    auto control = std::make_unique<example::Session>(2, requests, 0x9000, 4096, 1.75);
    while (!control->runtime.drained()) control->runtime.tick();
    example::require(second == control->report(requests), "B differs from its isolated control");
    control.reset();
    std::cout << "ARCH_EXPLORER_STANDALONE_RESULT {\"schema\":\"arch-explorer.standalone-icnt-example.v1\","
                 "\"status\":\"passed\",\"scope\":\"two native LocalInterconnect instances; synthetic echo endpoints; no memory model\","
                 "\"A_destroyed_before_B_finished\":true,\"invalid_configuration_rejected\":true,\"B_matches_isolated_control\":true,"
                 "\"B_native_before_A_destruction\":" << before
              << ",\"B_native_after_A_destruction\":" << after
              << ",\"B_ticks_at_A_destruction\":" << before_ticks
              << ",\"instances\":[" << first << ',' << second << "]}\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "standalone ICNT example failed: " << error.what() << '\n';
    return 1;
  }
}
