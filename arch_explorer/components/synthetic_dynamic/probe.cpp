// SPDX-License-Identifier: Apache-2.0
// Independent bounded-queue family: no native/device headers or libraries.
#include "runtime.hpp"
#include <deque>
#include <iostream>

using namespace ae_dynamic;

struct Source : Component {
  unsigned issued = 0;
  bool can_accept(const Transaction&, const Context&) const override { return false; }
  bool accept(const Transaction&, const Context&) override { return false; }
  bool drained() const override { return issued == 7; }
};

struct Buffer : Component {
  std::deque<Transaction> queue;
  bool can_accept(const Transaction&, const Context&) const override { return queue.size() < 1; }
  bool accept(const Transaction& tx, const Context& ctx) override {
    if (!can_accept(tx, ctx)) return false;
    queue.push_back(tx);
    return true;
  }
  bool drained() const override { return queue.empty(); }
};

struct Sink : Component {
  struct Entry { std::uint64_t id, due; unsigned stage; };
  std::deque<Entry> queue;
  bool can_accept(const Transaction&, const Context&) const override { return queue.size() < 1; }
  bool accept(const Transaction& tx, const Context& ctx) override {
    if (!can_accept(tx, ctx)) return false;
    queue.push_back({tx.id, ctx.tick+3, 0});
    return true;
  }
  bool drained() const override { return queue.empty(); }
};

int main() {
  try {
    Runtime runtime(1000);
    Source source;
    Buffer buffer;
    Sink sink;
    runtime.add_component("producer", source, {{"out", Direction::output}});
    runtime.add_component("buffer", buffer, {{"in", Direction::input}, {"out", Direction::output}});
    runtime.add_component("consumer", sink, {{"in", Direction::input}});
    runtime.connect("producer", "out", "buffer", "in");
    runtime.connect("buffer", "out", "consumer", "in");
    runtime.add_clock("producer", 1.0/1300.0);
    runtime.add_clock("queue", 1.0/700.0);
    runtime.add_clock("consumer", 1.0/1100.0);
    std::uint64_t separated = 0;
    runtime.add_phase("consumer", 0, "consumer", [&](const Context& ctx) {
      if (sink.queue.empty() || sink.queue.front().due > ctx.tick) return;
      auto& entry = sink.queue.front();
      const Stage stages[] = {Stage::consumer_ready, Stage::physical_complete, Stage::resource_ready};
      runtime.mark(entry.id, "consumer", stages[entry.stage]);
      ++entry.stage;
      entry.due = ctx.tick+2;
      if (entry.stage == 1 && runtime.pending() > 0 && !runtime.drained()) ++separated;
      if (entry.stage == 3) {
        sink.queue.pop_front();
        runtime.available("consumer", "in");
      }
    });
    runtime.add_phase("queue", 1, "buffer", [&](const Context&) {
      if (buffer.queue.empty()) return;
      Transaction tx = buffer.queue.front();
      tx.source = "buffer"; tx.source_port = "out";
      tx.destination = "consumer"; tx.destination_port = "in";
      if (runtime.route(tx)) {
        buffer.queue.pop_front();
        runtime.available("buffer", "in");
      }
    });
    runtime.add_phase("producer", 2, "producer", [&](const Context&) {
      if (source.issued == 7) return;
      Transaction tx;
      tx.id = source.issued+1; tx.parent_id = 100;
      tx.source = "producer"; tx.source_port = "out";
      tx.destination = "buffer"; tx.destination_port = "in";
      tx.operation = "transform"; tx.bytes = 8; tx.transfer_bytes = 16;
      tx.byte_mask = {255}; tx.ordering = "stream0";
      if (runtime.route(tx)) ++source.issued;
      runtime.take_wakeup("producer");
    });
    while (!runtime.drained()) runtime.tick();
    const auto& audit = runtime.audit();
    if (audit.accepted != 14 || audit.completed != 7 || !audit.rejected ||
        !audit.wakeups || separated != 7 || runtime.pending())
      throw std::runtime_error("queue/lifecycle proof failed");
    bool duplicate_rejected = false, port_rejected = false, bound_rejected = false;
    try { runtime.mark(1, "consumer", Stage::resource_ready); }
    catch (const std::runtime_error&) { duplicate_rejected = true; }
    Runtime invalid;
    invalid.add_component("p", source, {{"out", Direction::output}});
    invalid.add_component("q", sink, {{"in", Direction::input}});
    try { invalid.connect("p", "out", "q", "missing"); }
    catch (const std::runtime_error&) { port_rejected = true; }
    Runtime bounded(1);
    bounded.add_clock("clock", 1.0);
    bounded.tick();
    try { bounded.tick(); } catch (const std::runtime_error&) { bound_rejected = true; }
    if (!duplicate_rejected || !port_rejected || !bound_rejected)
      throw std::runtime_error("invalid dynamic execution accepted");
    std::cout << "{\"schema\":\"arch-explorer.dynamic-probe.v1\",\"status\":\"passed\","
              << "\"accepted\":" << audit.accepted << ",\"rejected\":" << audit.rejected
              << ",\"wakeups\":" << audit.wakeups << ",\"completed\":" << audit.completed
              << ",\"pending\":" << runtime.pending() << ",\"asynchronous_ticks\":" << audit.ticks
              << ",\"separate_lifecycle\":" << separated
              << ",\"bad_port_rejected\":true,\"duplicate_completion_rejected\":true,"
              << "\"bound_rejected\":true,\"system_drained\":true}\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
