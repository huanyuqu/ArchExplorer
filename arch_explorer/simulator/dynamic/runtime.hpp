// SPDX-License-Identifier: Apache-2.0
// Generic, single-threaded dynamic component coordinator. No device types.
#ifndef ARCH_EXPLORER_DYNAMIC_RUNTIME_HPP
#define ARCH_EXPLORER_DYNAMIC_RUNTIME_HPP

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <functional>
#include <limits>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace ae_dynamic {

struct Transaction {
  std::uint64_t id = 0, parent_id = 0;
  std::string source, source_port, destination, destination_port, operation;
  std::uint64_t address = 0, bytes = 0, transfer_bytes = 0;
  std::vector<std::uint8_t> byte_mask;
  std::string ordering;
};

struct Context {
  std::uint64_t tick = 0;
  double time = 0.0;
  int phase = -1;
  std::string clock, component;
};

enum class Direction { input, output };
struct Port {
  std::string name;
  Direction direction;
  std::string protocol = "transaction.v1";
};
enum class Stage { consumer_ready, physical_complete, resource_ready };

class Component {
 public:
  virtual ~Component() = default;
  // Both operations are passive: neither may advance another component.
  // can_accept is a pure credit/availability query. A rejected accept must not
  // mutate payload, queues, transaction ownership or resource lifecycle.
  virtual bool can_accept(const Transaction&, const Context&) const = 0;
  virtual bool accept(const Transaction&, const Context&) = 0;
  virtual bool drained() const = 0;
};

struct Audit {
  std::uint64_t ticks = 0, phase_calls = 0;
  std::uint64_t accepted = 0, rejected = 0, wakeups = 0;
  std::uint64_t consumer_ready = 0, physical_complete = 0, resource_ready = 0;
  std::uint64_t completed = 0;
};

class Runtime {
 public:
  explicit Runtime(std::uint64_t max_ticks = 1000000000,
                   std::uint64_t max_routes_per_tick = 10000000)
      : max_ticks_(max_ticks), max_routes_per_tick_(max_routes_per_tick) {
    if (!max_ticks || !max_routes_per_tick)
      fail("runtime bounds must be positive");
  }

  void add_component(const std::string& name, Component& component,
                     const std::vector<Port>& ports) {
    before_start();
    if (name.empty() || components_.count(name)) fail("duplicate/empty component");
    Node node{&component, {}};
    for (const auto& port : ports) {
      if (port.name.empty() || port.protocol.empty() || node.ports.count(port.name))
        fail("duplicate/empty port");
      node.ports.emplace(port.name, port);
    }
    components_.emplace(name, std::move(node));
  }

  void connect(const std::string& source, const std::string& source_port,
               const std::string& destination, const std::string& destination_port) {
    before_start();
    const auto& a = port(source, source_port, Direction::output);
    const auto& b = port(destination, destination_port, Direction::input);
    if (a.protocol != b.protocol) fail("connection protocol mismatch");
    if (!connections_.emplace(source, source_port, destination, destination_port).second)
      fail("duplicate connection");
  }

  void add_clock(const std::string& name, double period) {
    before_start();
    if (name.empty() || clocks_.count(name) || !std::isfinite(period) || period <= 0)
      fail("invalid/duplicate clock");
    clocks_.emplace(name, Clock{period, 0.0});
  }

  using Phase = std::function<void(const Context&)>;
  void add_phase(const std::string& clock, int order, const std::string& component,
                 Phase callback) {
    before_start();
    if (!clocks_.count(clock) || !components_.count(component) || !callback)
      fail("phase refers to unknown clock/component");
    phases_.push_back({clock, order, component, std::move(callback), phases_.size()});
  }

  // Retains the producer's original double periods and repeated-addition
  // semantics. All domains at the minimum timestamp run, in declared phase
  // order. There is no guessed idle skip or implicit receiver advancement.
  void tick() {
    if (in_tick_ || clocks_.empty()) fail("recursive tick or no clocks");
    if (audit_.ticks >= max_ticks_) fail("dynamic tick bound exceeded");
    started_ = true;
    if (!sorted_) {
      std::stable_sort(phases_.begin(), phases_.end(), [](const Hook& a, const Hook& b) {
        return std::tie(a.order, a.sequence) < std::tie(b.order, b.sequence);
      });
      sorted_ = true;
    }
    double next = std::numeric_limits<double>::infinity();
    for (const auto& entry : clocks_) next = std::min(next, entry.second.next);
    if (!std::isfinite(next) || next < context_.time) fail("invalid clock progress");
    std::set<std::string> active;
    for (auto& entry : clocks_) {
      if (entry.second.next <= next) {
        active.insert(entry.first);
        const double future = entry.second.next + entry.second.period;
        if (!std::isfinite(future) || future <= entry.second.next)
          fail("clock cannot advance");
        entry.second.next = future;
      }
    }
    context_ = Context{audit_.ticks, next, -1, "", ""};
    routes_this_tick_ = 0;
    in_tick_ = true;
    try {
      for (const auto& phase : phases_) {
        if (!active.count(phase.clock)) continue;
        context_.phase = phase.order;
        context_.clock = phase.clock;
        context_.component = phase.component;
        ++audit_.phase_calls;
        phase.callback(context_);
      }
    } catch (...) {
      in_tick_ = false;
      throw;
    }
    in_tick_ = false;
    ++audit_.ticks;
  }

  // No ownership/counter/waiter mutation. In this single-threaded coordinator,
  // can_route followed immediately by route cannot lose credit to another
  // clock phase. The caller must leave its packet queued until acceptance.
  bool can_route(const Transaction& transaction) const {
    validate_transaction(transaction);
    return components_.at(transaction.destination).component->can_accept(transaction, context_);
  }

  bool route(const Transaction& transaction) {
    validate_transaction(transaction);
    if (route_depth_) fail("passive accept cannot recursively route transactions");
    if (++routes_this_tick_ > max_routes_per_tick_ || route_depth_ >= 256)
      fail("dynamic route/recursion bound exceeded");
    auto& receiver = *components_.at(transaction.destination).component;
    bool accepted = false;
    if (receiver.can_accept(transaction, context_)) {
      ++route_depth_;
      try { accepted = receiver.accept(transaction, context_); }
      catch (...) { --route_depth_; throw; }
      --route_depth_;
    }
    if (!accepted) {
      ++audit_.rejected;
      waiters_[{transaction.destination, transaction.destination_port}].insert(transaction.source);
      return false;
    }
    auto found = transactions_.find(transaction.id);
    if (found == transactions_.end()) {
      transactions_.emplace(transaction.id, State{transaction.destination, false, false, false});
    } else {
      found->second.owner = transaction.destination;
    }
    ++audit_.accepted;
    return true;
  }

  void mark(std::uint64_t id, const std::string& owner, Stage stage) {
    auto found = transactions_.find(id);
    if (found == transactions_.end() || found->second.owner != owner)
      fail("lifecycle marker has no matching owner");
    auto& state = found->second;
    bool* flag = stage == Stage::consumer_ready ? &state.consumer :
                 stage == Stage::physical_complete ? &state.physical : &state.resource;
    if (*flag) fail("duplicate lifecycle marker");
    *flag = true;
    if (stage == Stage::consumer_ready) ++audit_.consumer_ready;
    else if (stage == Stage::physical_complete) ++audit_.physical_complete;
    else ++audit_.resource_ready;
    if (state.consumer && state.physical && state.resource) ++audit_.completed;
  }

  // Explicit availability notification, consumed by the sender on its own
  // phase. No callback can recursively advance an upstream clock domain.
  void available(const std::string& destination, const std::string& destination_port) {
    port(destination, destination_port, Direction::input);
    auto found = waiters_.find({destination, destination_port});
    if (found == waiters_.end()) return;
    for (const auto& source : found->second) {
      awake_.insert(source);
      ++audit_.wakeups;
    }
    waiters_.erase(found);
  }
  bool take_wakeup(const std::string& source) { return awake_.erase(source) != 0; }

  std::uint64_t pending() const {
    return static_cast<std::uint64_t>(transactions_.size()) - audit_.completed;
  }
  bool drained() const {
    if (pending()) return false;
    for (const auto& node : components_)
      if (!node.second.component->drained()) return false;
    return true;
  }
  const Audit& audit() const { return audit_; }
  const Context& context() const { return context_; }

 private:
  struct Node { Component* component; std::map<std::string, Port> ports; };
  struct Clock { double period, next; };
  struct Hook { std::string clock; int order; std::string component; Phase callback; std::size_t sequence; };
  struct State { std::string owner; bool consumer, physical, resource; };
  using Connection = std::tuple<std::string, std::string, std::string, std::string>;
  std::map<std::string, Node> components_;
  std::map<std::string, Clock> clocks_;
  std::vector<Hook> phases_;
  std::set<Connection> connections_;
  std::map<std::uint64_t, State> transactions_;
  std::map<std::pair<std::string, std::string>, std::set<std::string>> waiters_;
  std::set<std::string> awake_;
  Context context_;
  Audit audit_;
  std::uint64_t max_ticks_, max_routes_per_tick_, routes_this_tick_ = 0;
  unsigned route_depth_ = 0;
  bool started_ = false, sorted_ = false, in_tick_ = false;

  [[noreturn]] static void fail(const std::string& message) { throw std::runtime_error(message); }
  void before_start() const { if (started_) fail("topology/clock mutation after start"); }
  const Port& port(const std::string& node, const std::string& name, Direction direction) const {
    auto found = components_.find(node);
    if (found == components_.end()) fail("unknown component endpoint");
    auto entry = found->second.ports.find(name);
    if (entry == found->second.ports.end() || entry->second.direction != direction)
      fail("unknown port or wrong direction");
    return entry->second;
  }
  void validate_transaction(const Transaction& transaction) const {
    if (!transaction.id || transaction.parent_id == transaction.id || transaction.operation.empty())
      fail("invalid transaction identity/operation");
    if (!connections_.count({transaction.source, transaction.source_port,
                              transaction.destination, transaction.destination_port}))
      fail("transaction has no declared connection");
    auto old = transactions_.find(transaction.id);
    if (old != transactions_.end() && (old->second.owner != transaction.source ||
        (old->second.consumer && old->second.physical && old->second.resource)))
      fail("transaction has wrong owner or is already complete");
  }
};

}  // namespace ae_dynamic
#endif
