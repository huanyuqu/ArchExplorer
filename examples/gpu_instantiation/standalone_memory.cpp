// SPDX-License-Identifier: Apache-2.0
// External request workload using the public assembly of native L2/DRAM
// factories. No GPU, shader, kernel, warp or native transaction bridge.
#include "memory_system.hpp"
#include <algorithm>
#include <cstdint>
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
using ae_dynamic::Transaction;

void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}

std::string snapshot_json(const native::MemorySnapshot& value) {
  std::ostringstream out;
  out << "{\"cycles\":" << value.cycles << ",\"submitted\":" << value.submitted
      << ",\"received\":" << value.received << ",\"rejected\":" << value.rejected
      << ",\"physical\":" << value.physical << ",\"consumer\":" << value.consumer
      << ",\"resource\":" << value.resource << ",\"pending\":" << value.pending
      << ",\"drained\":" << (value.drained ? "true" : "false") << ",\"l2\":[";
  for (std::size_t i = 0; i < value.l2.size(); ++i) {
    const auto& m = value.l2[i];
    if (i) out << ',';
    out << "{\"accesses\":" << m.accesses << ",\"misses\":" << m.misses
        << ",\"hits\":" << m.hits << ",\"pending_hits\":" << m.pending_hits
        << ",\"reservation_fails\":" << m.reservation_fails
        << ",\"port_available_cycles\":" << m.port_available_cycles
        << ",\"data_port_busy_cycles\":" << m.data_port_busy_cycles
        << ",\"fill_port_busy_cycles\":" << m.fill_port_busy_cycles << ",\"tracked\":" << m.tracked
        << ",\"ingress\":" << m.ingress << ",\"dram_requests\":" << m.dram_requests
        << ",\"dram_replies\":" << m.dram_replies << ",\"responses\":" << m.responses
        << ",\"rop\":" << m.rop << '}';
  }
  out << "],\"dram\":[";
  for (std::size_t i = 0; i < value.dram.size(); ++i) {
    const auto& m = value.dram[i];
    if (i) out << ',';
    out << "{\"commands\":" << m.commands << ",\"activity\":" << m.activity
        << ",\"nop\":" << m.nop << ",\"act\":" << m.act << ",\"pre\":" << m.pre
        << ",\"reads\":" << m.reads << ",\"writes\":" << m.writes
        << ",\"writebacks\":" << m.writebacks << ",\"requests\":" << m.requests
        << ",\"delay\":" << m.delay << ",\"controller\":" << m.controller
        << ",\"read_write\":" << m.read_write << ",\"replies\":" << m.replies
        << ",\"scheduled_reads\":" << m.scheduled_reads << ",\"scheduled_writes\":" << m.scheduled_writes
        << ",\"bank_requests\":" << m.bank_requests << '}';
  }
  out << "]}";
  return out.str();
}

struct Request {
  std::uint64_t address;
  unsigned bytes;
  bool write;
  std::string phase;
};

struct Receipt { std::uint64_t id; Request request; Transaction response; };

// The workload owns only public request/receipt metadata. Queues, admission,
// tags, MSHRs, bank timing, writebacks and completion belong to the native API.
class Workload {
 public:
  native::MemoryOptions options;
  native::MemorySystem memory;
  explicit Workload(const native::MemoryOptions& config, unsigned cold_lines)
      : options(config), memory(options), cold_lines_(cold_lines) { submit_phase(); }

  void advance() {
    require(++ticks_ <= 1000000, "standalone memory workload exceeded tick bound");
    memory.tick();
    Transaction response;
    while (memory.receive(response)) {
      const auto found = outstanding_.find(response.id);
      if (found == outstanding_.end()) {
        std::cerr << "unmatched response: id=" << response.id << " parent=" << response.parent_id
                  << " address=" << response.address << " bytes=" << response.bytes
                  << " operation=" << response.operation << " phase=" << phase_ << " expected=";
        for (const auto& entry : outstanding_) std::cerr << entry.first << ',';
        std::cerr << " native=" << snapshot_json(memory.snapshot()) << '\n';
      }
      require(found != outstanding_.end(), "unknown or repeated memory response");
      const auto& request = found->second;
      require(response.parent_id == 0 && response.address == request.address && response.bytes == request.bytes,
              "memory response identity/address/length changed");
      require(response.operation == (request.write ? "write_ack" : "read_reply"),
              "memory response operation changed");
      require(response.byte_mask.size() == 128, "memory response lacks the native 128B line mask");
      for (unsigned i = 0; i < response.byte_mask.size(); ++i) {
        const unsigned offset = request.address % 128;
        require(response.byte_mask[i] == (i >= offset && i < offset + request.bytes ? 1 : 0),
                "memory response byte mask changed");
      }
      // Public sector protocol: a read reply carries 32B plus its 8B control;
      // a write acknowledgement carries 8B. Aggregation sums real responses.
      const auto expected_transfer = (request.bytes / 32) * (request.write ? 8U : 40U);
      require(response.ordering == "native-fifo" && response.transfer_bytes == expected_transfer,
              "memory response ordering or aggregated sector transfer length changed");
      receipts_.push_back({response.id, request, response});
      outstanding_.erase(found);
    }
    const auto state = memory.snapshot();
    peak_pending_ = std::max(peak_pending_, state.pending);
    if (outstanding_.empty() && !memory.drained()) observed_tail_ = true;
    if (outstanding_.empty() && memory.drained() && phase_ < 4) {
      ++phase_;
      if (phase_ < 4) submit_phase();
    }
  }

  bool done() const { return phase_ == 4 && outstanding_.empty() && memory.drained(); }

  std::string report() const {
    require(done(), "memory source or physical native queues did not drain");
    const auto state = memory.snapshot();
    require(state.pending == 0 && state.submitted == receipts_.size() && state.received == receipts_.size(),
            "memory completion/request accounting differs");
    require(state.physical >= receipts_.size() && state.consumer >= receipts_.size() &&
            state.resource >= receipts_.size(), "memory lifecycle phases are incomplete");
    std::uint64_t hits = 0, misses = 0, reads = 0, writes = 0, writebacks = 0;
    for (const auto& m : state.l2) {
      hits += m.hits; misses += m.misses;
      require(!m.tracked && !m.ingress && !m.dram_requests && !m.dram_replies && !m.responses && !m.rop,
              "L2 reports physical pending work after drain");
    }
    for (const auto& m : state.dram) {
      reads += m.reads; writes += m.writes; writebacks += m.writebacks;
      require(!m.delay && !m.controller && !m.read_write && !m.replies && !m.bank_requests,
              "DRAM reports physical pending work after drain");
    }
    // The native metrics distinguish ordinary writes from dirty writebacks.
    // This writeback cache can perform all DRAM writes through the latter.
    require(hits && misses && reads && (writes || writebacks) && writebacks && state.rejected,
            "native hits/misses/read/write/dirty eviction/backpressure coverage missing");
    std::ostringstream out;
    out << std::setprecision(17) << "{\"channels\":" << options.channels
        << ",\"subpartitions_per_channel\":" << options.subpartitions_per_channel
        << ",\"cache_sets\":" << options.cache_sets << ",\"cache_ways\":" << options.cache_ways
        << ",\"controller_queue\":" << options.controller_queue
        << ",\"simple_dram\":" << (options.simple_dram ? "true" : "false")
        << ",\"dram_latency\":" << options.dram_latency << ",\"rop_latency\":" << options.rop_latency
        << ",\"cache_period\":" << options.cache_period << ",\"dram_period\":" << options.dram_period
        << ",\"cold_lines\":" << cold_lines_ << ",\"caller_ticks\":" << ticks_
        << ",\"peak_pending\":" << peak_pending_ << ",\"observed_physical_tail\":" << (observed_tail_ ? "true" : "false")
        << ",\"native\":" << snapshot_json(state) << ",\"responses\":[";
    for (std::size_t i = 0; i < receipts_.size(); ++i) {
      const auto& r = receipts_[i]; const auto& t = r.response;
      if (i) out << ',';
      out << "{\"id\":" << r.id << ",\"phase\":\"" << r.request.phase << "\",\"write\":"
          << (r.request.write ? "true" : "false") << ",\"parent_id\":" << t.parent_id
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

 private:
  void submit(std::uint64_t address, unsigned bytes, bool write, const char* label) {
    const auto id = memory.submit(address, bytes, write);
    require(id && outstanding_.emplace(id, Request{address, bytes, write, label}).second,
            "native source returned an invalid/repeated request ID");
  }
  void submit_phase() {
    if (phase_ == 0) submit(0x1000, 32, false, "warm");
    else if (phase_ == 1) {
      for (unsigned i = 0; i < 8; ++i) submit(0x1000, 32, false, "hit");
    } else if (phase_ == 2) {
      for (unsigned i = 0; i < 16; ++i) submit(0x1000 + (i % 4) * 128, 32, true, "dirty");
    } else if (phase_ == 3) {
      // More distinct lines than either configured cache can hold. Sector
      // offsets/lengths vary within a native 128B line; no timing is modelled.
      for (unsigned i = 0; i < cold_lines_; ++i)
        submit(0x8000 + i * 128 + (i % 4) * 32, i % 4 == 0 ? 64 : 32, false, "conflict");
    }
  }
  const unsigned cold_lines_;
  unsigned phase_ = 0;
  std::uint64_t ticks_ = 0, peak_pending_ = 0;
  bool observed_tail_ = false;
  std::map<std::uint64_t, Request> outstanding_;
  std::vector<Receipt> receipts_;
};

native::MemoryOptions configuration(bool second) {
  native::MemoryOptions c;
  c.channels = second ? 1 : 2;
  c.subpartitions_per_channel = 1;
  c.cache_sets = second ? 4 : 2;
  c.cache_ways = 1;
  c.controller_queue = second ? 8 : 4;
  c.simple_dram = false;
  c.cache_period = second ? 1.25 : 1.0;
  c.dram_period = second ? 4.5 : 2.0;
  return c;
}
}  // namespace example

int main() {
  try {
    auto a = std::make_unique<example::Workload>(example::configuration(false), 128);
    auto b = std::make_unique<example::Workload>(example::configuration(true), 512);
    while (!a->done()) { a->advance(); b->advance(); }
    const auto first = a->report();
    example::require(!b->done(), "second memory system finished before isolation control");
    const auto before = example::snapshot_json(b->memory.snapshot());
    a.reset();
    example::require(before == example::snapshot_json(b->memory.snapshot()), "destroying A changed B");
    bool refused = false;
    try {
      auto invalid = example::configuration(false); invalid.channels = 0;
      example::native::MemorySystem bad(invalid);
    } catch (const std::invalid_argument&) { refused = true; }
    example::require(refused && before == example::snapshot_json(b->memory.snapshot()),
                     "invalid independent memory construction changed B");
    std::string aborted;
    {
      example::native::MemorySystem unfinished(example::configuration(false));
      for (unsigned i = 0; i < 64; ++i) unfinished.submit(0x1000 + i * 128, 32, true);
      unfinished.tick();
      example::require(!unfinished.drained(), "abort control had no in-flight memory work");
      aborted = example::snapshot_json(unfinished.snapshot());
    }
    const auto after = example::snapshot_json(b->memory.snapshot());
    example::require(before == after, "destroying another active memory system changed B");
    while (!b->done()) b->advance();
    const auto second = b->report();
    b.reset();
    auto control = std::make_unique<example::Workload>(example::configuration(true), 512);
    while (!control->done()) control->advance();
    example::require(second == control->report(), "interleaved/destruction run differs from isolated B control");
    control.reset();
    std::cout << "ARCH_EXPLORER_STANDALONE_RESULT {\"schema\":\"arch-explorer.standalone-memory-example.v1\","
                 "\"status\":\"passed\",\"scope\":\"external read/write workload using native L2/DRAM factory assembly; no GPU program\","
                 "\"A_destroyed_before_B_finished\":true,\"invalid_configuration_rejected\":true,"
                 "\"B_matches_isolated_control\":true,\"B_before_other_destruction\":" << before
              << ",\"B_after_other_destruction\":" << after
              << ",\"aborted_memory_before_destruction\":" << aborted
              << ",\"instances\":[" << first << ',' << second << "]}\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "standalone memory example failed: " << error.what() << '\n';
    return 1;
  }
}
