// SPDX-License-Identifier: Apache-2.0
#include "transactions.hpp"
#include "../gpu-sim.h"
#include "../mem_fetch.h"
#include "../../../libcuda/gpgpu_context.h"
#include <algorithm>
#include <stdexcept>

namespace ae_accelsim { namespace standalone {
using ae_dynamic::Stage;
using ae_dynamic::Transaction;

void NativeRequests::bind(ae_dynamic::Runtime& runtime) {
  if (runtime_ || closing_) throw std::logic_error("transaction domain already bound/closed");
  runtime_ = &runtime;
}

void NativeRequests::check() const {
  if (fault_) throw std::logic_error(fault_);
  if (closing_) throw std::logic_error("transaction domain is closing");
}

mem_fetch* NativeRequests::resolve(std::uint64_t id) {
  check();
  auto found = payload_.find(id);
  if (found == payload_.end() || deleted_.count(id))
    throw std::invalid_argument("transaction does not belong to this live native domain");
  return found->second;
}

Transaction NativeRequests::describe(mem_fetch* request) {
  check();
  if (!request) throw std::invalid_argument("null native transaction");
  const auto id = request->get_request_uid();
  auto found = payload_.find(id);
  if (found == payload_.end() || found->second != request)
    throw std::invalid_argument("native transaction allocation belongs to another domain");
  const auto old = life_.find(id);
  if (old != life_.end() && (old->second.resource || old->second.cancelled))
    throw std::logic_error("native transaction used after resource release");
  Transaction tx;
  tx.id = id;
  if (request->get_original_mf()) tx.parent_id = request->get_original_mf()->get_request_uid();
  tx.operation = request->is_reply() ? (request->get_is_write() ? "write_ack" : "read_reply") :
                                      (request->get_is_write() ? "write" : "read");
  if (request->isatomic()) tx.operation = "atomic";
  tx.address = request->get_addr();
  tx.bytes = request->get_data_size();
  tx.transfer_bytes = request->is_reply() ?
      (request->get_is_write() ? request->get_ctrl_size() : request->size()) :
      (request->get_is_write() ? request->size() : request->get_ctrl_size());
  const auto mask = request->get_access_byte_mask();
  for (unsigned i=0; i<mask.size(); ++i) tx.byte_mask.push_back(mask[i] ? 1 : 0);
  tx.ordering = "native-fifo";
  return tx;
}

void NativeRequests::allocated(unsigned id, mem_fetch* request) {
  check();
  if (!request || !id || payload_.count(id) || life_.count(id))
    throw std::invalid_argument("duplicate/invalid native transaction allocation");
  payload_.emplace(id,request);
  ++allocated_count_;
}

void NativeRequests::validate_source_request(std::uint64_t address, unsigned bytes) {
  if (!bytes || bytes > 128 || address % 32 || bytes % 32 || address % 128 + bytes > 128)
    throw std::invalid_argument("memory request source requires aligned sectors within one 128B line");
}

Transaction NativeRequests::create(std::uint64_t address, unsigned bytes, bool write,
    const memory_config& config, gpgpu_context& context, const CycleView& clock) {
  check();
  validate_source_request(address, bytes);
  if (config.ae_transaction_observer != this || !config.ae_request_ids || config.gpgpu_ctx != &context)
    throw std::invalid_argument("memory source configuration belongs to another transaction domain");
  active_mask_t lanes;
  lanes.set(0);
  mem_access_byte_mask_t byte_mask;
  mem_access_sector_mask_t sectors;
  for (unsigned i=0; i<bytes; ++i) byte_mask.set(address % 128 + i);
  for (unsigned i=0; i<bytes/32; ++i) sectors.set((address % 128)/32+i);
  mem_access_t access(write ? GLOBAL_ACC_W : GLOBAL_ACC_R,address,bytes,write,
                      lanes,byte_mask,sectors,&context);
  auto* request = new mem_fetch(access,nullptr,0,8,0,0,0,&config,clock.absolute());
  try {
    return describe(request);
  } catch (...) {
    delete request;
    throw;
  }
}

bool NativeRequests::route(const Transaction& transaction, bool complete) {
  check();
  if (!runtime_) throw std::logic_error("native transaction domain has no Runtime");
  resolve(transaction.id);
  ++route_depth_;
  bool accepted;
  try {
    accepted = runtime_->route(transaction);
  } catch (...) {
    --route_depth_;
    throw;
  }
  --route_depth_;
  if (accepted) {
    auto& state = life_[transaction.id];
    if (state.resource || state.cancelled)
      throw std::logic_error("native route after transaction completion");
    state.owner = transaction.destination;
    if (complete) physical(transaction.id);
  }
  complete_destroyed();
  return accepted;
}

void NativeRequests::physical(std::uint64_t id) {
  auto& state = life_.at(id);
  if (!state.physical) {
    runtime_->mark(id,state.owner,Stage::physical_complete);
    state.physical = true;
    ++physical_count_;
  }
}

void NativeRequests::destroyed(unsigned id) noexcept {
  ++destroyed_count_;
  try {
    if (closing_) {
      payload_.erase(id);
      return;
    }
    // Internal native requests that never crossed a wire have no Runtime
    // lifecycle. Their owning cache/controller is responsible for draining.
    if (life_.count(id) || payload_.count(id)) deleted_.insert(id);
  } catch (...) {
    fault_ = "cannot retain native destruction notification";
  }
}

void NativeRequests::complete_destroyed() {
  check();
  if (route_depth_) return;
  while (!deleted_.empty()) {
    const auto id = *deleted_.begin();
    deleted_.erase(deleted_.begin());
    auto found = life_.find(id);
    if (found != life_.end()) {
      // SIM38's lifecycle observer counted only requests that crossed a
      // wire. Track all allocations separately without changing that field.
      ++native_deleted_;
      auto& state = found->second;
      if (state.resource) throw std::logic_error("native transaction destroyed twice");
      physical(id);
      if (!state.consumer) {
        runtime_->mark(id,state.owner,Stage::consumer_ready);
        state.consumer = true;
        ++consumer_count_;
      }
      runtime_->mark(id,state.owner,Stage::resource_ready);
      state.resource = true;
      ++resource_count_;
    }
    payload_.erase(id);
  }
}

std::uint64_t NativeRequests::pending() const {
  std::uint64_t count = 0;
  for (const auto& entry : life_) count += !entry.second.resource && !entry.second.cancelled;
  return count;
}

void NativeRequests::cancel() noexcept {
  closing_ = true;
  runtime_ = nullptr;
  for (auto& entry : life_)
    if (!entry.second.resource) entry.second.cancelled = true;
}

NativeRequests::~NativeRequests() {
  // Components must already be gone. Cancellation releases resources without
  // inventing successful Runtime stage transitions or a drained result.
  cancel();
  while (!payload_.empty()) {
    auto found = payload_.begin();
    auto* request = found->second;
    payload_.erase(found);
    delete request;
  }
}

}}  // namespace ae_accelsim::standalone
