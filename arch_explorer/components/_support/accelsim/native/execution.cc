// SPDX-License-Identifier: Apache-2.0
#include "execution.hpp"
#include "abstract_hardware_model.h"

#include <algorithm>
#include <limits>
#include <stdexcept>

namespace ae_accelsim { namespace standalone {

ProgramExecution::ProgramExecution(ProgramOptions options,
                                 const CycleView& cycles,
                                 ShaderCounters& counters)
    : options_(options), cycles_(cycles), counters_(counters),
      slots_(options.max_concurrent_kernels) {
  if (!options.max_concurrent_kernels || !options.threads_per_shader)
    throw std::invalid_argument("program slots and shader thread capacity must be positive");
}

ProgramExecution::~ProgramExecution() = default;

void ProgramExecution::launch(std::shared_ptr<kernel_info_t> kernel) {
  if (!kernel) throw std::invalid_argument("cannot launch a null kernel");
  if (kernel->threads_per_cta() > options_.threads_per_shader)
    throw std::invalid_argument("kernel CTA exceeds shader thread capacity");
  const unsigned uid = kernel->get_uid();
  if (retired_.count(uid)) throw std::invalid_argument("kernel UID already retired");
  for (const auto& slot : slots_)
    if (slot && slot->get_uid() == uid)
      throw std::invalid_argument("duplicate kernel UID");
  for (auto& slot : slots_) {
    if (!slot || slot->done()) {
      slot = std::move(kernel);
      kernel_times_[slot->get_streamID()][uid] = {cycles_.absolute(), 0};
      return;
    }
  }
  throw std::runtime_error("all program execution slots are occupied");
}

bool ProgramExecution::can_start_kernel() const {
  for (const auto& slot : slots_) if (!slot || slot->done()) return true;
  return false;
}

bool ProgramExecution::hit_max_cta_count() const {
  return options_.max_ctas && issued_ctas_ >= options_.max_ctas;
}

bool ProgramExecution::kernel_more_cta_left(kernel_info_t* kernel) const {
  return !hit_max_cta_count() && kernel && !kernel->no_more_ctas_to_run();
}

bool ProgramExecution::get_more_cta_left() const {
  for (const auto& slot : slots_)
    if (kernel_more_cta_left(slot.get())) return true;
  return false;
}

void ProgramExecution::record_issued_ctas(unsigned count) {
  if (count > std::numeric_limits<std::uint64_t>::max() - issued_ctas_)
    throw std::overflow_error("issued CTA counter exhausted");
  issued_ctas_ += count;
}

void ProgramExecution::decrement_kernel_latency() {
  for (auto& slot : slots_)
    if (slot && slot->m_kernel_TB_latency) --slot->m_kernel_TB_latency;
}

void ProgramExecution::mark_selected(kernel_info_t& kernel) {
  if (std::find(executed_uids_.begin(), executed_uids_.end(), kernel.get_uid()) ==
      executed_uids_.end()) {
    kernel.start_cycle = cycles_.absolute();
    executed_uids_.push_back(kernel.get_uid());
    executed_names_.push_back(kernel.name());
  }
}

kernel_info_t* ProgramExecution::select_kernel() {
  // Preserve the upstream last-issued preference, followed by circular scan.
  auto* last = slots_[last_issued_].get();
  if (kernel_more_cta_left(last) && !last->m_kernel_TB_latency) {
    mark_selected(*last);
    return last;
  }
  for (unsigned n = 0; n < slots_.size(); ++n) {
    const unsigned index = (n + last_issued_ + 1) % slots_.size();
    auto* kernel = slots_[index].get();
    if (kernel_more_cta_left(kernel) && !kernel->m_kernel_TB_latency) {
      last_issued_ = index;
      mark_selected(*kernel);
      return kernel;
    }
  }
  return nullptr;
}

void ProgramExecution::set_kernel_done(kernel_info_t* kernel) {
  for (auto& slot : slots_) {
    if (slot.get() == kernel) {
      const auto uid = kernel->get_uid();
      last_stream_id_ = kernel->get_streamID();
      last_finished_uid_ = uid;
      kernel_times_[last_stream_id_][uid].second = cycles_.absolute();
      kernel->end_cycle = cycles_.absolute();
      retired_.emplace(uid, std::move(slot));
      finished_.push_back(uid);
      return;
    }
  }
  throw std::logic_error("completion does not belong to this program execution");
}

void ProgramExecution::inc_completed_cta() { ++counters_.completed_ctas; }

unsigned ProgramExecution::finished_kernel() {
  if (finished_.empty()) {
    last_stream_id_ = static_cast<std::uint64_t>(-1);
    return 0;
  }
  const unsigned uid = finished_.front();
  finished_.pop_front();
  return uid;
}

std::shared_ptr<kernel_info_t> ProgramExecution::release_finished(unsigned uid) {
  auto it = retired_.find(uid);
  if (it == retired_.end()) throw std::invalid_argument("kernel is not retired here");
  auto kernel = std::move(it->second);
  retired_.erase(it);
  return kernel;
}

bool ProgramExecution::active() const {
  for (const auto& slot : slots_) if (slot && !slot->done()) return true;
  return false;
}

std::unique_ptr<ProgramExecution> make_program_execution(
    ProgramOptions options, const CycleView& cycles, ShaderCounters& counters) {
  return std::unique_ptr<ProgramExecution>(new ProgramExecution(options, cycles, counters));
}

}}  // namespace ae_accelsim::standalone
