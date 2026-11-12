// SPDX-License-Identifier: Apache-2.0
#ifndef ARCH_EXPLORER_ACCELSIM_PROGRAM_EXECUTION_HPP
#define ARCH_EXPLORER_ACCELSIM_PROGRAM_EXECUTION_HPP

#include <deque>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include "shader_services.hpp"

namespace ae_accelsim { namespace standalone {

struct ProgramOptions {
  unsigned max_concurrent_kernels = 1;
  unsigned threads_per_shader = 0;
  std::uint64_t max_ctas = 0;  // zero preserves the upstream unlimited mode
};

// Kernel/CTA policy extracted from upstream select_kernel/launch/retirement.
// This service owns only program state, not clusters, ICNT, caches or memory.
class ProgramExecution final : public ProgramDispatch {
 public:
  ProgramExecution(ProgramOptions options, const CycleView& cycles,
                   ShaderCounters& counters);
  ~ProgramExecution();
  ProgramExecution(const ProgramExecution&) = delete;
  ProgramExecution& operator=(const ProgramExecution&) = delete;

  void launch(std::shared_ptr<kernel_info_t> kernel);
  bool can_start_kernel() const;
  bool get_more_cta_left() const;
  bool hit_max_cta_count() const;
  void record_issued_ctas(unsigned count);
  std::uint64_t issued_ctas() const { return issued_ctas_; }
  void decrement_kernel_latency();
  unsigned finished_kernel();
  std::shared_ptr<kernel_info_t> release_finished(unsigned uid);
  bool active() const;
  kernel_info_t* select_kernel() override;
  bool kernel_more_cta_left(kernel_info_t* kernel) const override;
  void set_kernel_done(kernel_info_t* kernel) override;
  void inc_completed_cta() override;
  const std::vector<unsigned>& executed_uids() const { return executed_uids_; }
  const std::vector<std::string>& executed_names() const { return executed_names_; }
  std::uint64_t last_stream_id() const { return last_stream_id_; }
  unsigned last_finished_uid() const { return last_finished_uid_; }
  using Timeline = std::map<std::uint64_t, std::map<unsigned,
      std::pair<std::uint64_t, std::uint64_t>>>;
  const Timeline& kernel_times() const { return kernel_times_; }

 private:
  void mark_selected(kernel_info_t& kernel);
  ProgramOptions options_;
  const CycleView& cycles_;
  ShaderCounters& counters_;
  std::vector<std::shared_ptr<kernel_info_t>> slots_;
  std::map<unsigned, std::shared_ptr<kernel_info_t>> retired_;
  std::deque<unsigned> finished_;
  std::vector<unsigned> executed_uids_;
  std::vector<std::string> executed_names_;
  Timeline kernel_times_;
  unsigned last_issued_ = 0;
  std::uint64_t last_stream_id_ = static_cast<std::uint64_t>(-1);
  unsigned last_finished_uid_ = 0;
  std::uint64_t issued_ctas_ = 0;
};

std::unique_ptr<ProgramExecution> make_program_execution(
    ProgramOptions options, const CycleView& cycles, ShaderCounters& counters);

}}  // namespace ae_accelsim::standalone
#endif
