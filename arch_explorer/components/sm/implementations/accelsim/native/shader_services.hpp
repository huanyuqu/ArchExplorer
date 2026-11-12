// SPDX-License-Identifier: Apache-2.0
#ifndef ARCH_EXPLORER_ACCELSIM_SHADER_SERVICES_HPP
#define ARCH_EXPLORER_ACCELSIM_SHADER_SERVICES_HPP

#include <cstdint>
#include "services.hpp"

class kernel_info_t;
class mem_fetch;

namespace ae_accelsim { namespace standalone {

// Implemented by the program execution domain, never by the Runtime. SMs ask
// only about dispatch and completion; they cannot inspect another component.
class ProgramDispatch {
 public:
  virtual ~ProgramDispatch() = default;
  virtual kernel_info_t* select_kernel() = 0;
  virtual bool kernel_more_cta_left(kernel_info_t* kernel) const = 0;
  virtual void set_kernel_done(kernel_info_t* kernel) = 0;
  virtual void inc_completed_cta() = 0;
};

// This is an instance-owned counter record. It has no execution methods,
// component lookup, queues, configuration, or process-global identity.
struct ShaderCounters {
  std::uint64_t instructions = 0;
  std::uint64_t completed_instructions = 0;
  std::uint64_t completed_ctas = 0;
  std::uint64_t core_to_icnt_stalls = 0;
  std::uint64_t icnt_to_core_stalls = 0;
  std::uint64_t last_instruction_cycle = 0;
  unsigned last_instruction_shader = 0;
};

struct ShaderOptions {
  std::uint64_t core_frequency_hz = 0;
  bool flush_l1 = false;
};

// The upstream opcode-latency option strings determine pipeline depths.
// The configuration owner supplies them before shader_core_config::init().
// They are read synchronously during init and are not a functional executor.
struct InstructionTiming {
  const char* integer = nullptr;
  const char* floating = nullptr;
  const char* dual = nullptr;
  const char* special = nullptr;
  const char* tensor = nullptr;
};

// Only the native component domain sees mem_fetch. An assembly translates
// these calls into typed Runtime transactions. can_send has no side effects;
// send must accept the previously credited request synchronously or throw
// without taking ownership. pop_reply transfers one already accepted reply.
class ShaderTransport {
 public:
  virtual ~ShaderTransport() = default;
  virtual bool can_send(unsigned cluster, unsigned bytes) const = 0;
  virtual void send(unsigned cluster, unsigned memory_device,
                    mem_fetch* request, unsigned bytes) = 0;
  virtual mem_fetch* pop_reply(unsigned cluster) = 0;
};

// Cache mechanics and allocation services are owned by the memory component
// domain. Forward declaration prevents a shader service locator dependency.
struct CacheServices;

}}  // namespace ae_accelsim::standalone
#endif
