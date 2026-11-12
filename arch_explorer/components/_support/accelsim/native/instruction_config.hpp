// SPDX-License-Identifier: Apache-2.0
#ifndef ARCH_EXPLORER_ACCELSIM_INSTRUCTION_CONFIG_HPP
#define ARCH_EXPLORER_ACCELSIM_INSTRUCTION_CONFIG_HPP

#include "option_parser.h"
#include "shader_services.hpp"
#include <stdexcept>

namespace ae_accelsim { namespace standalone {

// Only native instruction configuration strings, not a functional simulator.
// reg_options is generated verbatim from the pinned original registration.
// The owner retains these parsed strings until all clusters are destroyed.
struct InstructionConfig {
  char* opcode_latency_int = nullptr;
  char* opcode_latency_fp = nullptr;
  char* opcode_latency_dp = nullptr;
  char* opcode_latency_sfu = nullptr;
  char* opcode_latency_tensor = nullptr;
  char* opcode_initiation_int = nullptr;
  char* opcode_initiation_fp = nullptr;
  char* opcode_initiation_dp = nullptr;
  char* opcode_initiation_sfu = nullptr;
  char* opcode_initiation_tensor = nullptr;
  char* cdp_latency_str = nullptr;
  void reg_options(option_parser_t);
  InstructionTiming shader_timing() const {
    if (!opcode_latency_int || !opcode_latency_fp || !opcode_latency_dp ||
        !opcode_latency_sfu || !opcode_latency_tensor)
      throw std::invalid_argument("native instruction timing configuration is incomplete");
    return {opcode_latency_int,opcode_latency_fp,opcode_latency_dp,
            opcode_latency_sfu,opcode_latency_tensor};
  }
};

}}  // namespace ae_accelsim::standalone
#endif
