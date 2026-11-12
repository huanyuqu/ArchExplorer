// SPDX-License-Identifier: Apache-2.0
#ifndef ARCH_EXPLORER_ACCELSIM_CONFIGURATION_API_HPP
#define ARCH_EXPLORER_ACCELSIM_CONFIGURATION_API_HPP
#include "instruction_config.hpp"
#include "gpgpu-sim/gpu-sim.h"
#include "gpgpu_context.h"
#include <array>
#include <cmath>
#include <stdexcept>
namespace ae_accelsim { namespace standalone {
// Configuration access only. No simulation state or component instances.
struct ConfigurationAccess {
  static shader_core_config& shader(gpgpu_sim_config& c){return c.m_shader_config;}
  static memory_config& memory(gpgpu_sim_config& c){return c.m_memory_config;}
  static std::array<double,4> periods(const gpgpu_sim_config& c){return {c.core_period,c.icnt_period,c.l2_period,c.dram_period};}
  static std::uint64_t max_cycles(const gpgpu_sim_config& c){return c.gpu_max_cycle_opt;}
  static std::uint64_t max_instructions(const gpgpu_sim_config& c){return c.gpu_max_insn_opt;}
  static unsigned max_ctas(const gpgpu_sim_config& c){return c.gpu_max_cta_opt;}
  static bool flush_l2(const gpgpu_sim_config& c){return c.gpgpu_flush_l2_cache;}
  static void initialize(gpgpu_sim_config& c,const InstructionConfig& timing) {
    if(c.ae_sim_mode || c.ae_cdp_enabled || c.ae_ptx_instruction_classification ||
       c.ae_kernel_launch_latency<0 || c.ae_tb_launch_latency<0 ||
       c.g_power_simulation_enabled || c.g_visualizer_enabled || Trace::enabled ||
       c.gpgpu_cflog_interval || c.gpgpu_flush_l2_cache)
      throw std::invalid_argument("standalone trace assembly does not provide functional/CDP/power/visualizer/logging/flush-L2 services");
    c.gpgpu_ctx->ae_kernel_launch_latency=c.ae_kernel_launch_latency;
    c.gpgpu_ctx->ae_tb_launch_latency=c.ae_tb_launch_latency;
    c.m_shader_config.set_instruction_timing(timing.shader_timing());
    c.m_shader_config.init();
    c.m_memory_config.init(&c.m_shader_config);
    c.init_clock_domains();
    for(double value:periods(c))if(!std::isfinite(value) || value<=0)
      throw std::invalid_argument("invalid native clock frequency");
    c.m_valid=true;
  }
};
}}
#endif
