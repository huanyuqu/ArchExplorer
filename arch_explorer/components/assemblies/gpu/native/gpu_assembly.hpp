// SPDX-License-Identifier: Apache-2.0
#ifndef ARCH_EXPLORER_ACCELSIM_GPU_ASSEMBLY_HPP
#define ARCH_EXPLORER_ACCELSIM_GPU_ASSEMBLY_HPP
#include <memory>
#include <string>
#include <vector>
namespace ae_accelsim { namespace standalone {
struct GpuOptions {
  std::vector<std::string> configurations;
  std::string topology;
};
// A wiring/phase assembly, owning factory products. It is not a native GPU
// superclass, a functional simulator, or a service supplied to a component.
class GpuAssembly {
 public:
  explicit GpuAssembly(const GpuOptions&);
  ~GpuAssembly();
  GpuAssembly(const GpuAssembly&)=delete;
  GpuAssembly& operator=(const GpuAssembly&)=delete;
  void run_trace(const std::string& kernels_list,const std::string& metrics_path);
 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};
}}
#endif
