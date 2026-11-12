// SPDX-License-Identifier: Apache-2.0
#include "gpu_assembly.hpp"
#include <iostream>
#include <stdexcept>
int main(int argc,char** argv) {
  try {
    ae_accelsim::standalone::GpuOptions options;std::string metrics,trace;
    for(int i=1;i<argc;++i){std::string key=argv[i];if(++i>=argc)throw std::invalid_argument("missing native argument value");
      std::string value=argv[i];
      if(key=="-ae_mode"){if(value!="components")throw std::invalid_argument("standalone image has no original GPU engine path");}
      else if(key=="-ae_topology")options.topology=value;else if(key=="-ae_metrics")metrics=value;
      else if(key=="-config")options.configurations.push_back(value);else if(key=="-trace")trace=value;
      else throw std::invalid_argument("unknown standalone argument: "+key);}
    if(metrics.empty()||trace.empty())throw std::invalid_argument("explicit trace and metrics paths required");
    ae_accelsim::standalone::GpuAssembly assembly(options);assembly.run_trace(trace,metrics);return 0;
  }catch(const std::exception& error){std::cerr<<"standalone execution failed: "<<error.what()<<'\n';return 1;}
}
