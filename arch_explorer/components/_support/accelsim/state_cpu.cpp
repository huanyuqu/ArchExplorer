// Concrete Accel-Sim resource; owned by the component provider.
// SPDX-License-Identifier: Apache-2.0
// CPU-only boundary injection for the production measurement executor.
// This translation unit neither includes nor links the CUDA Driver.
#include "state_executor.hpp"
#include "catalog.hpp"
#include <iostream>
#include <memory>

#ifndef AE_MEASUREMENT_STATE_V1
#error state-cpu requires a generated versioned measurement-state catalog
#endif
#ifndef AE_STATE_EXECUTOR_SHA256
#error AE_STATE_EXECUTOR_SHA256 must bind the compiled executor source
#endif
#ifndef AE_STATE_CPU_SHA256
#error AE_STATE_CPU_SHA256 must bind this CPU translation unit
#endif

namespace {
constexpr bool source_sha(const char* s,size_t n) {
  if(n!=65||s[64]!='\0')return false;
  for(size_t i=0;i<64;++i)if(!((s[i]>='0'&&s[i]<='9')||(s[i]>='a'&&s[i]<='f')))return false;
  return true;
}
static_assert(source_sha(AE_STATE_EXECUTOR_SHA256,sizeof(AE_STATE_EXECUTOR_SHA256)),"invalid executor source SHA");
static_assert(source_sha(AE_STATE_CPU_SHA256,sizeof(AE_STATE_CPU_SHA256)),"invalid CPU source SHA");
const char executor_source[] __attribute__((section(".ae_state_exec"),used,aligned(1)))=AE_STATE_EXECUTOR_SHA256;
const char cpu_source[] __attribute__((section(".ae_state_cpu"),used,aligned(1)))=AE_STATE_CPU_SHA256;

using ae_state::Bytes;
using ae_state::Handle;
using ae_state::Program;
using ae_state::Slot;

std::vector<uint32_t> words(const Bytes& input) {
  ae_state::require(input.size()%4==0,"CPU oracle u32 input length differs");
  std::vector<uint32_t> result;
  for(size_t i=0;i<input.size();i+=4)
    result.push_back(uint32_t(input[i])|(uint32_t(input[i+1])<<8)|(uint32_t(input[i+2])<<16)|(uint32_t(input[i+3])<<24));
  return result;
}
Bytes pack(const std::vector<uint32_t>& input) {
  Bytes result;
  for(uint32_t x:input)for(unsigned shift=0;shift<32;shift+=8)result.push_back(static_cast<uint8_t>(x>>shift));
  return result;
}

class FakeDriver final:public ae_state::Driver {
  uint64_t now_,deadline_;Handle next_=4096,start_=0,stop_=0;
  std::string fault_;
  std::map<Handle,Bytes> memory_;
  std::map<Handle,std::string> names_;
  std::set<Handle> events_;
  unsigned interval_=0,launches_=0;
  bool final_downloaded_=false;
  const Slot* current_=nullptr;
  void tick() {ae_state::require(now_<=std::numeric_limits<uint64_t>::max()-100,"fake clock overflow");now_+=100;}
 public:
  FakeDriver(uint64_t start,uint64_t deadline,std::string fault):now_(start),deadline_(deadline),fault_(std::move(fault)) {}
  std::string kind() const override {return "synthetic";}
  std::string context() const override {return "synthetic-cpu-context";}
  uint64_t now_ns() override {return now_;}
  int allocate(const std::string& name,size_t n,Handle& output) override {
    tick();if(interval_)return 700;
    if(fault_=="allocate-"+name){output=0;return 2;}
    output=next_++;memory_.emplace(output,Bytes(n));names_.emplace(output,name);
    // A nonzero error code does not erase an API output handle. Execution must
    // still record and release that handle once, through its ordinary cleanup.
    return fault_=="allocate-"+name+"-handle"?2:0;
  }
  int upload(Handle address,const Bytes& value) override {
    tick();if(interval_||!memory_.count(address)||memory_.at(address).size()!=value.size())return 700;
    if(fault_=="upload-"+names_.at(address))return 1;
    memory_[address]=value;return 0;
  }
  int context_synchronize() override {
    tick();interval_=0;return fault_=="cleanup-sync"&&final_downloaded_?201:0;
  }
  int event_create(Handle& output) override {
    tick();if(interval_)return 700;
    if(start_&&!stop_&&fault_=="event-stop"){output=0;return 2;}
    output=next_++;events_.insert(output);
    if(!start_)start_=output;else if(!stop_)stop_=output;else return 700;
    return fault_=="event-stop-handle"&&output==stop_?2:0;
  }
  int event_record(Handle event) override {
    tick();if(!events_.count(event))return 700;
    if(event==start_&&interval_==0){interval_=1;return 0;}
    if(event==stop_&&interval_==2){interval_=0;return 0;}
    return 700;
  }
  int launch(const Program& program,const ae_state::Buffers& buffers,const Slot& slot) override {
    tick();
    if(interval_!=1||slot.sequence!=launches_||!memory_.count(buffers.input)||!memory_.count(buffers.output))return 700;
    ++launches_;current_=&slot;
    if(fault_=="launch"&&slot.sequence==0)return 719;
    auto values=words(memory_.at(buffers.input));
    if(program.family=="chase") {
      if(!memory_.count(buffers.auxiliary))return 700;
      auto positions=words(memory_.at(buffers.auxiliary));
      for(unsigned n=0;n<program.steps;++n)
        for(auto& position:positions){if(position>=values.size())return 700;position=values[position];}
      memory_[buffers.output]=pack(positions);
    } else {
      auto constants=words(program.auxiliary);if(constants.size()!=2)return 700;
      for(unsigned n=0;n<program.steps;++n)
        for(auto& value:values)value=value*constants[0]+constants[1];
      memory_[buffers.output]=pack(values);
    }
    if(fault_=="preparation"&&slot.role=="preparation"&&slot.role_index==0)memory_[buffers.output][0]^=1;
    if(fault_=="timeout"&&slot.sequence==1)now_=deadline_+1;
    interval_=2;return 0;
  }
  int event_synchronize(Handle event) override {tick();return !interval_&&event==stop_&&events_.count(event)?0:700;}
  int event_elapsed(float& ms,Handle a,Handle b) override {
    tick();if(interval_||a!=start_||b!=stop_||!current_)return 700;
    ms=current_->role=="diagnostic"?0.04f:current_->role=="preparation"?0.03f+0.0001f*current_->role_index:0.02f+0.0001f*current_->role_index;
    if(fault_=="quality"&&current_->role=="formal")ms=0.02f+0.01f*current_->role_index;
    if(fault_=="nonfinite"&&current_->sequence==0)ms=std::numeric_limits<float>::quiet_NaN();
    if(fault_=="zero"&&current_->sequence==0)ms=0;
    return 0;
  }
  int download(Handle address,Bytes& output) override {
    tick();if(interval_||!memory_.count(address)||memory_.at(address).size()!=output.size())return 700;
    output=memory_.at(address);
    if(current_&&current_->role=="formal"&&current_->role_index==2)final_downloaded_=true;
    return 0;
  }
  int event_destroy(Handle event) override {
    tick();if(fault_=="cleanup"&&event==stop_)return 201;
    return events_.erase(event)==1?0:700;
  }
  int free(Handle address) override {
    tick();if(names_.count(address)&&fault_=="free-"+names_.at(address))return 201;
    if(memory_.erase(address)!=1)return 700;
    names_.erase(address);return 0;
  }
};

ae_state::Plan compiled_plan() {
  auto plan=ae_measurement_generated::plan();
  ae_state::require(plan.registry_sha256()==sim30::kRegistrySha&&plan.entries().size()==sim30::kCases.size(),"state/generated program registry binding differs");
  for(size_t i=0;i<sim30::kCases.size();++i) {
    const auto& a=sim30::kCases[i];const auto& b=plan.entries()[i];
    ae_state::require(a.id==b.id&&a.function==b.function&&a.partition==b.partition,"ordered state/program catalog differs");
  }
  return plan;
}
void inspect(const ae_state::Plan& plan) {
  std::cout<<"{\"schema\":\"arch-explorer.measurement-state-inspection.v1\",\"registry_sha256\":"<<ae_state::quote(plan.registry_sha256())
    <<",\"plan_sha256\":"<<ae_state::quote(plan.plan_sha256())<<",\"directory_sha256\":"<<ae_state::quote(plan.directory_sha256())<<",\"cases\":[";
  for(size_t i=0;i<plan.entries().size();++i) {
    const auto& e=plan.entries()[i];
    std::cout<<(i?",":"")<<"{\"case\":"<<ae_state::quote(e.id)<<",\"function\":"<<ae_state::quote(e.function)<<",\"partition\":"<<ae_state::quote(e.partition)
      <<",\"diagnostic_count\":"<<e.diagnostic<<",\"preparation_count\":"<<e.preparation<<",\"formal_count\":"<<e.formal<<'}';
  }
  std::cout<<"],\"resource_authority\":false}\n";
}
} // namespace

int main(int argc,char** argv) {
  try {
    if(argc==3&&(std::string(argv[1])=="--inspect-plan"||std::string(argv[1])=="--check-plan")) {
      const auto data=ae_state::read_file(argv[2],8192);auto plan=ae_state::Plan::parse(data.data(),data.size());
      if(std::string(argv[1])=="--check-plan")compiled_plan().require_compatible(plan);
      inspect(plan);return 0;
    }
    std::map<std::string,std::string> arguments;
    const std::set<std::string> required={"--bundle","--case","--output","--run","--freeze","--start-ns","--deadline-ns"};
    for(int i=1;i<argc;i+=2) {
      ae_state::require(i+1<argc,"state-cpu option requires a value");std::string key=argv[i];
      ae_state::require(required.count(key)||key=="--fault","unknown state-cpu option");
      ae_state::require(arguments.emplace(key,argv[i+1]).second,"duplicate state-cpu option");
    }
    for(const auto& key:required)ae_state::require(arguments.count(key),"missing state-cpu option: "+key);
    const std::set<std::string> faults={"","preparation","quality","timeout","cleanup","launch","nonfinite","zero",
      "allocate-input","allocate-output","upload-input","event-stop","allocate-output-handle","event-stop-handle",
      "cleanup-sync","free-output","allocate-auxiliary","upload-auxiliary","free-auxiliary"};
    const auto fault=arguments.count("--fault")?arguments.at("--fault"):"";
    ae_state::require(faults.count(fault),"unknown injected CPU fault");
    auto plan=compiled_plan();const auto& selected=plan.find(arguments.at("--case"));
    if(fault=="preparation")ae_state::require(selected.preparation>0,"preparation fault requires a registered preparation slot");
    const ae_state::fs::path bundle=arguments.at("--bundle"),output=arguments.at("--output");
    ae_state::require(bundle.is_absolute()&&output.is_absolute(),"absolute CPU bundle/output required");
    ae_state::require(ae_state::reference(bundle/"REGISTRY.json").sha==plan.registry_sha256(),"compiled registry bytes changed");
    const sim30::Case* candidate=nullptr;
    for(const auto& c:sim30::kCases)if(c.id==arguments.at("--case"))candidate=&c;
    ae_state::require(candidate,"unknown generated CPU program");const auto& c=*candidate;
    if(fault=="allocate-auxiliary"||fault=="upload-auxiliary"||fault=="free-auxiliary")
      ae_state::require(std::string(c.family)=="chase","auxiliary fault requires a registered chase program");
    auto load=[&](const sim30::File& file){const auto data=ae_state::read_file(bundle/file.name);ae_state::require(data.size()==file.bytes&&ae_state::sha256(data)==file.sha256,"compiled input artifact differs");return data;};
    (void)load(sim30::kCubin);
    Program program{c.id,c.function,c.partition,c.family,c.steps,load(c.input),load(c.auxiliary),load(c.expected)};
    ae_state::Options options{arguments.at("--run"),arguments.at("--freeze"),
      ae_state::decimal(arguments.at("--start-ns"),std::numeric_limits<uint64_t>::max()-1000000),
      ae_state::decimal(arguments.at("--deadline-ns"),std::numeric_limits<uint64_t>::max()-1000000)};
    ae_state::require(ae_state::run_identifier(options.run)&&ae_state::digest(options.freeze),"invalid run/freeze");
    ae_state::require(options.start_ns>0&&options.start_ns<options.deadline_ns&&options.deadline_ns-options.start_ns<=120000000000ull,"invalid fixed CPU start/deadline");
    ae_state::require(ae_state::fs::create_directory(output),"new CPU output directory required");
    ae_state::write_new(output/"SYNTHETIC.json","{\"schema\":\"arch-explorer.measurement-state-cpu-boundary.v1\",\"kind\":\"synthetic\",\"run\":"+ae_state::quote(options.run)+
      ",\"case\":"+ae_state::quote(program.id)+",\"no_resource_authority\":true,\"real_world_resource_authority\":false,\"executor_source_sha256\":"+ae_state::quote(executor_source)+
      ",\"cpu_source_sha256\":"+ae_state::quote(cpu_source)+",\"executable\":"+ae_state::reference(ae_state::fs::canonical("/proc/self/exe")).json()+"}\n");
    std::string command="{\"argv\":[";
    for(int i=0;i<argc;++i)command+=(i?",":"")+ae_state::quote(argv[i]);
    ae_state::write_new(output/"COMMAND.json",command+"],\"kind\":\"synthetic\",\"device_operations\":0}\n");
    FakeDriver driver(options.start_ns,options.deadline_ns,fault);
    ae_state::Execution execution(plan,program,driver,options,output);
    const auto result=execution.run();
    std::cout<<"{\"status\":"<<ae_state::quote(result.status)<<",\"cleanup\":"<<ae_state::quote(result.cleanup_closed?"closed":"failed")
      <<",\"ledger\":"<<ae_state::reference(output/"LEDGER.json").json()<<",\"kind\":\"synthetic\",\"device_operations\":0}\n";
    return result.status=="returned"&&result.cleanup_closed?0:2;
  } catch(const std::exception& e) {
    std::cerr<<"STATE_CPU_REFUSED: "<<e.what()<<'\n';return 64;
  }
}
