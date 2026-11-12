"""Low-level factory/deallocation witness via the documented native-types SDK.

The delete observer is external to the native library, with a positive control;
no native mechanism/host class is patched or replaced. It is not user timing.
"""
SOURCE = r'''
#include "configuration_api.hpp"
#include "memory_api.hpp"
#include "memory_links.hpp"
#include "transactions.hpp"
#include "shader_api.hpp"
#include "mem_latency_stat.h"
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <new>
#include <stdexcept>
using namespace ae_accelsim::standalone;
static void* volatile watched=nullptr;
static volatile unsigned native_deletes=0;
void operator delete(void* p) noexcept {if(p && p==watched)++native_deletes;std::free(p);}
void operator delete(void* p,std::size_t) noexcept {::operator delete(p);}
void operator delete[](void* p) noexcept {::operator delete(p);}
void operator delete[](void* p,std::size_t) noexcept {::operator delete(p);}
void need(bool b,const char* text){if(!b)throw std::runtime_error(text);}
struct PassiveSource:ae_dynamic::Component {
 bool can_accept(const ae_dynamic::Transaction&,const ae_dynamic::Context&)const override{return false;}
 bool accept(const ae_dynamic::Transaction&,const ae_dynamic::Context&)override{return false;}
 bool drained()const override{return true;}
};
void memory_factories(){
 void* control=::operator new(64);watched=control;native_deletes=0;::operator delete(control);
 need(native_deletes==1,"external delete witness positive control failed");watched=nullptr;
 gpgpu_context context(true,false);memory_config config(&context);CycleView clock;RequestIds ids;
 auto arena=std::make_unique<NativeRequests>();ae_dynamic::Runtime runtime;
 auto parser=option_parser_create();config.reg_options(parser);
 option_parser_delimited_string(parser,"-gpgpu_n_mem 1;-gpgpu_n_sub_partition_per_mchannel 1;-gpgpu_lrc_enabled 0;-gpgpu_cache:dl2_texture_only 0;-gpgpu_memory_partition_indexing 0;-gpgpu_simple_dram_model 0;-gpgpu_cache:dl2 S:2:128:1,L:B:m:L:L,A:16:4,4:0,32;"," ;");
 option_parser_destroy(parser);config.ae_request_ids=&ids;config.ae_transaction_observer=arena.get();config.init(nullptr);
 need(!context.the_gpgpusim && !context.func_sim && !context.device_runtime && !context.stats,"memory initialized unrelated GPU/SM services");
 MemoryLinks links(*arena);MemoryServices services{{clock,context,nullptr},*arena,&links,nullptr};
 auto stats=make_memory_statistics(1,0,config,clock);
 auto l2=make_l2(0,config,*stats,services);need(l2->drained(),"standalone L2 not initially idle");
 bool missing=false;try{make_dram(0,config,*stats,services,{});}catch(const std::invalid_argument&){missing=true;}
 need(missing,"DRAM accepted absent borrowed L2");
 auto dram=make_dram(0,config,*stats,services,{l2.get()});need(dram->drained(),"standalone DRAM not initially idle");
 arena->bind(runtime);PassiveSource source;L2Endpoint endpoint(*l2,links);
 runtime.add_component("source",source,{{"request",ae_dynamic::Direction::output}});
 runtime.add_component("l2",endpoint,{{"request",ae_dynamic::Direction::input}});
 runtime.connect("source","request","l2","request");runtime.add_clock("source",1);
 const auto tx=arena->create(4096,32,false,config,context,clock);watched=arena->resolve(tx.id);native_deletes=0;
 bool accepted=false;runtime.add_phase("source",0,"source",[&](const ae_dynamic::Context&){accepted=links.forward(tx,{"source","request","l2","request"});});
 runtime.tick();need(accepted && !l2->drained() && arena->pending()==1,"real in-flight L2 request not created");
 need(runtime.audit().accepted==1 && !runtime.audit().completed,"inflight route already completed");
 arena->cancel();dram.reset();l2.reset();stats.reset();arena.reset();
 need(native_deletes==1,"native in-flight allocation not freed exactly once");watched=nullptr;
 need(runtime.audit().completed==0 && !runtime.audit().physical_complete && !runtime.audit().consumer_ready && !runtime.audit().resource_ready,"cancel invented successful lifecycle events");
}
struct IdleProgram:ProgramDispatch {
 kernel_info_t* select_kernel()override{return nullptr;}
 bool kernel_more_cta_left(kernel_info_t*)const override{return false;}
 void set_kernel_done(kernel_info_t*)override{throw std::logic_error("unexpected completion");}
 void inc_completed_cta()override{throw std::logic_error("unexpected CTA");}
};
struct IdleTransport:ShaderTransport {
 bool can_send(unsigned,unsigned)const override{return false;}
 void send(unsigned,unsigned,mem_fetch*,unsigned)override{throw std::logic_error("unexpected request");}
 mem_fetch* pop_reply(unsigned)override{return nullptr;}
};
void shader_factory(){
 bool default_bad=false;try{gpgpu_context disallowed;}catch(const std::exception&){default_bad=true;}need(default_bad,"default whole context accepted");
 gpgpu_context context(true,true);gpgpu_sim_config config(&context);InstructionConfig timing;
 auto parser=option_parser_create();config.reg_options(parser);timing.reg_options(parser);
 option_parser_delimited_string(parser,
  "-gpgpu_n_clusters 1;-gpgpu_n_cores_per_cluster 1;-gpgpu_n_mem 2;-gpgpu_n_sub_partition_per_mchannel 1;"
  "-gpgpu_n_chiplet_partition 1;-gpgpu_lrc_enabled 0;-visualizer_enabled 0;-power_simulation_enabled 0;"
  "-gpgpu_cache:il1 N:64:128:16,L:R:f:N:L,S:2:48,4;"
  "-gpgpu_tex_cache:l1 N:4:128:256,L:R:m:N:L,T:512:8,128:2;"
  "-gpgpu_const_cache:l1 N:128:64:8,L:R:f:N:L,S:2:64,4;"
  "-gpgpu_cache:dl1 S:4:128:64,L:T:m:L:L,A:512:64,16:0,32;"
  "-gpgpu_cache:dl2 S:4:128:1,L:B:m:L:L,A:16:4,4:0,32;"," ;");
 option_parser_destroy(parser);ConfigurationAccess::initialize(config,timing);
 auto& sm=ConfigurationAccess::shader(config);auto& mem=ConfigurationAccess::memory(config);CycleView clock,foreign;RequestIds ids;NativeRequests requests;
 mem.ae_request_ids=&ids;mem.ae_transaction_observer=&requests;MemoryServices services{{clock,context,nullptr},requests,nullptr,nullptr};
 auto ms=make_memory_statistics(1,sm.n_thread_per_shader/sm.warp_size,mem,clock);shader_core_stats ss(&sm);ShaderCounters counters;IdleProgram program;IdleTransport transport;
 bool bad_clock=false;CacheServices other{foreign,context,nullptr};
 try{make_shader_cluster(0,sm,mem,ss,*ms,clock,program,counters,transport,other,{1000000000,false});}catch(const std::invalid_argument&){bad_clock=true;}
 need(bad_clock,"shader accepted foreign clock domain");
 auto cluster=make_shader_cluster(0,sm,mem,ss,*ms,clock,program,counters,transport,services.cache,{1000000000,false});cluster->reinit();cluster->receive_cycle();cluster->core_cycle();
 need(cluster->drained() && !cluster->busy() && !counters.instructions,"idle shader created work");
 need(!context.the_gpgpusim && !context.func_sim && !context.device_runtime,"shader initialized whole GPU environment");
}
int main(int argc,char** argv){try{need(argc==2,"result path required");memory_factories();shader_factory();std::ofstream f(argv[1]);f<<"{\"status\":\"passed\",\"standalone_L2\":true,\"DRAM_explicit_borrow\":true,\"SM_narrow_services_idle_cycle\":true,\"external_delete_positive_control\":1,\"native_inflight_deletes\":1,\"cancel_successful_marks\":0,\"missing_L2_rejected\":true,\"foreign_SM_clock_rejected\":true,\"default_whole_context_rejected\":true}\n";return 0;}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
'''
