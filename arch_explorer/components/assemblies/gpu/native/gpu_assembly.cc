// SPDX-License-Identifier: Apache-2.0
#include "gpu_assembly.hpp"
#include "configuration_api.hpp"
#include "execution.hpp"
#include "gpu_transport.hpp"
#include "memory_api.hpp"
#include "memory_services.hpp"
#include "transactions.hpp"
#include "../mem_latency_stat.h"
#include "../l2cache.h"
#include "trace_driven.h"
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <set>
#include <sstream>
#include <stdexcept>

namespace ae_accelsim { namespace standalone {
using ae_dynamic::Transaction;
using ae_dynamic::Direction;
namespace {
void require(bool value,const std::string& reason){if(!value)throw std::invalid_argument(reason);}
std::string quote(const std::string& value) {
  std::ostringstream out;out<<'"';
  for(unsigned char ch:value){if(ch=='"'||ch=='\\')out<<'\\'<<ch;
    else if(ch<32)out<<"\\u"<<std::hex<<std::setw(4)<<std::setfill('0')<<unsigned(ch)<<std::dec;
    else out<<ch;}
  out<<'"';return out.str();
}
struct CoreTail {
  CycleView& clock;
  ProgramExecution& program;
  std::vector<ShaderCluster*> clusters;
  unsigned last=0;
  bool flush_l1=false;
  void tick() {
    ++clock.current;
    const auto previous=last;
    for(unsigned n=0;n<clusters.size();++n) {
      const unsigned id=(n+previous+1)%clusters.size();
      const unsigned issued=clusters[id]->issue_block2core();
      if(issued){last=id;program.record_issued_ctas(issued);}
    }
    program.decrement_kernel_latency();
    if(flush_l1)for(auto* node:clusters)if(!node->busy())node->cache_invalidate();
  }
};
struct PhaseAudit {
  std::map<std::string,std::uint64_t> calls;
  void add(ae_dynamic::Runtime& runtime,const std::string& clock,int order,
           const std::string& node,const std::string& label,std::function<void()> callback) {
    auto* counters=&calls;(*counters)[label]=0;
    runtime.add_phase(clock,order,node,[counters,label,callback](const ae_dynamic::Context&){++counters->at(label);callback();});
  }
};
}

struct GpuAssembly::Impl {
  gpgpu_context context{true,true};
  gpgpu_sim_config config{&context};
  InstructionConfig instruction;
  trace_config trace;
  CycleView clock;
  RequestIds ids;
  NativeRequests requests;
  ShaderCounters counters;
  ae_dynamic::Runtime runtime;
  std::unique_ptr<ProgramExecution> program;
  MemoryServices services{{clock,context,nullptr},requests,nullptr,nullptr};
  std::unique_ptr<shader_core_stats> shader_statistics;
  std::unique_ptr<memory_stats_t> memory_statistics;
  std::unique_ptr<Icnt> network;
  MemoryLinks links{requests};
  std::unique_ptr<GpuTransport> transport;
  std::vector<std::unique_ptr<L2Component>> l2;
  std::vector<std::unique_ptr<DramComponent>> dram;
  std::vector<std::unique_ptr<ShaderCluster>> shaders;
  std::vector<std::unique_ptr<ae_dynamic::Component>> endpoints;
  std::unique_ptr<CoreTail> core_tail;
  PhaseAudit phases;
  std::uint64_t stall_l2=0,stall_icnt=0,drain_ticks=0,drain_cycles=0;
  std::vector<std::string> kernel_records;
  std::string foreground;
  bool consumed=false;

  explicit Impl(const GpuOptions& options) {
    require(!options.configurations.empty() && !options.topology.empty(),"GPU assembly requires explicit native configuration and connections");
    auto parser=option_parser_create();
    config.reg_options(parser);instruction.reg_options(parser);trace.reg_options(parser);
    unsigned mode=1;char* interconfig=nullptr;
    unsigned local_in=64,local_out=64,local_subnets=2,local_arb=1,local_verbose=0,local_grant=1;
    int occupancy_target=0;bool save_converted_ptx=false;
    // Trace records already carry resource metadata. Accept the original
    // target declaration without creating a PTX loader or invoking ptxas.
    option_parser_register(parser,"-gpgpu_occupancy_sm_number",OPT_INT32,&occupancy_target,"PTX resource target metadata","0");
    option_parser_register(parser,"-gpgpu_ptx_save_converted_ptxplus",OPT_BOOL,&save_converted_ptx,"unsupported PTX conversion output","0");
    option_parser_register(parser,"-network_mode",OPT_UINT32,&mode,"native network mode","1");
    option_parser_register(parser,"-inter_config_file",OPT_CSTR,&interconfig,"native BookSim configuration","mesh");
    option_parser_register(parser,"-icnt_in_buffer_limit",OPT_UINT32,&local_in,"native input buffer","64");
    option_parser_register(parser,"-icnt_out_buffer_limit",OPT_UINT32,&local_out,"native output buffer","64");
    option_parser_register(parser,"-icnt_subnets",OPT_UINT32,&local_subnets,"native subnets","2");
    option_parser_register(parser,"-icnt_arbiter_algo",OPT_UINT32,&local_arb,"native arbitration","1");
    option_parser_register(parser,"-icnt_verbose",OPT_UINT32,&local_verbose,"native diagnostics","0");
    option_parser_register(parser,"-icnt_grant_cycles",OPT_UINT32,&local_grant,"native grant period","1");
    for(const auto& path:options.configurations)option_parser_cfgfile(parser,path.c_str());
    option_parser_destroy(parser);
    require(!save_converted_ptx && occupancy_target>=0,"standalone trace input does not provide PTX conversion services");
    ConfigurationAccess::initialize(config,instruction);trace.parse_config();
    auto& sm=ConfigurationAccess::shader(config);auto& memory=ConfigurationAccess::memory(config);
    require(sm.n_chiplet==1 && memory.n_chiplet==1 && !memory.lrc_enabled && !sm.gpgpu_perfect_mem && !memory.SST_mode,
            "independent GPU assembly requires single chiplet, LRC off, finite native memory");
    require(!local_verbose && local_subnets==2,"native Local ICNT diagnostics/subnet mode unsupported");
    memory.ae_request_ids=&ids;memory.ae_transaction_observer=&requests;
    services.routes=&links;
    program=make_program_execution({config.get_max_concurrent_kernel(),sm.n_thread_per_shader,ConfigurationAccess::max_ctas(config)},clock,counters);
    shader_statistics=std::make_unique<shader_core_stats>(&sm);
    memory_statistics=make_memory_statistics(sm.num_shader(),sm.n_thread_per_shader/sm.warp_size,memory,clock);
    IcntConfig net;net.mode=mode;net.request_nodes=sm.n_simt_clusters;net.memory_nodes=memory.m_n_mem_sub_partition;
    if(interconfig)net.booksim_config=interconfig;
    net.local={local_in,local_out,local_arb,local_grant};network=make_icnt(net);
    transport=std::make_unique<GpuTransport>(*network,requests,links,sm.n_simt_clusters);
    for(unsigned id=0;id<memory.m_n_mem_sub_partition;++id)l2.push_back(make_l2(id,memory,*memory_statistics,services));
    for(unsigned id=0;id<memory.m_n_mem;++id) {
      std::vector<L2Component*> peers;
      for(unsigned local=0;local<memory.m_n_sub_partition_per_memory_channel;++local)
        peers.push_back(l2[id*memory.m_n_sub_partition_per_memory_channel+local].get());
      dram.push_back(make_dram(id,memory,*memory_statistics,services,peers));
    }
    for(unsigned id=0;id<sm.n_simt_clusters;++id) {
      shaders.push_back(make_shader_cluster(id,sm,memory,*shader_statistics,*memory_statistics,clock,*program,counters,*transport,services.cache,{config.get_core_freq(),config.flush_l1()}));
      transport->bind(id);
    }
    core_tail=std::make_unique<CoreTail>(CoreTail{clock,*program,{},sm.n_simt_clusters-1,config.flush_l1()});
    for(auto& node:shaders)core_tail->clusters.push_back(node.get());
    setup(options.topology);
    for(auto& node:shaders)node->reinit();shader_statistics->new_grid();
    requests.bind(runtime);
  }

  void setup(const std::string& topology) {
    std::map<std::string,std::pair<std::string,unsigned>> nodes{{"icnt",{"icnt",0}}};
    std::set<NativeEdge> edges;
    for(unsigned i=0;i<shaders.size();++i){const auto name="sm"+std::to_string(i);nodes[name]={"sm",i};
      edges.insert({name,"request","icnt","request"+std::to_string(i)});edges.insert({"icnt","reply"+std::to_string(i),name,"reply"});}
    const auto per=ConfigurationAccess::memory(config).m_n_sub_partition_per_memory_channel;
    for(unsigned i=0;i<dram.size();++i)nodes["dram"+std::to_string(i)]={"dram",i};
    for(unsigned i=0;i<l2.size();++i){const auto name="l2"+std::to_string(i),d="dram"+std::to_string(i/per),local=std::to_string(i%per);
      nodes[name]={"l2",i};edges.insert({name,"dram_request",d,"request"+local});edges.insert({d,"reply"+local,name,"dram_reply"});
      edges.insert({"icnt","request_mem"+std::to_string(i),name,"request"});edges.insert({name,"reply","icnt","reply_mem"+std::to_string(i)});
      links.connect(*l2[i],*dram[i/per],{name,"dram_request",d,"request"+local},{d,"reply"+local,name,"dram_reply"});}
    std::ifstream input(topology);std::string line;require(bool(std::getline(input,line)) && line=="arch-explorer.accelsim-topology.v1","invalid native topology header");
    std::map<std::string,std::pair<std::string,unsigned>> given;std::set<NativeEdge> connections;
    while(std::getline(input,line)) {
      std::vector<std::string> f;std::istringstream row(line);std::string value;while(std::getline(row,value,'\t'))f.push_back(value);
      if(f.size()==4 && f[0]=="node"){
        std::size_t used=0;auto index=std::stoul(f[3],&used);require(used==f[3].size() && index<=UINT32_MAX,"invalid topology index");
        require(given.emplace(f[1],std::make_pair(f[2],unsigned(index))).second,"duplicate topology node");
      }else if(f.size()==5 && f[0]=="connect")require(connections.insert({f[1],f[2],f[3],f[4]}).second,"duplicate topology edge");
      else require(false,"invalid topology row");
    }
    require(given==nodes && connections==edges,"explicit GPU topology does not match native factory instances/ports");
    for(const auto& item:nodes) {
      const auto& name=item.first;const auto kind=item.second.first;const unsigned id=item.second.second;
      ae_dynamic::Component* endpoint;
      if(kind=="sm"){endpoints.push_back(std::make_unique<ShaderEndpoint>(id,*shaders[id],*transport));endpoint=endpoints.back().get();}
      else if(kind=="l2"){endpoints.push_back(std::make_unique<L2Endpoint>(*l2[id],links));endpoint=endpoints.back().get();}
      else if(kind=="dram"){endpoints.push_back(std::make_unique<DramEndpoint>(*dram[id],links));endpoint=endpoints.back().get();}
      else endpoint=transport.get();
      std::map<std::string,Direction> names;
      for(const auto& edge:edges){if(edge[0]==name)names.emplace(edge[1],Direction::output);if(edge[2]==name)names.emplace(edge[3],Direction::input);}
      std::vector<ae_dynamic::Port> ports;for(auto p:names)ports.push_back({p.first,p.second});runtime.add_component(name,*endpoint,ports);
    }
    for(const auto& edge:edges){runtime.connect(edge[0],edge[1],edge[2],edge[3]);links.declare(edge);}
    auto periods=ConfigurationAccess::periods(config);
    runtime.add_clock("core",periods[0]);runtime.add_clock("icnt",periods[1]);runtime.add_clock("l2",periods[2]);runtime.add_clock("dram",periods[3]);
    for(unsigned i=0;i<shaders.size();++i) {
      auto* node=shaders[i].get();auto* arena=&requests;auto* dispatch=program.get();const auto name="sm"+std::to_string(i);
      phases.add(runtime,"icnt",10,name,name+".receive",[node,arena]{node->receive_cycle();arena->complete_destroyed();});
      phases.add(runtime,"core",70,name,name+".advance",[node,arena,dispatch]{if(node->busy()||dispatch->get_more_cta_left())node->core_cycle();arena->complete_destroyed();});
    }
    for(unsigned i=0;i<l2.size();++i){auto* node=l2[i].get();auto* pipe=transport.get();auto* stalls=&stall_l2;auto* blocked=&stall_icnt;const auto name="l2"+std::to_string(i);
      phases.add(runtime,"l2",30,name,name+".reply",[node,pipe,stalls]{pipe->send_l2(*node,*stalls);});
      phases.add(runtime,"l2",50,name,name+".advance",[node,pipe,blocked]{pipe->receive_l2(*node,*blocked);});}
    for(unsigned i=0;i<dram.size();++i){auto* node=dram[i].get();auto* arena=&requests;const auto name="dram"+std::to_string(i);
      phases.add(runtime,"dram",40,name,name+".advance",[node,arena]{node->tick();arena->complete_destroyed();});}
    auto* net=network.get();phases.add(runtime,"icnt",60,"icnt","icnt.advance",[net]{net->advance();});
    auto* tail=core_tail.get();phases.add(runtime,"core",71,"icnt","core.retire_dispatch",[tail]{tail->tick();});
  }
  ~Impl(){requests.cancel();shaders.clear();dram.clear();l2.clear();network.reset();memory_statistics.reset();shader_statistics.reset();}
  unsigned native_busy() const {
    unsigned n=network->busy();for(const auto& p:shaders)n+=!p->drained();for(const auto& p:l2)n+=!p->drained();for(const auto& p:dram)n+=!p->drained();return n;
  }
  bool drained() const{return !native_busy() && !requests.pending() && runtime.drained();}
  void tick() {
    auto max=ConfigurationAccess::max_cycles(config),insn=ConfigurationAccess::max_instructions(config);
    if((max && clock.absolute()>=max)||(insn && counters.completed_instructions+counters.instructions>=insn)||program->hit_max_cta_count())
      throw std::runtime_error("native execution reached configured cycle/instruction/CTA limit before completion");
    runtime.tick();requests.complete_destroyed();
  }
  std::string snapshot() const;
  void metrics(const std::string& path) const;
  void run(const std::string& list,const std::string& path);
};

GpuAssembly::GpuAssembly(const GpuOptions& options):impl_(new Impl(options)) {}
GpuAssembly::~GpuAssembly()=default;
void GpuAssembly::run_trace(const std::string& kernels,const std::string& metrics){impl_->run(kernels,metrics);}

void GpuAssembly::Impl::run(const std::string& list,const std::string& path) {
  require(!consumed,"GPU assembly program already consumed");consumed=true;
  auto parser=std::make_shared<trace_parser>(list.c_str());const auto commands=parser->parse_commandlist_file();require(!commands.empty(),"empty native trace program");
  std::cout<<"Accel-Sim [1c9cc83] GPGPU-Sim [c24e885] standalone factories\n";
  for(const auto& command:commands) {
    require(command.m_type==command_type::kernel_launch,"standalone trace entry accepts explicit kernels; memcpy/functional operations are not services");
    auto parsed=std::shared_ptr<kernel_trace_t>(parser->parse_kernel_info(command.command_string),
      [parser](kernel_trace_t* value){if(value)parser->kernel_finalizer(value);});
    auto* raw=parsed.get();require(raw && raw->cuda_stream_id==0,"this ordered-program entry requires stream zero");
    gpgpu_ptx_sim_info info{};info.smem=raw->shmem;info.regs=raw->nregs;
    auto function=std::make_shared<trace_function_info>(info,&context);function->set_name(raw->kernel_name.c_str());
    std::shared_ptr<kernel_info_t> kernel(new trace_kernel_info_t(
      dim3(raw->grid_dim_x,raw->grid_dim_y,raw->grid_dim_z),dim3(raw->tb_dim_x,raw->tb_dim_y,raw->tb_dim_z),function.get(),parser.get(),&trace,raw),
      [parsed,function](kernel_info_t* p){delete p;});
    const auto uid=kernel->get_uid();
    std::cout<<"launching kernel name: "<<raw->kernel_name<<" uid: "<<uid<<" cuda_stream_id: "<<raw->cuda_stream_id<<'\n';
    program->launch(kernel);
    unsigned finished=0;
    while(!finished){tick();finished=program->finished_kernel();}
    require(finished==uid,"ordered program retired another kernel");
    kernel_records.push_back("{\"uid\":"+std::to_string(uid)+",\"metrics\":"+snapshot()+"}");
    std::uint64_t warps=0;auto& sm=ConfigurationAccess::shader(config);
    for(unsigned i=0;i<sm.num_shader();++i)warps+=shader_statistics->m_num_sim_winsn[i];
    std::cout<<"kernel_name = "<<raw->kernel_name<<"\nkernel_launch_uid = "<<uid
      <<"\ngpu_sim_cycle = "<<clock.current<<"\ngpu_tot_sim_cycle = "<<clock.absolute()
      <<"\ngpu_sim_insn = "<<counters.instructions<<"\ngpu_tot_sim_insn = "<<counters.completed_instructions+counters.instructions
      <<"\ngpu_tot_issued_cta = "<<program->issued_ctas()<<"\ngpgpu_n_tot_w_icount = "<<warps<<'\n';
    auto retired=program->release_finished(uid);kernel.reset();retired.reset();
    memory_statistics->memlatstat_lat_pw();
    clock.completed+=clock.current;clock.current=0;counters.completed_instructions+=counters.instructions;counters.instructions=0;counters.completed_ctas=0;
  }
  requests.complete_destroyed();foreground=snapshot();const auto cycle=clock.absolute();
  while(!drained()){tick();++drain_ticks;}
  drain_cycles=clock.absolute()-cycle;metrics(path);
  std::cout<<"GPGPU-Sim: *** exit detected ***\nGPGPU-Sim: *** simulation thread exiting ***\n";
}

std::string GpuAssembly::Impl::snapshot() const {
  const auto net=network->metrics();std::ostringstream out;
  out<<"{\"cycles\":"<<clock.current<<",\"total_cycles\":"<<clock.absolute()<<",\"instructions\":"<<counters.instructions
    <<",\"total_instructions\":"<<counters.completed_instructions+counters.instructions<<",\"total_cta\":"<<program->issued_ctas()
    <<",\"stall_core_to_icnt\":"<<counters.core_to_icnt_stalls<<",\"stall_icnt_to_l2\":"<<stall_icnt
    <<",\"stall_l2_to_icnt\":"<<stall_l2<<",\"stall_icnt_to_core\":"<<counters.icnt_to_core_stalls
    <<",\"icnt_injected\":"<<net.accepted<<",\"icnt_ejected\":"<<net.delivered<<",\"icnt_packet_bytes\":"<<net.transfer_bytes<<",\"l2\":[";
  for(unsigned i=0;i<l2.size();++i){if(i)out<<',';const auto s=l2[i]->metrics();
    out<<"{\"index\":"<<i<<",\"accesses\":"<<s.accesses<<",\"misses\":"<<s.misses<<",\"hits\":"<<s.hits
      <<",\"pending_hits\":"<<s.pending_hits<<",\"reservation_fails\":"<<s.reservation_fails
      <<",\"port_available_cycles\":"<<s.port_available_cycles<<",\"data_port_busy_cycles\":"<<s.data_port_busy_cycles<<",\"fill_port_busy_cycles\":"<<s.fill_port_busy_cycles<<'}';}
  out<<"],\"dram\":[";
  for(unsigned i=0;i<dram.size();++i){if(i)out<<',';const auto s=dram[i]->metrics();
    out<<"{\"index\":"<<i<<",\"commands\":"<<s.commands<<",\"activity\":"<<s.activity<<",\"nop\":"<<s.nop
      <<",\"activate\":"<<s.act<<",\"precharge\":"<<s.pre<<",\"read\":"<<s.reads<<",\"write\":"<<s.writes<<",\"writeback\":"<<s.writebacks<<",\"requests\":"<<s.requests<<'}';}
  out<<"],\"native_busy\":"<<native_busy()<<",\"pending\":"<<runtime.pending()<<",\"system_drained\":"<<(drained()?"true":"false")<<'}';return out.str();
}
void GpuAssembly::Impl::metrics(const std::string& path) const {
  std::ofstream out(path);require(bool(out),"cannot create native metrics");const auto periods=ConfigurationAccess::periods(config);const auto& audit=runtime.audit();
  out<<"{\"schema\":\"arch-explorer.accelsim-component-metrics.v2\",\"mode\":\"components\",\"original_cycle_calls\":0,\"component_ticks\":"<<audit.ticks
    <<",\"clock_periods\":{\"core\":"<<std::setprecision(17)<<periods[0]<<",\"icnt\":"<<periods[1]<<",\"l2\":"<<periods[2]<<",\"dram\":"<<periods[3]<<"},\"phase_calls\":{";
  bool first=true;for(const auto& p:phases.calls){if(!first)out<<',';first=false;out<<quote(p.first)<<':'<<p.second;}
  out<<"},\"routing\":{\"ticks\":"<<audit.ticks<<",\"phase_calls\":"<<audit.phase_calls<<",\"accepted\":"<<audit.accepted<<",\"rejected\":"<<audit.rejected
    <<",\"wakeups\":"<<audit.wakeups<<",\"consumer_ready\":"<<audit.consumer_ready<<",\"physical_complete\":"<<audit.physical_complete<<",\"resource_ready\":"<<audit.resource_ready<<",\"completed\":"<<audit.completed<<"},\"ports\":[";
  first=true;for(const auto& item:links.counters()){if(!first)out<<',';first=false;const auto& edge=item.first;const auto& s=item.second;
    out<<"{\"source\":"<<quote(edge[0])<<",\"source_port\":"<<quote(edge[1])<<",\"destination\":"<<quote(edge[2])<<",\"destination_port\":"<<quote(edge[3])
      <<",\"accepted\":"<<s.accepted<<",\"logical_bytes\":"<<s.logical_bytes<<",\"transfer_bytes\":"<<s.transfer_bytes<<'}';}
  out<<"],\"lifecycle\":{\"native_requests\":"<<requests.lifetimes().size()<<",\"native_destroyed\":"<<requests.native_deleted()<<",\"consumer_ready\":"<<requests.consumer_count()
    <<",\"physical_complete\":"<<requests.physical_count()<<",\"resource_ready\":"<<requests.resource_count()<<",\"pending\":"<<requests.pending()
    <<"},\"allocation_domain\":{\"allocated\":"<<requests.allocated_count()<<",\"destroyed\":"<<requests.destroyed_count()<<",\"live\":"<<requests.live_count()
    <<"},\"oracle_observations\":{\"schema\":\"arch-explorer.accelsim-component-metrics.v1\",\"ports\":[";
  first=true;for(const auto& item:links.counters()){if(!first)out<<',';first=false;const auto& edge=item.first;const auto& s=item.second;
    out<<"{\"source\":"<<quote(edge[0])<<",\"source_port\":"<<quote(edge[1])<<",\"destination\":"<<quote(edge[2])<<",\"destination_port\":"<<quote(edge[3])
      <<",\"accepted\":"<<s.accepted<<",\"logical_bytes\":"<<s.logical_bytes<<",\"transfer_bytes\":"<<s.oracle_observed_bytes<<'}';}
  out<<"]},\"source\":{\"requests\":0,\"accepted\":0,\"completed\":0,\"rejected\":0},\"kernels\":[";
  first=true;for(const auto& record:kernel_records){if(!first)out<<',';first=false;out<<record;}
  out<<"],\"foreground_final\":"<<foreground<<",\"drain_ticks\":"<<drain_ticks<<",\"drain_core_cycles\":"<<drain_cycles<<",\"final\":"<<snapshot()<<"}\n";
  out.close();require(bool(out),"native metrics write failed");
}
}}
