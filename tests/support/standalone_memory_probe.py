"""External public-SDK caller generator; not a cache/DRAM implementation."""
import json

SOURCE = r'''
#include "memory_system.hpp"
#include <algorithm>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <set>
#include <sstream>
#include <stdexcept>
using namespace ae_accelsim::standalone;
using ae_dynamic::Transaction;
std::ofstream out;
void need(bool b,const char* why){if(!b)throw std::runtime_error(why);}
std::string boolean(bool x){return x?"true":"false";}
std::string snapshot(const MemorySnapshot& s){
 std::ostringstream o;o<<"{\"cycles\":"<<s.cycles<<",\"submitted\":"<<s.submitted<<",\"received\":"<<s.received
  <<",\"rejected\":"<<s.rejected<<",\"pending\":"<<s.pending<<",\"physical\":"<<s.physical
  <<",\"consumer\":"<<s.consumer<<",\"resource\":"<<s.resource<<",\"drained\":"<<boolean(s.drained)<<",\"l2\":[";
 for(unsigned i=0;i<s.l2.size();++i){const auto& x=s.l2[i];if(i)o<<',';
  o<<'['<<x.accesses<<','<<x.misses<<','<<x.hits<<','<<x.pending_hits<<','<<x.reservation_fails<<','
   <<x.port_available_cycles<<','<<x.data_port_busy_cycles<<','<<x.fill_port_busy_cycles<<','<<x.tracked<<','
   <<x.ingress<<','<<x.dram_requests<<','<<x.dram_replies<<','<<x.responses<<','<<x.rop<<']';}
 o<<"],\"dram\":[";for(unsigned i=0;i<s.dram.size();++i){const auto& x=s.dram[i];if(i)o<<',';
  o<<'['<<x.commands<<','<<x.activity<<','<<x.nop<<','<<x.act<<','<<x.pre<<','<<x.reads<<','<<x.writes<<','
   <<x.writebacks<<','<<x.requests<<','<<x.delay<<','<<x.controller<<','<<x.read_write<<','<<x.replies<<','
   <<x.scheduled_reads<<','<<x.scheduled_writes<<','<<x.bank_requests<<']';}
 o<<"]}";return o.str();
}
void record(const std::string& phase,const std::string& name,const std::string& kind,uint64_t tick,const std::string& more=""){
 out<<"{\"phase\":\""<<phase<<"\",\"instance\":\""<<name<<"\",\"kind\":\""<<kind<<"\",\"tick\":"<<tick<<more<<"}\n";out.flush();
}
struct Spec{unsigned id;uint64_t address;unsigned bytes;bool write;std::vector<unsigned> after;};
// SPECS
std::string options_json(const MemoryOptions& x){std::ostringstream o;o<<"{\"channels\":"<<x.channels<<",\"subpartitions\":"<<x.subpartitions_per_channel
 <<",\"cache_sets\":"<<x.cache_sets<<",\"cache_ways\":"<<x.cache_ways<<",\"queue\":"<<x.controller_queue<<",\"cache_period\":"<<x.cache_period
 <<",\"dram_period\":"<<x.dram_period<<",\"simple_dram\":"<<boolean(x.simple_dram)<<'}';return o.str();}
void validate(const Transaction& r,uint64_t id,uint64_t address,unsigned bytes,bool write){
 need(r.id==id && !r.parent_id && r.address==address && r.bytes==bytes,"logical root id/address/width differs");
 need(r.operation==(write?"write_ack":"read_reply"),"logical operation differs");
 need(r.byte_mask.size()==128,"native mask must preserve 128B line geometry");
 for(unsigned i=0;i<128;++i)need(r.byte_mask[i]==unsigned(i>=address%128 && i<address%128+bytes),"logical mask incomplete/overbroad");
 need(r.transfer_bytes==(bytes/32)*(write?8:40),"sector physical response lengths differ");
}
void receipt(const std::string& phase,const std::string& name,uint64_t tick,const Transaction& r,unsigned client){
 std::ostringstream x;x<<",\"client_id\":"<<client<<",\"wire_id\":"<<r.id<<",\"parent_id\":"<<r.parent_id<<",\"address\":"<<r.address
 <<",\"bytes\":"<<r.bytes<<",\"operation\":\""<<r.operation<<"\",\"transfer_bytes\":"<<r.transfer_bytes<<",\"mask\":[";
 for(unsigned i=0;i<r.byte_mask.size();++i){if(i)x<<',';x<<unsigned(r.byte_mask[i]);}x<<']';record(phase,name,"reply",tick,x.str());
}
struct Driver{
 std::string phase,name;MemoryOptions options;std::vector<Spec> specs;std::unique_ptr<MemorySystem> memory;
 std::set<unsigned> issued,completed;std::map<uint64_t,const Spec*> handles;unsigned steps=0;
 Driver(std::string p,std::string n,MemoryOptions o,std::vector<Spec> s):phase(p),name(n),options(o),specs(s),memory(new MemorySystem(options)){
  record(phase,name,"create",0,",\"configuration\":"+options_json(options));}
 bool done()const{return completed.size()==specs.size() && memory->drained();}
 void step(){
  need(++steps<200000,"memory did not make bounded physical progress");
  for(const auto& s:specs)if(!issued.count(s.id) && std::all_of(s.after.begin(),s.after.end(),[&](unsigned d){return completed.count(d);})){
   auto wire=memory->submit(s.address,s.bytes,s.write);need(handles.emplace(wire,&s).second,"native ID reused");issued.insert(s.id);
   std::ostringstream x;x<<",\"client_id\":"<<s.id<<",\"wire_id\":"<<wire<<",\"address\":"<<s.address<<",\"bytes\":"<<s.bytes<<",\"operation\":\""<<(s.write?"write":"read")<<"\"";
   record(phase,name,"offer",memory->snapshot().cycles,x.str());}
  const auto before=snapshot(memory->snapshot());bool available=memory->drained();const auto after=snapshot(memory->snapshot());
  need(before==after,"public passive snapshot/drained advanced state");
  record(phase,name,"query",memory->snapshot().cycles,",\"available\":"+boolean(available)+",\"before\":"+before+",\"after\":"+after);
  memory->tick();Transaction r;while(memory->receive(r)){
   auto it=handles.find(r.id);need(it!=handles.end(),"unknown sector leaked to logical user");const auto& s=*it->second;
   need(!completed.count(s.id),"logical request completed twice");validate(r,r.id,s.address,s.bytes,s.write);completed.insert(s.id);receipt(phase,name,memory->snapshot().cycles,r,s.id);}
 }
 void close(){need(done(),"destroy before real physical drain");auto s=memory->snapshot();
  need(!s.pending && s.physical==s.consumer && s.consumer==s.resource,"wire lifecycle did not close");
  record(phase,name,"physical",s.cycles,",\"snapshot\":"+snapshot(s));memory.reset();record(phase,name,"destroy",s.cycles);}
};
void root_case(unsigned bytes,unsigned offset,bool write,uint64_t line=4096){
 MemorySystem system;auto id=system.submit(line+offset,bytes,write);unsigned received=0,partial=0;Transaction r;MemorySnapshot s;
 for(unsigned step=0;step<200000;++step){system.tick();s=system.snapshot();
  if(s.physical<bytes/32){need(!s.received,"root completed before its sectors");if(s.physical)++partial;}
  while(system.receive(r)){validate(r,id,line+offset,bytes,write);need(++received==1,"root returned multiple receipts");receipt("roots","root",s.cycles,r,1);}
  if(system.drained())break;}
 need(system.drained() && received==1 && !s.pending && s.submitted==1 && s.received==1,"valid root did not complete/drain");
 need(s.physical>=bytes/32 && s.physical==s.consumer && s.consumer==s.resource,"root lifecycle incomplete");
 record("roots","root","case",s.cycles,",\"length\":"+std::to_string(bytes)+",\"offset\":"+std::to_string(offset)+",\"write\":"+boolean(write)+",\"partial_samples\":"+std::to_string(partial)+",\"snapshot\":"+snapshot(s));
}
void pressure_case(MemoryOptions a){
 Driver pressure("pressure","pressure",a,spec_pressure);while(!pressure.done())pressure.step();auto s=pressure.memory->snapshot();uint64_t hit=0,miss=0,wb=0,blocked=0;
 for(const auto& x:s.l2){hit+=x.hits;miss+=x.misses;blocked+=x.reservation_fails;}for(const auto& x:s.dram)wb+=x.writebacks;
 need(hit && miss && wb,"pressure did not exercise actual cache hits/misses/dirty eviction");
 // An accepted source queue is not a promise of immediate cache admission.
 // Keep outer source rejection distinct from native cache reservation failure.
 need(blocked || s.rejected,"neither native cache nor outer admission experienced backpressure");pressure.close();
}
int main(int argc,char** argv){try{
 need(argc==2 || argc==3,"output jsonl path required");out.open(argv[1]);need(bool(out),"cannot create observer log");
 if(argc==3){need(std::string(argv[2])=="pressure-only","unknown test selection");MemoryOptions p;p.channels=1;p.cache_sets=2;p.cache_ways=1;p.controller_queue=1;p.cache_period=2;p.dram_period=3;
  pressure_case(p);record("summary","all","passed",0);return 0;}
 root_case(32,0,false);root_case(64,0,false);root_case(64,32,true);root_case(96,32,false);root_case(128,0,true);
 root_case(64,64,false,std::numeric_limits<uint64_t>::max()-127);
 MemoryOptions a,b;a.channels=1;a.cache_sets=2;a.cache_ways=1;a.controller_queue=1;a.cache_period=2;a.dram_period=3;
 b.channels=2;b.cache_sets=2;b.cache_ways=1;b.controller_queue=2;b.cache_period=5;b.dram_period=7;
 {Driver solo("solo","mem-b",b,spec_b);while(!solo.done())solo.step();solo.close();}
 {Driver first("interleaved","mem-a",a,spec_a),second("interleaved","mem-b",b,spec_b);
  for(unsigned i=0;i<30;++i){if(!first.done())first.step();if(!second.done())second.step();}
  unsigned cross=0;while(!first.done()){auto before=snapshot(second.memory->snapshot());first.step();need(before==snapshot(second.memory->snapshot()),"A advanced B state");++cross;}
  auto before=snapshot(second.memory->snapshot());first.close();need(before==snapshot(second.memory->snapshot()),"destroy A corrupted B");
  for(unsigned kind=0;kind<6;++kind){MemoryOptions bad=a;if(kind==0)bad.channels=3;if(kind==1)bad.controller_queue=0;if(kind==2)bad.cache_sets=3;
   if(kind==3)bad.cache_period=0;if(kind==4)bad.dram_period=std::numeric_limits<double>::quiet_NaN();if(kind==5)bad.cache_ways=0;
   bool refused=false;try{MemorySystem invalid(bad);}catch(const std::invalid_argument&){refused=true;}need(refused,"invalid factory config accepted");need(before==snapshot(second.memory->snapshot()),"bad factory corrupted B");}
  {auto doomed=std::make_unique<MemorySystem>(a);doomed->submit(4096,64,false);doomed->tick();auto d=doomed->snapshot();need(!doomed->drained(),"inflight cancellation not exercised");
   record("controls","cancel","before-destroy",d.cycles,",\"snapshot\":"+snapshot(d));doomed.reset();}
  need(before==snapshot(second.memory->snapshot()),"cancel A corrupted B");while(!second.done())second.step();second.close();
  record("controls","peer","isolation",0,",\"cross_checks\":"+std::to_string(cross)+",\"invalid_config_rejections\":6,\"inflight_cancel\":true");}
 pressure_case(a);
 record("summary","all","passed",0);return 0;
 }catch(const std::exception& error){if(out)record("summary","all","failed",0);std::cerr<<error.what()<<'\n';return 1;}}
'''


def source(plans):
    declarations = []
    for name, requests in plans.items():
        values = []
        for r in requests:
            deps = ','.join(str(x) for x in r.get('after', []))
            values.append('{%d,%dULL,%d,%s,{%s}}' %
                          (r['id'], r['address'], r['bytes'],
                           'true' if r['operation'] == 'write' else 'false', deps))
        declarations.append('const std::vector<Spec> spec_%s={%s};' % (name, ','.join(values)))
    return SOURCE.replace('// SPECS', '\n'.join(declarations))
