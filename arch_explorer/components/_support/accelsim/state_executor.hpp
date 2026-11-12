// Concrete Accel-Sim resource; owned by the component provider.
// SPDX-License-Identifier: Apache-2.0
#pragma once
// Measurement order, role expansion and failure semantics have one owner.
// This header uses no CUDA API; production and CPU Drivers call the same code.
#include <openssl/evp.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
#include <fcntl.h>
#include <unistd.h>

namespace ae_state {
namespace fs=std::filesystem;
using Bytes=std::vector<uint8_t>;
using Handle=uint64_t;
inline void require(bool value,const std::string& why) {
  if(!value)throw std::runtime_error(why);
}
inline std::string quote(const std::string& text) {
  std::ostringstream out;out<<'"';
  for(unsigned char c:text) {
    if(c=='"'||c=='\\')out<<'\\'<<c;
    else if(c<32)out<<"\\u"<<std::hex<<std::setw(4)<<std::setfill('0')<<unsigned(c)<<std::dec;
    else out<<c;
  }
  out<<'"';return out.str();
}
inline std::string sha256(const void* data,size_t size) {
  unsigned char hash[EVP_MAX_MD_SIZE]{};unsigned length=0;
  require(EVP_Digest(data,size,hash,&length,EVP_sha256(),nullptr)==1&&length==32,"state SHA256 failed");
  std::ostringstream out;
  for(unsigned i=0;i<length;++i)out<<std::hex<<std::setw(2)<<std::setfill('0')<<unsigned(hash[i]);
  return out.str();
}
inline std::string sha256(const Bytes& raw) {return sha256(raw.data(),raw.size());}
inline bool digest(const std::string& s) {
  return s.size()==64&&s.find_first_not_of("0123456789abcdef")==std::string::npos;
}
inline bool identifier(const std::string& s,bool function=false) {
  if(s.empty()||s.size()>128)return false;
  auto alpha=[](unsigned char c){return(c>='a'&&c<='z')||(c>='A'&&c<='Z')||c=='_';};
  if(!alpha(s[0]))return false;
  for(unsigned char c:s)if(!alpha(c)&&!(c>='0'&&c<='9')&&!(c=='-'&&!function))return false;
  return true;
}
inline bool run_identifier(const std::string& s) {
  if(s.empty()||s.size()>128||s[0]=='-')return false;
  for(unsigned char c:s)
    if(!((c>='a'&&c<='z')||(c>='A'&&c<='Z')||(c>='0'&&c<='9')||c=='_'||c=='-'))return false;
  return true;
}
inline uint64_t decimal(const std::string& s,uint64_t maximum) {
  require(!s.empty()&&(s.size()==1||s[0]!='0'),"noncanonical state integer");
  uint64_t value=0;
  for(unsigned char c:s) {
    require(c>='0'&&c<='9',"nondecimal state integer");
    require(value<=maximum/10&&(value<maximum/10||uint64_t(c-'0')<=maximum%10),"state integer exceeds bound");
    value=value*10+c-'0';
  }
  return value;
}
inline Bytes read_file(const fs::path& p,size_t bound=16*1024*1024) {
  require(fs::is_regular_file(p)&&fs::file_size(p)<=bound,"missing/oversize state input: "+p.string());
  Bytes data(fs::file_size(p));std::ifstream f(p,std::ios::binary);
  if(!data.empty())f.read(reinterpret_cast<char*>(data.data()),data.size());
  require(bool(f),"state input read failed: "+p.string());return data;
}
inline void write_new(const fs::path& p,const void* data,size_t size) {
  int fd=::open(p.c_str(),O_WRONLY|O_CREAT|O_EXCL|O_CLOEXEC,0600);
  require(fd>=0,"new state output required: "+p.string());
  const auto* ptr=static_cast<const uint8_t*>(data);size_t at=0;
  while(at<size) {
    ssize_t n=::write(fd,ptr+at,size-at);
    if(n<=0){::close(fd);throw std::runtime_error("state output write failed");}
    at+=static_cast<size_t>(n);
  }
  const int flushed=::fsync(fd),closed=::close(fd);
  require(!flushed&&!closed,"state output flush/close failed");
}
inline void write_new(const fs::path& p,const std::string& data) {write_new(p,data.data(),data.size());}
inline void write_new(const fs::path& p,const Bytes& data) {write_new(p,data.data(),data.size());}
struct Reference {
  std::string path,sha;
  std::string json() const {return path.empty()?"null":"{\"path\":"+quote(path)+",\"sha256\":"+quote(sha)+"}";}
};
inline Reference reference(const fs::path& p) {return {fs::absolute(p).string(),sha256(read_file(p))};}

struct Entry {
  std::string id,function,partition;
  unsigned diagnostic=0,preparation=0,formal=0;
  unsigned count() const {return diagnostic+preparation+formal;}
};
class Plan {
  Bytes bytes_;
  std::string registry_,plan_;
  std::vector<Entry> entries_;
 public:
  static Plan parse(const void* data,size_t size) {
    require(data&&size>=16&&size<=8192,"bounded measurement-state directory required");
    const auto* begin=static_cast<const uint8_t*>(data);
    for(size_t i=0;i<size;++i)require(begin[i]>=9&&begin[i]<=126&&begin[i]!=13,"measurement-state ASCII differs");
    std::string text(reinterpret_cast<const char*>(data),size);size_t at=0;
    auto line=[&](){auto end=text.find('\n',at);require(end!=std::string::npos,"truncated measurement-state directory");auto v=text.substr(at,end-at);at=end+1;return v;};
    require(line()=="AE-MEASUREMENT-STATE-V1","measurement-state version/magic differs");
    Plan result;result.registry_=line();result.plan_=line();
    require(digest(result.registry_)&&digest(result.plan_),"measurement-state registry/plan SHA required");
    unsigned count=static_cast<unsigned>(decimal(line(),12));require(count>0,"measurement-state case count required");
    std::set<std::string> ids,names;
    for(unsigned i=0;i<count;++i) {
      std::string row=line();std::array<std::string,6> fields;size_t pos=0;
      for(unsigned j=0;j<6;++j) {
        size_t end=row.find('\t',pos);
        require((j==5)==(end==std::string::npos),"measurement-state row field count differs");
        fields[j]=row.substr(pos,end==std::string::npos?end:end-pos);pos=end==std::string::npos?row.size():end+1;
      }
      Entry entry{fields[0],fields[1],fields[2],static_cast<unsigned>(decimal(fields[3],1)),
        static_cast<unsigned>(decimal(fields[4],8)),static_cast<unsigned>(decimal(fields[5],3))};
      require(identifier(entry.id)&&identifier(entry.function,true),"illegal measurement-state name");
      require(ids.insert(entry.id).second&&names.insert(entry.function).second,"duplicate measurement-state case/function");
      require(entry.partition=="fit"||entry.partition=="heldout","unknown measurement partition");
      require(entry.diagnostic==1&&entry.formal==3,"exactly one diagnostic and three formal executions required");
      result.entries_.push_back(std::move(entry));
    }
    const auto body_end=at;auto checksum=line();
    require(digest(checksum)&&checksum==sha256(data,body_end)&&at==size,"measurement-state checksum/trailing bytes differ");
    result.bytes_.assign(begin,begin+size);return result;
  }
  const std::string& registry_sha256() const {return registry_;}
  const std::string& plan_sha256() const {return plan_;}
  std::string directory_sha256() const {return sha256(bytes_);}
  const Bytes& bytes() const {return bytes_;}
  const std::vector<Entry>& entries() const {return entries_;}
  const Entry& find(const std::string& id) const {
    for(const auto& entry:entries_)if(entry.id==id)return entry;
    throw std::runtime_error("unknown measurement-state case");
  }
  void require_compatible(const Plan& other) const {
    require(bytes_==other.bytes_,"complete measurement-state directory differs, including recomputed replacements");
  }
};

struct Program {
  std::string id,function,partition,family;
  unsigned steps=0;
  Bytes input,auxiliary,expected;
};
struct Buffers {Handle input=0,auxiliary=0,output=0;};
struct Slot {
  unsigned sequence=0,role_index=0;
  std::string role,status="not_started",error;
  bool attempted=false,launch_attempted=false,elapsed_called=false,output_checked=false,correct=false;
  uint64_t started_ns=0,ended_ns=0;
  Handle start_event=0,end_event=0;
  float elapsed_ms=0;
  std::map<std::string,int> returncodes;
  Reference started,event,actual_output,original;
  std::string json(bool include_original=true) const {
    uint32_t bits=0;std::memcpy(&bits,&elapsed_ms,sizeof(bits));
    std::ostringstream out;out<<std::setprecision(std::numeric_limits<double>::max_digits10);
    out<<"{\"sequence\":"<<sequence<<",\"role\":"<<quote(role)<<",\"role_index\":"<<role_index
       <<",\"formal_index\":"<<(role=="formal"?std::to_string(role_index):"null")<<",\"status\":"<<quote(status)
       <<",\"attempted\":"<<(attempted?"true":"false")<<",\"launch_attempted\":"<<(launch_attempted?"true":"false")
       <<",\"started_realtime_ns\":"<<(started_ns?std::to_string(started_ns):"null")
       <<",\"ended_realtime_ns\":"<<(ended_ns?std::to_string(ended_ns):"null")
       <<",\"start_event\":"<<(start_event?quote(std::to_string(start_event)):"null")
       <<",\"end_event\":"<<(end_event?quote(std::to_string(end_event)):"null")
       <<",\"event_elapsed_ms_float32_bits\":"<<(elapsed_called?std::to_string(bits):"null")
       <<",\"event_elapsed_ms\":";
    if(elapsed_called&&std::isfinite(elapsed_ms))out<<double(elapsed_ms);else out<<"null";
    out<<",\"elapsed_ns\":";
    if(elapsed_called&&std::isfinite(elapsed_ms))out<<double(elapsed_ms)*1e6;else out<<"null";
    out<<",\"unit\":\"ns\",\"returncodes\":{";bool first=true;
    for(const auto& r:returncodes){out<<(first?"":",")<<quote(r.first)<<':'<<r.second;first=false;}
    out<<"},\"correct\":"<<(output_checked?(correct?"true":"false"):"null")
       <<",\"actual_output\":"<<actual_output.json()<<",\"started\":"<<started.json()<<",\"event\":"<<event.json();
    if(include_original)out<<",\"original\":"<<original.json();
    out<<",\"error\":"<<(error.empty()?"null":quote(error))<<'}';return out.str();
  }
};
inline std::vector<Slot> slots(const Entry& entry) {
  std::vector<Slot> result;
  auto append=[&](const char* role,unsigned count){for(unsigned i=0;i<count;++i){Slot slot;slot.sequence=result.size();slot.role=role;slot.role_index=i;result.push_back(std::move(slot));}};
  append("diagnostic",entry.diagnostic);append("preparation",entry.preparation);append("formal",entry.formal);return result;
}

// Driver methods return raw API codes and never perform file IO. The executor
// owns every call and its order. No adapter chooses roles or adds launches.
struct Driver {
  virtual ~Driver()=default;
  virtual std::string kind() const=0;
  virtual std::string context() const=0;
  virtual uint64_t now_ns()=0;
  virtual int allocate(const std::string&,size_t,Handle&)=0;
  virtual int upload(Handle,const Bytes&)=0;
  virtual int context_synchronize()=0;
  virtual int event_create(Handle&)=0;
  virtual int event_record(Handle)=0;
  virtual int launch(const Program&,const Buffers&,const Slot&)=0;
  virtual int event_synchronize(Handle)=0;
  virtual int event_elapsed(float&,Handle,Handle)=0;
  virtual int download(Handle,Bytes&)=0;
  virtual int event_destroy(Handle)=0;
  virtual int free(Handle)=0;
};
struct Options {std::string run,freeze;uint64_t start_ns=0,deadline_ns=0;};
// These are observations of API arguments and outputs, not an inferred resource
// summary. In particular a failed create/allocate may still return a handle.
struct Resource {
  std::string kind,name;Handle handle=0;size_t bytes=0;std::string sha256;
  std::string json() const {
    return "{\"kind\":"+quote(kind)+",\"name\":"+quote(name)+",\"handle\":"+std::to_string(handle)+
      ",\"bytes\":"+std::to_string(bytes)+",\"sha256\":"+(sha256.empty()?"null":quote(sha256))+"}";
  }
};
using Resources=std::vector<Resource>;
inline Resource allocation(const std::string& name,Handle handle,size_t bytes=0,const std::string& sha={}) {
  return {"allocation",name,handle,bytes,sha};
}
inline Resource event(const std::string& name,Handle handle) {return {"event",name,handle,0,{}};}
struct Trace {
  size_t index=0;int sequence=-1;std::string operation;int return_code=0;
  std::string run,case_id,context;Resources resources;
  std::string json() const {
    std::string out="{\"index\":"+std::to_string(index)+",\"sequence\":"+(sequence<0?"null":std::to_string(sequence))+
      ",\"operation\":"+quote(operation)+",\"return_code\":"+std::to_string(return_code)+
      ",\"scope\":{\"run\":"+quote(run)+",\"case\":"+quote(case_id)+",\"context\":"+quote(context)+"},\"resources\":[";
    for(size_t i=0;i<resources.size();++i)out+=(i?",":"")+resources[i].json();
    return out+"]}";
  }
};
struct Result {
  std::string status="failed",error;
  bool cleanup_closed=false;
  uint64_t ended_ns=0;
  size_t application_bytes=0;
  std::vector<Slot> slots;
  std::vector<Trace> trace;
  std::vector<Handle> remaining_events,remaining_allocations;
  std::string quality="not_evaluated";
  double formal_mean_ns=0,formal_range_over_mean=0;
  unsigned attempted() const {return std::count_if(slots.begin(),slots.end(),[](const Slot& s){return s.attempted;});}
  unsigned launches() const {return std::count_if(slots.begin(),slots.end(),[](const Slot& s){return s.launch_attempted;});}
};

class Execution {
  const Plan& plan_;const Program& program_;Driver& driver_;Options options_;fs::path root_;bool consumed_=false;
  Result result_;std::string context_;
  std::vector<std::pair<std::string,Handle>> allocations_;
  std::vector<std::pair<std::string,Handle>> events_;
  std::string base() const {
    return "\"kind\":"+quote(driver_.kind())+",\"run\":"+quote(options_.run)+",\"case\":"+quote(program_.id)+
      ",\"function\":"+quote(program_.function)+",\"partition\":"+quote(program_.partition)+
      ",\"registry_sha256\":"+quote(plan_.registry_sha256())+",\"plan_sha256\":"+quote(plan_.plan_sha256())+
      ",\"directory_sha256\":"+quote(plan_.directory_sha256())+",\"freeze_sha256\":"+quote(options_.freeze)+
      ",\"context\":"+quote(context_);
  }
  void record(const std::string& op,int sequence,int rc,Resources resources={}) {
    result_.trace.push_back({result_.trace.size(),sequence,op,rc,options_.run,program_.id,context_,std::move(resources)});
  }
  template<class F,class Observe> int call(const std::string& op,Slot* slot,F f,Observe observe) {
    const int rc=f();
    // Observe out-parameters and downloaded bytes after the actual call, even
    // when its raw return code is nonzero. No adapter synthesizes this trace.
    record(op,slot?int(slot->sequence):-1,rc,observe());if(slot)slot->returncodes[op]=rc;return rc;
  }
  template<class F> int call(const std::string& op,Slot* slot,F f) {
    return call(op,slot,f,[]{return Resources{};});
  }
  template<class F,class Observe> void checked(const std::string& op,Slot* slot,F f,Observe observe) {
    const int rc=call(op,slot,f,observe);require(rc==0,op+" returned "+std::to_string(rc));
  }
  template<class F> void checked(const std::string& op,Slot* slot,F f) {
    checked(op,slot,f,[]{return Resources{};});
  }
  void deadline() {require(driver_.now_ns()<options_.deadline_ns,"fixed host/absolute deadline exceeded");}
  std::string prefix(unsigned sequence) const {
    std::ostringstream out;out<<"slot-"<<std::setw(3)<<std::setfill('0')<<sequence;return out.str();
  }
  void save_slot(Slot& slot) {
    auto file=root_/(prefix(slot.sequence)+".json");
    write_new(file,"{\"schema\":\"arch-explorer.measurement-state-slot.v1\","+base()+",\"slot\":"+slot.json(false)+"}\n");
    slot.original=reference(file);record("write_result",slot.sequence,0);
  }
  void cleanup() noexcept {
    bool good=true;
    auto attempt=[&](const std::string& op,auto f,const Resources& resources){
      try {const int rc=call(op,nullptr,f,[&]{return resources;});good=good&&rc==0;return rc==0;}
      catch(...){record(op,-1,-1,resources);good=false;return false;}
    };
    if(!events_.empty()||!allocations_.empty())attempt("cleanup_context_synchronize",[&]{return driver_.context_synchronize();},{});
    for(auto it=events_.rbegin();it!=events_.rend();++it)
      if(!attempt("event_destroy_"+it->first,[&]{return driver_.event_destroy(it->second);},{event(it->first,it->second)}))result_.remaining_events.push_back(it->second);
    for(auto it=allocations_.rbegin();it!=allocations_.rend();++it)
      if(!attempt("free_"+it->first,[&]{return driver_.free(it->second);},{allocation(it->first,it->second)}))result_.remaining_allocations.push_back(it->second);
    result_.cleanup_closed=good&&result_.remaining_events.empty()&&result_.remaining_allocations.empty();
  }
  void save_ledger() {
    unsigned returned=0,failed=0,formal_returned=0;
    for(const auto& slot:result_.slots){returned+=slot.status=="returned";failed+=slot.status=="failed";formal_returned+=slot.role=="formal"&&slot.status=="returned";}
    std::ostringstream out;out<<std::setprecision(std::numeric_limits<double>::max_digits10);
    out<<"{\"schema\":\"arch-explorer.measurement-state-host.v2\","<<base()<<",\"status\":"<<quote(result_.status)
       <<",\"error\":"<<(result_.error.empty()?"null":quote(result_.error))<<",\"start_ns\":"<<options_.start_ns
       <<",\"deadline_ns\":"<<options_.deadline_ns<<",\"ended_ns\":"<<result_.ended_ns<<",\"application_bytes\":"<<result_.application_bytes
       <<",\"slots\":[";
    for(size_t i=0;i<result_.slots.size();++i)out<<(i?",":"")<<result_.slots[i].json();
    out<<"],\"trace\":[";
    for(size_t i=0;i<result_.trace.size();++i)out<<(i?",":"")<<result_.trace[i].json();
    out<<"],\"cleanup\":{\"status\":"<<quote(result_.cleanup_closed?"closed":"failed")
       <<",\"all_resources_released\":"<<(result_.cleanup_closed?"true":"false")<<",\"remaining_events\":[";
    for(size_t i=0;i<result_.remaining_events.size();++i)out<<(i?",":"")<<result_.remaining_events[i];
    out<<"],\"remaining_allocations\":[";
    for(size_t i=0;i<result_.remaining_allocations.size();++i)out<<(i?",":"")<<result_.remaining_allocations[i];
    out<<"]},\"counts\":{\"planned\":"<<result_.slots.size()<<",\"attempted\":"<<result_.attempted()
       <<",\"launch_attempted\":"<<result_.launches()<<",\"returned\":"<<returned<<",\"failed\":"<<failed
       <<",\"not_started\":"<<result_.slots.size()-returned-failed<<",\"formal_planned\":3,\"formal_returned\":"<<formal_returned
       <<"},\"quality\":{\"status\":"<<quote(result_.quality)<<",\"formal_mean_ns\":";
    if(result_.quality!="not_evaluated")out<<result_.formal_mean_ns;else out<<"null";
    out<<",\"formal_range_over_mean\":";
    if(result_.quality!="not_evaluated")out<<result_.formal_range_over_mean;else out<<"null";
    out<<",\"max_sample_range_over_mean\":\"0.03\",\"assumed_event_interval_abs_error_ns\":\"1000\",\"samples_shrink_assumed_bound\":false}"
       <<",\"input\":"<<reference(root_/"input.bin").json()<<",\"auxiliary\":"<<reference(root_/"auxiliary.bin").json()
       <<",\"expected\":"<<reference(root_/"expected.bin").json()<<",\"output_initial\":"<<reference(root_/"output.initial.bin").json()
       <<",\"hardware_core_mhz\":null,\"clock_state\":\"unknown\",\"cache_state\":\"unknown\",\"physical_parameters\":null"
       <<",\"hardware_accuracy\":\"not validated\",\"retry_allowed\":false,\"no_resource_authority\":"<<(driver_.kind()=="synthetic"?"true":"false")<<"}\n";
    const auto data=out.str();require(data.size()<1024*1024,"state ledger byte budget exceeded");write_new(root_/"LEDGER.json",data);
  }
 public:
  Execution(const Plan& plan,const Program& program,Driver& driver,Options options,fs::path root)
      :plan_(plan),program_(program),driver_(driver),options_(std::move(options)),root_(fs::absolute(root)) {
    const auto& entry=plan_.find(program_.id);
    require(entry.function==program_.function&&entry.partition==program_.partition,"compiled program/state order binding differs");
    require(run_identifier(options_.run)&&digest(options_.freeze),"run/freeze identity required");
    require(options_.start_ns>0&&options_.start_ns<options_.deadline_ns&&options_.deadline_ns-options_.start_ns<=120000000000ull,
            "fixed positive host start/deadline within 120s required");
    require(driver_.kind()=="synthetic"||driver_.kind()=="hardware","explicit execution kind required");
    require(!program_.input.empty()&&!program_.expected.empty()&&program_.expected.size()%4==0,"whole numeric input/output required");
    require(program_.family=="dep"||program_.family=="quad"||program_.family=="chase","unsupported complete program family");
    require(program_.steps>0&&program_.steps<=4096,"bounded program steps required");
    if(program_.family!="chase")require(program_.auxiliary.size()==8,"integer scalar pair required");
    require(program_.input.size()+program_.auxiliary.size()+2*program_.expected.size()<=1024*1024,"state application byte budget exceeded");
    require(fs::is_directory(root_),"existing new state output directory required");
    context_=driver_.context();require(!context_.empty(),"actual Driver context required");result_.slots=slots(entry);
  }
  Result run() {
    require(!consumed_,"measurement-state execution cannot be retried");consumed_=true;
    const Bytes initial(program_.expected.size(),0xa5);
    write_new(root_/"input.bin",program_.input);write_new(root_/"auxiliary.bin",program_.auxiliary);
    write_new(root_/"expected.bin",program_.expected);write_new(root_/"output.initial.bin",initial);
    Buffers buffers;Handle start=0,stop=0;
    try {
      deadline();require(driver_.now_ns()>=options_.start_ns,"driver clock predates frozen host start");
      auto alloc=[&](const std::string& name,size_t size,Handle& handle){
        deadline();
        const int rc=call("allocate_"+name,nullptr,[&]{return driver_.allocate(name,size,handle);},
          [&]{return Resources{allocation(name,handle,size)};});
        if(handle){allocations_.push_back({name,handle});result_.application_bytes+=size;}
        require(rc==0&&handle,"allocation failed: "+name);
      };
      alloc("input",program_.input.size(),buffers.input);alloc("output",initial.size(),buffers.output);
      deadline();
      checked("upload_input",nullptr,[&]{return driver_.upload(buffers.input,program_.input);},
        [&]{return Resources{allocation("input",buffers.input,program_.input.size(),sha256(program_.input))};});
      if(program_.family=="chase") {
        alloc("auxiliary",program_.auxiliary.size(),buffers.auxiliary);
        deadline();
        checked("upload_auxiliary",nullptr,[&]{return driver_.upload(buffers.auxiliary,program_.auxiliary);},
          [&]{return Resources{allocation("auxiliary",buffers.auxiliary,program_.auxiliary.size(),sha256(program_.auxiliary))};});
      }
      auto create=[&](const std::string& name,Handle& handle){
        deadline();
        const int rc=call("event_create_"+name,nullptr,[&]{return driver_.event_create(handle);},
          [&]{return Resources{event(name,handle)};});
        if(handle)events_.push_back({name,handle});
        require(rc==0&&handle,"event creation failed: "+name);
      };
      create("start",start);create("stop",stop);
      for(auto& slot:result_.slots) {
        deadline();slot.attempted=true;slot.status="attempted";slot.started_ns=driver_.now_ns();slot.start_event=start;slot.end_event=stop;
        try {
          auto file=root_/(prefix(slot.sequence)+".started.json");
          write_new(file,"{\"schema\":\"arch-explorer.measurement-state-started.v1\","+base()+",\"slot\":"+slot.json(false)+"}\n");
          slot.started=reference(file);record("write_started",slot.sequence,0);
          checked("reset_output",&slot,[&]{return driver_.upload(buffers.output,initial);},
            [&]{return Resources{allocation("output",buffers.output,initial.size(),sha256(initial))};});
          checked("context_synchronize",&slot,[&]{return driver_.context_synchronize();});deadline();
          // Do not insert filesystem IO, copies, queries, synchronization or
          // another kernel between these three consecutive Driver calls.
          checked("event_record_start",&slot,[&]{return driver_.event_record(start);},[&]{return Resources{event("start",start)};});
          slot.launch_attempted=true;
          checked("launch",&slot,[&]{return driver_.launch(program_,buffers,slot);},[&]{
            Resources resources{allocation("input",buffers.input),allocation("output",buffers.output)};
            if(program_.family=="chase")resources.push_back(allocation("auxiliary",buffers.auxiliary));
            return resources;
          });
          checked("event_record_stop",&slot,[&]{return driver_.event_record(stop);},[&]{return Resources{event("stop",stop)};});
          checked("event_synchronize_stop",&slot,[&]{return driver_.event_synchronize(stop);},[&]{return Resources{event("stop",stop)};});
          slot.elapsed_called=true;
          const int elapsed_rc=call("event_elapsed",&slot,[&]{return driver_.event_elapsed(slot.elapsed_ms,start,stop);},
            [&]{return Resources{event("start",start),event("stop",stop)};});
          file=root_/(prefix(slot.sequence)+".event.json");
          write_new(file,"{\"schema\":\"arch-explorer.measurement-state-event.v1\","+base()+",\"slot\":"+slot.json(false)+"}\n");
          slot.event=reference(file);record("write_event",slot.sequence,0);require(elapsed_rc==0,"event_elapsed returned "+std::to_string(elapsed_rc));
          Bytes actual(program_.expected.size());checked("download_output",&slot,[&]{return driver_.download(buffers.output,actual);},
            [&]{return Resources{allocation("output",buffers.output,actual.size(),sha256(actual))};});
          file=root_/(prefix(slot.sequence)+".actual.bin");write_new(file,actual);slot.actual_output=reference(file);record("write_output",slot.sequence,0);
          slot.output_checked=true;slot.correct=actual==program_.expected;
          require(slot.correct,"full output differs; no retry or replacement");
          require(std::isfinite(slot.elapsed_ms)&&slot.elapsed_ms>0,"nonpositive/nonfinite elapsed event; raw float32 bits retained");
          deadline();slot.status="returned";slot.ended_ns=driver_.now_ns();save_slot(slot);
        } catch(const std::exception& e) {
          slot.status="failed";slot.error=e.what();slot.ended_ns=driver_.now_ns();
          if(slot.original.path.empty())save_slot(slot);
          throw;
        }
      }
      std::vector<double> formal;
      for(const auto& slot:result_.slots)if(slot.role=="formal")formal.push_back(double(slot.elapsed_ms)*1e6);
      require(formal.size()==3,"exact formal denominator differs");
      result_.formal_mean_ns=(formal[0]+formal[1]+formal[2])/3;
      result_.formal_range_over_mean=(*std::max_element(formal.begin(),formal.end())-*std::min_element(formal.begin(),formal.end()))/result_.formal_mean_ns;
      // Compare the original float32 dyadics before the ns conversion.
      // Near the threshold their exponents differ by at most one, so these
      // sums/products are exact in double: range/mean <= 3/100 iff
      // 100*(max-min) <= sum. No tolerance widens the registered threshold.
      std::vector<double> milliseconds;
      for(const auto& slot:result_.slots)if(slot.role=="formal")milliseconds.push_back(double(slot.elapsed_ms));
      const auto high=*std::max_element(milliseconds.begin(),milliseconds.end());
      const auto low=*std::min_element(milliseconds.begin(),milliseconds.end());
      result_.quality=100*(high-low)<=milliseconds[0]+milliseconds[1]+milliseconds[2]?"passed":"refused";
      require(result_.quality=="passed","preregistered formal sample quality refused");deadline();result_.status="returned";
    } catch(const std::exception& e){result_.error=e.what();result_.status="failed";}
    cleanup();
    if(!result_.cleanup_closed){result_.status="failed";if(result_.error.empty())result_.error="state resource cleanup incomplete";}
    result_.ended_ns=driver_.now_ns();save_ledger();return result_;
  }
};
} // namespace ae_state
