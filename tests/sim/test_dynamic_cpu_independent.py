"""Compile a second component family using the installed public C++ contract.

Requires a C++17 compiler; absence is not reported as successful acceptance.
The program owns its bounded source/service mechanism and expected timeline.
It neither includes GPU headers nor links a native Accel-Sim library.
"""
import json
import os
from pathlib import Path
import resource
import shutil
import subprocess


def test_two_clock_dynamic_family_retains_physical_and_resource_lifetimes(tmp_path):
    from arch_explorer.simulator.dynamic import runtime_header
    compiler = shutil.which("g++")
    assert compiler, "This C++17 contract test requires g++"
    source = tmp_path / "independent-family.cpp"
    source.write_text(CPP_SOURCE)
    executable = tmp_path / "independent-family"

    def bounds():
        resource.setrlimit(resource.RLIMIT_AS, (2**30, 2**30))
        resource.setrlimit(resource.RLIMIT_CPU, (60, 60))

    env = {key: value for key, value in os.environ.items() if key not in ("LD_PRELOAD", "LD_AUDIT")}
    env["CUDA_VISIBLE_DEVICES"] = ""
    compiled = subprocess.run([compiler, "-std=c++17", "-O0", "-Wall", "-Wextra",
        "-I", str(runtime_header().parent), str(source), "-o", str(executable)],
        cwd=tmp_path, env=env, capture_output=True, text=True, timeout=90, preexec_fn=bounds)
    assert compiled.returncode == 0, compiled.stderr
    process = subprocess.run([str(executable)], cwd=tmp_path, env=env,
        capture_output=True, text=True, timeout=90, preexec_fn=bounds)
    assert process.returncode == 0, process.stderr
    assert json.loads(process.stdout) == {
        "status": "passed", "acceptance": [0, 10], "consumer": [3, 12],
        "physical": [6, 15], "resource": [9, 18], "rejected": 1,
        "wakeups": 1, "completed": 2, "pending": 0, "last_time": 18,
    }


CPP_SOURCE = r'''
#include "runtime.hpp"
#include <deque>
#include <iostream>
#include <stdexcept>
using namespace ae_dynamic;
void check(bool v,const char* message){if(!v)throw std::runtime_error(message);}
struct Producer:Component{
 unsigned issued=0;
 bool can_accept(const Transaction&,const Context&)const override{return false;}
 bool accept(const Transaction&,const Context&)override{return false;}
 bool drained()const override{return issued==2;}
};
struct Service:Component{
 std::deque<Transaction> queue; unsigned stage=0;
 bool can_accept(const Transaction&,const Context&)const override{return queue.empty();}
 bool accept(const Transaction& t,const Context&)override{if(!queue.empty())return false;check(t.address==0x8000+t.id*64 && t.bytes==64,"wire payload differs");queue.push_back(t);stage=0;return true;}
 bool drained()const override{return queue.empty();}
};
int main(){try{
 Runtime r(100);Producer p;Service m;
 r.add_component("producer",p,{{"out",Direction::output}});r.add_component("memory",m,{{"in",Direction::input}});r.connect("producer","out","memory","in");
 r.add_clock("issue",2.0);r.add_clock("service",3.0);bool blocked=false;unsigned wakeups=0;std::vector<double> received,consumer,physical,released;
 r.add_phase("service",0,"memory",[&](const Context& c){if(m.queue.empty())return;const Stage stages[]={Stage::consumer_ready,Stage::physical_complete,Stage::resource_ready};r.mark(m.queue.front().id,"memory",stages[m.stage]);
 if(m.stage==0){consumer.push_back(c.time);check(!r.drained() && r.pending()>0,"consumer mistaken for physical drain");}
 else if(m.stage==1){physical.push_back(c.time);check(!r.drained(),"physical completion released resources early");}
 else{released.push_back(c.time);m.queue.pop_front();r.available("memory","in");}++m.stage;});
 r.add_phase("issue",20,"producer",[&](const Context& c){if(p.issued==2)return;if(blocked){if(!r.take_wakeup("producer"))return;blocked=false;++wakeups;}Transaction t;t.id=p.issued+1;t.source="producer";t.source_port="out";t.destination="memory";t.destination_port="in";t.operation="read";t.bytes=64;t.transfer_bytes=72;t.address=0x8000+t.id*64;t.byte_mask={255};t.ordering="fifo";
 const auto before=m.queue.size();if(r.route(t)){++p.issued;received.push_back(c.time);}else{check(m.queue.size()==before && r.pending()==1 && p.issued==1,"rejection mutated state/ownership");blocked=true;}});
 while(!r.drained())r.tick();
 check(received==std::vector<double>({0,10}),"acceptance time oracle differs");check(consumer==std::vector<double>({3,12}),"consumer time oracle differs");check(physical==std::vector<double>({6,15}),"physical time oracle differs");check(released==std::vector<double>({9,18}),"resource time oracle differs");
 check(r.audit().accepted==2 && r.audit().rejected==1 && r.audit().completed==2 && wakeups==1 && r.pending()==0,"transaction counts differ");
 bool duplicate=false;try{r.mark(1,"memory",Stage::resource_ready);}catch(const std::runtime_error&){duplicate=true;}check(duplicate,"duplicate completion accepted");
 unsigned rejections=0;
 auto rejected=[&](auto operation){bool failed=false;try{operation();}catch(const std::runtime_error&){failed=true;}check(failed,"invalid response/route accepted");++rejections;};
 rejected([&]{r.mark(999,"memory",Stage::consumer_ready);});
 rejected([&]{r.mark(1,"producer",Stage::physical_complete);});
 Transaction bad;bad.id=3;bad.operation="read";bad.source="producer";bad.source_port="out";bad.destination="memory";bad.destination_port="missing";
 rejected([&]{r.route(bad);});bad.id=1;bad.destination_port="in";rejected([&]{r.route(bad);});
 check(r.pending()==0 && r.audit().completed==2,"invalid response changed closure");
 // Missing completion cannot be made successful by an idle component or time.
 Runtime bounded(2);Producer missing;bounded.add_component("missing",missing,{});bounded.add_clock("clock",1);bounded.add_phase("clock",0,"missing",[](const Context&){});
 bounded.tick();bounded.tick();rejected([&]{bounded.tick();});check(!bounded.drained(),"missing completion silently drained");
 check(rejections==5,"rejection coverage differs");
 std::cout<<"{\"status\":\"passed\",\"acceptance\":[0,10],\"consumer\":[3,12],\"physical\":[6,15],\"resource\":[9,18],\"rejected\":1,\"wakeups\":1,\"completed\":2,\"pending\":0,\"last_time\":"<<r.context().time<<"}\n";return 0;
 }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
'''
