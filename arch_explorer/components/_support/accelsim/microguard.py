"""Absolute CPU supervision around an explicitly coordinated device worker.

Import performs no process/device/lock operations. Only ``run`` accepts a fresh
handoff. ``selfcheck`` supervises CPU Python children and never opens a GPU lock.
The source, deadline and worker command are frozen before device work begins.
"""
import argparse,ctypes,fcntl,json,os,signal,subprocess,sys,time,uuid
from pathlib import Path
from arch_explorer.components.accelsim.microexperiment import ref,write_new,validate_design
from arch_explorer.components.accelsim.microrecords import original
from arch_explorer.components.accelsim.microbundle import json_file,load_registry
from arch_explorer.components.accelsim.microcampaign import check_installation,check_handoff
from arch_explorer.components.accelsim.microcontract import modern,work_item


def processes():
    table={}
    for p in Path('/proc').iterdir():
        if not p.name.isdecimal():continue
        try:
            fields=(p/'stat').read_text().rsplit(')',1)[1].split()
            table[int(p.name)]=(int(fields[1]),int(fields[19]))
        except (FileNotFoundError,ProcessLookupError,PermissionError):pass
    return table


def descendants(table,root):
    selected={root}
    while True:
        more={pid for pid,(parent,_) in table.items() if parent in selected}-selected
        if not more:break
        selected|=more
    return {pid:table[pid][1] for pid in selected if pid!=root and pid in table}


def kill_owned(owned):
    table=processes()
    for pid,tick in owned.items():
        if pid in table and table[pid][1]==tick:
            try:os.kill(pid,signal.SIGKILL)
            except ProcessLookupError:pass


def reap():
    while True:
        try:
            pid,status=os.waitpid(-1,os.WNOHANG)
            if not pid:break
        except ChildProcessError:break


def supervise(command,root,hard_ns,*,work,run_id,design=None,handoff=None,lock=None,reserve_s=60,kind='hardware-worker'):
    root=Path(root).resolve();root.mkdir(parents=True,exist_ok=False)
    now=time.time_ns()
    if type(hard_ns) is not int or type(reserve_s) is not int or not 1<=reserve_s<=120:
        raise ValueError('integer absolute cutoff and bounded cleanup reserve required; no worker started')
    remaining=hard_ns-now
    if remaining<=max(3,reserve_s+1)*10**9 or remaining>1800*10**9:
        raise ValueError('absolute cutoff lacks cleanup margin or exceeds frozen window; no worker started')
    if ctypes.CDLL(None).prctl(36,1,0,0,0)!=0:raise RuntimeError('own-descendant subreaper unavailable; no worker started')
    soft_ns=hard_ns-reserve_s*10**9;force_ns=hard_ns-max(1,reserve_s//2)*10**9
    mono_hard=time.monotonic_ns()+remaining
    start=write_new(root/'START.json',{'schema':'arch-explorer.measurement-absolute-supervisor.v2',
         'kind':kind,'work_item':work,'run_id':run_id,'design':design,'handoff':handoff,
         'supervisor_pid':os.getpid(),'source_module':ref(__file__),'command':command,
         'started_realtime_ns':now,'absolute_deadline_realtime_ns':hard_ns,
         'soft_stop_realtime_ns':soft_ns,'forced_cleanup_realtime_ns':force_ns,'cleanup_reserve_s':reserve_s})
    owned={};sent=False;forced=False;error=None;code=None;p=None
    with (root/'worker.stdout').open('xb') as out,(root/'worker.stderr').open('xb') as err:
        try:
            p=subprocess.Popen(command,stdout=out,stderr=err,env={'PATH':'/usr/bin:/bin','LC_ALL':'C','CUDA_VISIBLE_DEVICES':''},start_new_session=True)
            write_new(root/'WORKER.json',{'pid':p.pid,'parent_pid':os.getpid(),'command':command,'started_realtime_ns':time.time_ns()})
            while p.poll() is None:
                owned.update(descendants(processes(),os.getpid()))
                left=min(hard_ns-time.time_ns(),mono_hard-time.monotonic_ns())
                if not sent and left<=reserve_s*10**9:
                    try:os.kill(p.pid,signal.SIGALRM)
                    except ProcessLookupError:pass
                    sent=True
                if left<=max(1,reserve_s//2)*10**9:forced=True;kill_owned(owned);break
                time.sleep(.1)
        except BaseException as e:
            error=type(e).__name__+': '+str(e);forced=True
        finally:
            owned.update(descendants(processes(),os.getpid()))
            if p is not None:
                if p.poll() is None:kill_owned(owned)
                try:code=p.wait(timeout=max(.01,min(10,(hard_ns-time.time_ns())/1e9)))
                except subprocess.TimeoutExpired:
                    error='owned worker did not exit during reserved cleanup';forced=True
    # Refresh after worker exit: subreaper adoption also captures descendants
    # born between samples; only our tree with matching start ticks is signalled.
    owned.update(descendants(processes(),os.getpid()));kill_owned(owned)
    until=min(time.monotonic()+10,time.monotonic()+max(0,(hard_ns-time.time_ns())/1e9))
    while True:
        reap();table=processes();owned.update(descendants(table,os.getpid()))
        active={pid:s for pid,s in owned.items() if pid in table and table[pid][1]==s}
        if not active or time.monotonic()>=until:break
        kill_owned(active);time.sleep(.05)
    reap();table=processes();active={pid:s for pid,s in owned.items() if pid in table and table[pid][1]==s}
    free=None
    if lock is not None and not active:
        fd=os.open(lock,os.O_RDWR)
        try:
            try:fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB);free=True;fcntl.flock(fd,fcntl.LOCK_UN)
            except BlockingIOError:free=False
        finally:os.close(fd)
    closed={'schema':'arch-explorer.measurement-absolute-supervisor-close.v2','kind':kind,'work_item':work,'run_id':run_id,
            'start':start,'exit_code':code,'error':error,'signal_soft_stop_sent':sent,'forced_cleanup':forced,
            'owned_processes':owned,'remaining_owned_processes':active,'all_owned_processes_reaped':not active,
            'common_lock_rechecked_free':free,'ended_realtime_ns':time.time_ns(),
            'absolute_deadline_realtime_ns':hard_ns,'closed_before_absolute_deadline':time.time_ns()<hard_ns}
    write_new(root/'CLOSE.json',closed)
    if active or (lock is not None and free is not True) or not closed['closed_before_absolute_deadline']:
        raise RuntimeError('own resource closure failed; no queue continuation authorization')
    return closed


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);s=p.add_subparsers(dest='command',required=True)
    c=s.add_parser('selfcheck');c.add_argument('--root',required=True)
    r=s.add_parser('run')
    for name in ('design','handoff','window','root'):r.add_argument('--'+name,required=True)
    a=p.parse_args(argv)
    if a.command=='selfcheck':
        child='import time;time.sleep(60)'
        code='import subprocess,signal,sys,time;subprocess.Popen([sys.executable,"-c",'+repr(child)+']);signal.signal(signal.SIGALRM,lambda *a:sys.exit(42));time.sleep(60)'
        result=supervise([sys.executable,'-c',code],a.root,time.time_ns()+5*10**9,
                         work='CPU-SELF-CHECK',run_id=uuid.uuid4().hex,reserve_s=2,kind='synthetic')
        if not result['signal_soft_stop_sent'] or not result['all_owned_processes_reaped'] or result['common_lock_rechecked_free'] is not None:
            raise AssertionError('CPU descendant deadline/cleanup check failed')
    else:
        design=json_file(a.design)[0];bundle=load_registry(original(design['registry']))
        validate_design(bundle,design);check_installation(design)
        if not modern(bundle):raise ValueError('new supervisor requires explicit work-item protocol, not an old window')
        handoff=check_handoff(design,ref(a.handoff))
        if handoff['absolute_deadline_realtime_ns']-time.time_ns() < 90*10**9:raise TimeoutError('insufficient absolute start margin; no worker started')
        result=supervise([sys.executable,'-m','arch_explorer.components.accelsim.microworker',
                          str(Path(a.design).resolve()),str(Path(a.handoff).resolve()),str(Path(a.window).resolve())],
                         a.root,handoff['absolute_deadline_realtime_ns'],work=work_item(bundle),run_id=design['run_id'],
                         design=ref(a.design),handoff=ref(a.handoff),lock=handoff['lock_file'],reserve_s=handoff['close_reserve_s'])
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
