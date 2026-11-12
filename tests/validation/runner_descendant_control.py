"""Independent Linux same-group descendant control; no model or real scale lock.

Loads one exact runner source file. CORE's lock selector is overridden only to
an isolated control path, with that override explicitly recorded.
"""
import argparse
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import sys
import time
from unittest.mock import patch


def group_members(pgid):
    members=[]
    for path in Path('/proc').glob('[0-9]*/stat'):
        try:
            raw=path.read_text();words=raw[raw.rindex(')')+2:].split()
            if int(words[2])==pgid:
                members.append(dict(pid=int(path.parent.name),state=words[0],start_ticks=int(words[19])))
        except (FileNotFoundError,ProcessLookupError): pass
    return sorted(members,key=lambda row:row['pid'])


def exercise(module, kind, output, *, mode='exit'):
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=False)
    lock=output/'temporary.lock';ready=output/'descendant.json'
    descendant=('import os,json,time;from pathlib import Path;Path('+repr(str(ready)+'.tmp')+')'
                '.write_text(json.dumps(dict(pid=os.getpid(),pgid=os.getpgrp())));'
                'os.replace('+repr(str(ready)+'.tmp')+','+repr(str(ready))+');time.sleep(20)')
    parent=('import subprocess,sys,time;from pathlib import Path;'
            'subprocess.Popen([sys.executable,"-c",'+repr(descendant)+']);'
            'ready=Path('+repr(str(ready))+');deadline=time.monotonic()+.7\n'
            'while not ready.exists() and time.monotonic()<deadline:time.sleep(.005)\n'
            'assert ready.exists(),"descendant did not become ready"\n'+
            ('time.sleep(20)' if mode=='timeout' else 'raise SystemExit('+('7' if mode=='failure' else '0')+')'))
    command=[sys.executable,'-c',parent]
    facts=dict(schema='arch-explorer.runner-descendant-control.v1',kind=kind,mode=mode,
               source=str(Path(module.__file__).resolve()),
               source_sha256=hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest(),
               temporary_lock=str(lock),real_scale_lock_touched=False,model_or_simulation_executions=0,
               scope='same-group descendants; zombies are terminal, not claimed reaped',passed=False)
    identity=None
    try:
        if kind=='qa':
            result=module.run(command,cwd=output,dest=output/'stage',seconds=1,gib=.125,
                              scale=True,scale_lock=lock)
        elif kind=='core':
            facts['lock_selector_override']='scale_lock_path -> isolated temporary control path'
            with patch.object(module,'scale_lock_path',return_value=str(lock)):
                result=module.run_scale_stage(command,cwd=output,output_dir=output/'stage',
                    wall_seconds=1,memory_bytes=128*1024**2,stage='reference',
                    dimensions={'kind':'same-group-process-cleanup-control'},
                    estimate={'kind':'two tiny Python processes; no workload'})
        else:raise ValueError('unknown runner kind')
        facts['runner_result']=result
        identity=json.loads(ready.read_text());facts['descendant']=identity
        assert identity['pgid']==result['pid'],'descendant escaped intended control group'
        with lock.open('a') as contender:
            try:
                fcntl.flock(contender,fcntl.LOCK_EX|fcntl.LOCK_NB)
                facts['lock_available_after_return']=True
            except BlockingIOError:
                facts['lock_available_after_return']=False
            members=group_members(identity['pgid'])
            facts['members_after_return']=members
            active=[row for row in members if row['state'] not in ('Z','X')]
            facts['running_after_return']=active
            facts['lock_free_with_running_members']=facts['lock_available_after_return'] and bool(active)
        facts['parent_returncode']=result['returncode']
        expected=result['returncode']<0 if mode=='timeout' else result['returncode']==(7 if mode=='failure' else 0)
        facts['passed']=expected and not active and facts['lock_available_after_return']
    except BaseException as error:
        facts.update(error_type=type(error).__name__,error=str(error))
        raise
    finally:
        if identity is None and ready.exists():identity=json.loads(ready.read_text())
        if identity is not None:
            if identity['pgid']==os.getpgrp():raise AssertionError('unsafe control group identity')
            try:os.killpg(identity['pgid'],signal.SIGKILL)
            except ProcessLookupError:pass
            until=time.monotonic()+2
            while time.monotonic()<until:
                active=[row for row in group_members(identity['pgid']) if row['state'] not in ('Z','X')]
                if not active:break
                time.sleep(.01)
            facts['probe_cleanup_running_remaining']=active
        (output/'CHECK.json').write_text(json.dumps(facts,indent=2,allow_nan=False)+'\n')
    return facts


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--kind',choices=('qa','core'),required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--mode',choices=('exit','failure','timeout'),default='exit')
    args=parser.parse_args()
    spec=importlib.util.spec_from_file_location('controlled_runner_source',args.source.resolve())
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    result=exercise(module,args.kind,args.output,mode=args.mode)
    print(json.dumps({key:result[key] for key in ('kind','mode','source_sha256','parent_returncode','passed',
        'lock_available_after_return','lock_free_with_running_members')}))
    return 0 if result['passed'] else 1


if __name__=='__main__':raise SystemExit(main())
