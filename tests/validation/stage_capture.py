"""Bounded subprocess observations and assertion-safe per-stage capture.

Linux wait4 maxrss is the largest process high-water mark, not aggregate RSS.
The sampled live process-group RSS is separately labelled and bounded. No
stage name implies construction, compilation, service or numerical success.
"""
import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import resource
import signal
import subprocess
import time

# A failed observation must not silently release a scale slot in a live caller.
# Retain the descriptor and child handle for explicit caller recovery.
_unconfirmed_groups = {}

def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


@contextmanager
def stage(path, name, **identity):
    """Callers append counters/outputs before assertions; finally always saves."""
    start = time.monotonic()
    value = dict(schema='arch-explorer.stage-observation.v1', stage=name, identity=identity,
                 started_at=datetime.now(timezone.utc).isoformat(), status='running',
                 observations={})
    save(Path(path), value)
    try:
        yield value['observations']
    except BaseException as error:
        value.update(status='failed', error_type=type(error).__name__, error=str(error))
        raise
    else:
        value['status'] = 'succeeded'
    finally:
        value.update(wall_seconds=time.monotonic()-start,
                     process_high_water_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
        save(Path(path), value)


def group_snapshot(pgid):
    members = []
    errors = []
    for entry in Path('/proc').iterdir():
        if not entry.name.isdigit(): continue
        try:
            raw = (entry/'stat').read_text()
            fields = raw[raw.rindex(')')+2:].split()
            if int(fields[2]) == pgid:
                members.append(dict(pid=int(entry.name), state=fields[0],
                    start_ticks=int(fields[19]), rss_bytes=int(fields[21])*os.sysconf('SC_PAGE_SIZE')))
        except (FileNotFoundError, ProcessLookupError): pass
        except (OSError, ValueError, IndexError) as error:
            errors.append(dict(pid=int(entry.name), error_type=type(error).__name__))
    return dict(pgid=pgid, members=sorted(members,key=lambda row:row['pid']), errors=errors)


def group_rss(pgid):
    return sum(row['rss_bytes'] for row in group_snapshot(pgid)['members'])


def close_group(pgid, *, seconds=2):
    """Boundedly observe no executing group members; zombies stay labelled."""
    start=time.monotonic()
    before=group_snapshot(pgid)
    sent=False
    try:
        os.killpg(pgid,signal.SIGKILL)
        sent=True
    except ProcessLookupError: pass
    while True:
        after=group_snapshot(pgid)
        active=[row for row in after['members'] if row['state'] not in ('Z','X')]
        confirmed=not active and not after['errors']
        if confirmed or time.monotonic()-start>=seconds: break
        time.sleep(.01)
    return dict(status='confirmed-no-running-members' if confirmed else 'unconfirmed',
                confirmed=confirmed, signal='SIGKILL', signal_sent=sent,
                before=before, after=after, remaining_running=active,
                terminal_members=[row for row in after['members'] if row['state'] in ('Z','X')],
                elapsed_seconds=time.monotonic()-start,
                scope='same process group; zombie descendants are not claimed reaped')


def run(command, *, cwd, dest, seconds=120, gib=8, scale=False, scale_lock=None):
    if not command or seconds<=0 or seconds>(600 if scale else 120) or gib<=0 or gib>(16 if scale else 8):
        raise ValueError('probe exceeds explicit contract budget')
    dest = Path(dest); dest.mkdir(parents=True,exist_ok=False)
    budget = int(gib*1024**3)
    env={k:v for k,v in os.environ.items() if k not in ('PYTHONHOME',)}
    env.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',
               NUMEXPR_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1',PYTEST_DISABLE_PLUGIN_AUTOLOAD='1')
    record=dict(schema='arch-explorer.bounded-process.v1',argv=list(command),cwd=str(Path(cwd).resolve()),
                seconds=seconds,address_space_bytes=budget,group_rss_limit_bytes=budget,
                environment={k:env[k] for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS')},
                scale_lock=None,scale_lock_acquired=False,scale_lock_released=False,
                status='starting',returncode=None,
                started_at=datetime.now(timezone.utc).isoformat(),observed_group_peak_rss_bytes=0)
    start=time.monotonic();child=None;lock=None;usage=None;code=None
    def terminate():
        if child is not None:
            try: os.killpg(child.pid,signal.SIGKILL)
            except ProcessLookupError: pass
    def limit():
        resource.setrlimit(resource.RLIMIT_AS,(budget,budget))
        resource.setrlimit(resource.RLIMIT_FSIZE,(256*1024**2,256*1024**2))
    try:
        if scale:
            if scale_lock is None:
                record.update(status='rejected',reason='missing-scale-lock')
                raise ValueError('scale=True requires an explicit absolute scale_lock path')
            selected_lock=os.fspath(scale_lock)
            if not isinstance(selected_lock,str) or not selected_lock or not Path(selected_lock).is_absolute():
                record.update(status='rejected',reason='scale-lock-must-be-explicit-absolute-path')
                raise ValueError('scale=True requires an explicit absolute scale_lock path')
            record['scale_lock']=selected_lock
            lock=open(selected_lock,'a')
            try:
                fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
                record['scale_lock_acquired']=True
            except BlockingIOError:
                record.update(status='busy',reason='scale-lock-held');return record
        with (dest/'stdout.log').open('wb') as out,(dest/'stderr.log').open('wb') as err:
            child=subprocess.Popen(command,cwd=cwd,env=env,stdout=out,stderr=err,
                                   start_new_session=True,preexec_fn=limit)
            record['pid']=child.pid
            while True:
                pid,status,ru=os.wait4(child.pid,os.WNOHANG)
                if pid:
                    code=os.waitstatus_to_exitcode(status);usage=ru;child.returncode=code;break
                rss=group_rss(child.pid)
                record['observed_group_peak_rss_bytes']=max(rss,record['observed_group_peak_rss_bytes'])
                if time.monotonic()-start>seconds:
                    record['status']='timeout';terminate()
                elif rss>budget:
                    record['status']='rss-budget-exceeded';terminate()
                time.sleep(.02)
        if record['status']=='starting':record['status']='succeeded' if code==0 else 'failed'
        return record
    except BaseException as error:
        record.update(status=record['status'] if record['status']=='rejected' else
                      ('cancelled' if isinstance(error,KeyboardInterrupt) else 'runner-error'),
                      error_type=type(error).__name__,error=str(error))
        raise
    finally:
        cleanup=dict(status='not-started',confirmed=False)
        if child is not None:
            try:
                cleanup=close_group(child.pid)
            except OSError as error:
                cleanup=dict(status='observation-error',confirmed=False,
                             error_type=type(error).__name__,error=str(error))
        if child is not None and child.returncode is None and cleanup['confirmed']:
            _,status,usage=os.wait4(child.pid,0);code=os.waitstatus_to_exitcode(status);child.returncode=code
        parent_reaped=child is not None and child.returncode is not None
        closed=parent_reaped and cleanup['confirmed']
        if child is not None and not closed:
            record['status']='cleanup-unconfirmed'
            _unconfirmed_groups[child.pid]=dict(child=child,lock=lock,cleanup=cleanup)
        elif lock is not None:
            lock.close()
            record['scale_lock_released']=record['scale_lock_acquired']
        record.update(returncode=code,wall_seconds=time.monotonic()-start,
                      parent_reaped=parent_reaped,process_group_closed=closed,process_group_cleanup=cleanup,
                      cleanup_lock_retained=child is not None and not closed and record['scale_lock_acquired'],
                      max_process_high_water_rss_bytes=None if usage is None else usage.ru_maxrss*1024,
                      user_cpu_seconds=None if usage is None else usage.ru_utime,
                      system_cpu_seconds=None if usage is None else usage.ru_stime,
                      runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
        record['artifacts']={str(p.relative_to(dest)):dict(bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest())
                             for p in sorted(dest.rglob('*')) if p.is_file() and p.name!='PROCESS.json'}
        save(dest/'PROCESS.json',record)
        if child is not None and not closed:
            raise RuntimeError('process group cleanup unconfirmed; see PROCESS.json and retained runner handles')


def main():
    p=argparse.ArgumentParser();p.add_argument('--cwd',required=True);p.add_argument('--dest',required=True)
    p.add_argument('--seconds',type=float,default=120);p.add_argument('--gib',type=float,default=8)
    p.add_argument('--scale',action='store_true');p.add_argument('--scale-lock')
    p.add_argument('command',nargs=argparse.REMAINDER)
    a=p.parse_args();cmd=a.command[1:] if a.command[:1]==['--'] else a.command
    try:
        r=run(cmd,cwd=a.cwd,dest=a.dest,seconds=a.seconds,gib=a.gib,scale=a.scale,scale_lock=a.scale_lock)
    except ValueError as error:
        p.error(str(error))
    print(json.dumps(r));return 0 if r['status']=='succeeded' else 1


if __name__=='__main__':raise SystemExit(main())
