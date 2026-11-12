"""Build the component-only image from fixed, local upstream inputs.

Base preparation and extraction are explicit durable stages. The standalone
link never includes gpu-sim.cc, gpgpusim_entrypoint or the old trace frontend.
An independent unchanged oracle manifest is supplied by the caller.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import shutil
import signal
import subprocess
import time

from build import BASES, PATCHES


def sha(path):
    with Path(path).open('rb') as file:
        return hashlib.file_digest(file, 'sha256').hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True)+'\n')


class Commands:
    def __init__(self, root):
        self.root = root
        self.entries = []
        self.stamp = time.time_ns()

    def run(self, argv, cwd, *, env=None, timeout=3600):
        index = len(self.entries)
        entry = {'argv':list(map(str,argv)), 'cwd':str(cwd), 'timeout_s':timeout,
                 'started_monotonic_ns':time.monotonic_ns(), 'started_realtime_ns':time.time_ns()}
        path = self.root/f'command-{self.stamp}-{index:03d}.log'
        def bound():
            resource.setrlimit(resource.RLIMIT_AS, (8*1024**3,8*1024**3))
        with path.open('wb') as output:
            process = subprocess.Popen(entry['argv'],cwd=cwd,env=env,stdout=output,stderr=subprocess.STDOUT,
                                       start_new_session=True,preexec_fn=bound)
            entry['pid'] = process.pid
            try:
                code = process.wait(timeout=timeout)
                entry['timed_out'] = False
            except BaseException:
                os.killpg(process.pid,signal.SIGKILL)
                process.wait()
                entry.update(timed_out=True,exit_code=process.returncode)
                raise
            finally:
                # Compiler/linker children cannot survive their command group.
                try:
                    os.killpg(process.pid,0)
                except ProcessLookupError:
                    entry['group_closed'] = True
                else:
                    os.killpg(process.pid,signal.SIGKILL)
                    entry['group_closed'] = False
                entry.update(ended_monotonic_ns=time.monotonic_ns(),ended_realtime_ns=time.time_ns(),
                             exit_code=process.returncode,wait_completed=True,
                             log={'path':str(path),'sha256':sha(path)})
                self.entries.append(entry)
                write(self.root/f'COMMANDS-{self.stamp}.json',self.entries)
        if code or not entry['group_closed']:
            raise RuntimeError(f'isolated build command failed ({code}); see {path}')
        return entry


def base(args, commands, repo):
    root = args.root
    records = {}
    for name, source in [('accelsim',args.accel_source),('gpgpusim',args.gpgpu_source)]:
        target = root/name
        patch = repo/'tools/accelsim_native/patches'/f'{name}-sm120-typed-v1.patch'
        if sha(patch) != PATCHES[name]:
            raise ValueError('fixed typed patch differs: '+name)
        commands.run(['git','clone','--shared','--no-checkout',source.resolve(),target],root)
        commands.run(['git','checkout','--detach',BASES[name]],target)
        commands.run(['git','apply','--index','--binary',patch],target)
        tree = subprocess.check_output(['git','write-tree'],cwd=target,text=True).strip()
        records[name] = {'commit':BASES[name],'typed_patch':{'path':str(patch),'sha256':PATCHES[name]},'typed_tree':tree}
    write(root/'BASE.json',{'schema':'arch-explorer.standalone-base.v1','sources':records,
                            'commands':commands.entries,'new_tree_only':True})


def extract_components(args, commands, repo):
    from standalone_core import patch_core
    from standalone_memory import patch_memory
    from standalone_shader import patch_shader
    from standalone_icnt import generate as generate_icnt
    from standalone_config import configuration_source,trace_type_source
    root = args.root
    if not (root/'BASE.json').is_file() or (root/'EXTRACTION-STANDALONE.json').exists():
        raise ValueError('a fresh, prepared typed base is required')
    from arch_explorer.components._support.accelsim.native_resources import native_files
    component = native_files()
    include = root/'gpgpusim/src/gpgpu-sim/ae_component'
    include.mkdir()
    for name, path in component.items():
        if path.suffix in {'.hpp', '.h'}:
            shutil.copy2(path,include/name)
    shutil.copy2(args.runtime,include/'runtime.hpp')
    records = {'core':patch_core(root/'gpgpusim'),
               'memory':patch_memory(root/'gpgpusim'),
               'shader':patch_shader(root),
               'icnt':generate_icnt(root/'gpgpusim/src')}
    records['configuration'] = configuration_source(root/'gpgpusim',include/'configuration.cc')
    records['trace_metadata'] = trace_type_source(root/'gpgpusim',include/'trace_types.cc')
    source = root/'component-source'
    source.mkdir()
    for name, path in component.items():
        if path.suffix in {'.cc', '.cpp'}:
            shutil.copy2(path,source/name)
            shutil.copy2(path,include/name)
    from arch_explorer.components._support.accelsim import native_resources
    resources = [*component.values(), Path(native_resources.__file__)]
    resources += [p for p in sorted(Path(__file__).parent.iterdir())
                  if p.is_file() and p.suffix == '.py']
    inputs = {str(p.relative_to(repo)):{'path':str(p),'sha256':sha(p)} for p in resources}
    inputs['runtime.hpp'] = {'path':str(args.runtime),'sha256':sha(args.runtime)}
    write(root/'EXTRACTION-STANDALONE.json',{'schema':'arch-explorer.standalone-extraction.v1',
        'records':records,'inputs':inputs,'source':str(source),'commands':commands.entries,
        'forbidden_link_sources':['gpu-sim.cc','gpgpusim_entrypoint.cc','accel-sim.cc','main.cc']})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['root','accel-source','gpgpu-source','cuda','tools','runtime']:
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--stage',choices=['base','extract','build'],default='build')
    parser.add_argument('--target',choices=['icnt','memory','gpu'],default='gpu')
    args = parser.parse_args()
    args.root = args.root.resolve()
    for name in ['accel_source','gpgpu_source','cuda','tools','runtime']:
        setattr(args,name,getattr(args,name).resolve(strict=True))
    repo = Path(__file__).resolve().parents[4]
    if args.stage == 'base':
        args.root.mkdir(parents=True,exist_ok=False)
    elif not args.root.is_dir():
        parser.error('prepare a new base root first')
    commands = Commands(args.root)
    status = {'stage':args.stage,'started_realtime_ns':time.time_ns(),'jobs':1,'new_GPU_calls':0}
    try:
        if args.stage == 'base':
            base(args,commands,repo)
        elif args.stage == 'extract':
            extract_components(args,commands,repo)
        else:
            from standalone_compile import compile_components
            compile_components(args,commands,repo)
        status['status'] = 'completed'
    except BaseException as error:
        status.update(status='failed',error={'type':type(error).__name__,'message':str(error)})
        raise
    finally:
        status.update(ended_realtime_ns=time.time_ns(),commands=commands.entries)
        write(args.root/f'STAGE-{args.stage}-{commands.stamp}.json',status)
    print(args.root)


if __name__ == '__main__':
    main()
