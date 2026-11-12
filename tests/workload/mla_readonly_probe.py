"""Standalone fresh-process candidate read-only check with execution forbidden.

Usage: python mla_readonly_probe.py saved-result.json output-receipt.json
Both files live outside the repository during installed acceptance. This tool
does not manufacture or normalize saved sources, environment or product hashes.
"""
import ctypes
import builtins
from contextlib import ExitStack
import hashlib
import io
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
from unittest.mock import patch


def main():
    sys.dont_write_bytecode = True
    from arch_explorer.simulator import api
    # Initialize registry before importing and patching its selected provider.
    from arch_explorer.components.extensions import get_provider
    get_provider('events')
    from arch_explorer.components.events_provider import EventsProvider
    from arch_explorer.simulator.events import backend,engine
    from arch_explorer import search
    from arch_explorer.workload import api as workload_api, compilation
    from arch_explorer.simulator import compilation as simulator_compilation
    original = Path(sys.argv[1]);destination = Path(sys.argv[2])
    raw = original.read_bytes();saved = json.loads(raw)
    called = []
    rebuilds = []
    def forbidden(*args,**kwargs):
        called.append(True)
        raise AssertionError('fresh readonly attempted prepare/engine/solver/native/network/file mutation')
    validation_original = compilation.validate_compilation
    def observe_validation(document,*args,**options):
        # Patch only the simulator's imported reference. The original workload
        # validator/compiler and all selected source identities stay intact.
        # A global profiler changes locals() behavior on CPython 3.10.
        record=dict(module=validation_original.__module__,function=validation_original.__qualname__)
        rebuilds.append(record)
        answer=validation_original(document,*args,**options)
        record.update(schema=answer.get('schema'),binding=answer.get('binding'))
        return answer
    open_original,io_original,os_original = builtins.open,io.open,os.open
    def readonly_open(delegate):
        def guarded(file,mode='r',*args,**kwargs):
            if any(flag in mode for flag in 'wax+'):
                return forbidden(file,mode)
            return delegate(file,mode,*args,**kwargs)
        return guarded
    def readonly_os_open(path,flags,*args,**kwargs):
        if flags & (os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC|os.O_APPEND):
            return forbidden(path,flags)
        return os_original(path,flags,*args,**kwargs)
    with ExitStack() as guards:
        for name in ('Popen','run','check_call','check_output'):
            guards.enter_context(patch.object(subprocess,name,forbidden))
        for name in ('CDLL','PyDLL'):guards.enter_context(patch.object(ctypes,name,forbidden))
        for module,name in ((socket,'socket'),(engine,'simulate'),
                            (engine.ConsistentScheduler,'run'),(backend,'simulate'),
                            (EventsProvider,'_execute'),(EventsProvider,'prepare_request'),
                            (search,'run_search'),(workload_api,'layout_candidates'),
                            (api,'prepare_compiled'),(api,'prepare_request')):
            guards.enter_context(patch.object(module,name,forbidden))
        guards.enter_context(patch.object(simulator_compilation,'validate_compilation',observe_validation))
        guards.enter_context(patch.object(builtins,'open',readonly_open(open_original)))
        guards.enter_context(patch.object(io,'open',readonly_open(io_original)))
        guards.enter_context(patch.object(os,'open',readonly_os_open))
        for name in ('mkdir','makedirs','remove','unlink','rmdir','rename','replace','chmod','utime','truncate'):
            guards.enter_context(patch.object(os,name,forbidden))
        result = api.evaluate_request(saved)
    assert result == saved
    assert not called
    assert original.read_bytes() == raw
    destination.write_text(json.dumps(dict(schema='arch-explorer.val61.fresh-readonly.v1',
        result_sha256=hashlib.sha256(raw).hexdigest(),result_path=str(original.resolve()),
        execution_forbidden=True,unchanged=True,fresh_process=True,
        pure_compilation_validations=rebuilds,pure_compilation_validation_count=len(rebuilds),
        pure_reconstruction_allowed=True,global_profiler_used=False,
        prepare_engine_solver_native_writes_forbidden=True,
        executable=sys.executable),indent=2)+'\n')


if __name__ == '__main__':
    main()
