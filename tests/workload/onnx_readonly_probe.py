"""Fresh installed replay with pure rebuilding allowed and side effects trapped."""
from contextlib import ExitStack
import builtins,ctypes,hashlib,io,json,os,re,socket,subprocess,sys
from functools import wraps
from pathlib import Path
from unittest.mock import patch

def main():
    sys.dont_write_bytecode=True
    import numpy,onnx
    import arch_explorer
    from arch_explorer.components.extensions import get_provider
    get_provider('events')
    from arch_explorer.components.events_provider import EventsProvider
    from arch_explorer.components import events_provider
    from arch_explorer.simulator.events import backend,engine
    from arch_explorer.simulator import api,compilation as sim_compilation
    from arch_explorer.workload import api as w,compilation
    from arch_explorer.exploration import replay
    from arch_explorer import search
    assert 'site-packages' in arch_explorer.__file__
    source=Path(sys.argv[1]);destination=Path(sys.argv[2]);raw=source.read_bytes();saved=json.loads(raw)
    calls=[];validations=[];positive=[]
    expected_error=sys.argv[3] if len(sys.argv)>3 else None
    semantic_error=None
    pure_counts=dict(validation_compile=0,taskgraph_reconstruction=0,provider_plan_reconstruction=0,reference=0,
                     onnx_checker=0,onnx_shape_inference=0,graph_reconstruction=0)
    def counted(original,key):
        @wraps(original)
        def observed(*args,**kwargs):
            pure_counts[key]+=1
            return original(*args,**kwargs)
        return observed
    def forbidden(*args,**kwargs):
        calls.append(True);raise AssertionError('VAL62 readonly attempted prepare/engine/solver/native/network/write')
    original=compilation.validate_compilation
    def validate(value,*args,**kwargs):
        validations.append(dict(module=original.__module__,function=original.__qualname__))
        return original(value,*args,**kwargs)
    open0,io0,os0=builtins.open,io.open,os.open
    def read_open(delegate):
        def guarded(file,mode='r',*args,**kwargs):
            if any(c in mode for c in 'wax+'):return forbidden()
            return delegate(file,mode,*args,**kwargs)
        return guarded
    def os_open(file,flags,*args,**kwargs):
        if flags&(os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC|os.O_APPEND):return forbidden()
        return os0(file,flags,*args,**kwargs)
    with ExitStack() as guards:
        targets=[(subprocess,n) for n in ('Popen','run','check_call','check_output')]
        targets += [(ctypes,n) for n in ('CDLL','PyDLL')]
        targets += [(socket,'socket'),(engine,'simulate'),(engine.ConsistentScheduler,'run'),
            (engine,'allocate'),(engine,'_advance'),(backend,'run_mapping'),(backend,'evaluate_events'),
            (backend,'simulate'),(EventsProvider,'_execute'),(EventsProvider,'prepare_request'),
            (api,'prepare_compiled'),(api,'prepare_request'),(sim_compilation,'prepare_compiled'),
            (search,'run_search'),(w,'layout_candidates')]
        for module,name in targets:guards.enter_context(patch.object(module,name,forbidden))
        guards.enter_context(patch.object(sim_compilation,'validate_compilation',validate))
        for module,name,key in ((compilation,'compile_workload','validation_compile'),
            (backend,'prepare_mapping','taskgraph_reconstruction'),
            (events_provider,'_prepare','provider_plan_reconstruction'),
            (w,'execute_reference','reference'),(w,'workload_from_spec','graph_reconstruction'),
            (onnx.checker,'check_model','onnx_checker'),
            (onnx.shape_inference,'infer_shapes','onnx_shape_inference')):
            guards.enter_context(patch.object(module,name,counted(getattr(module,name),key)))
        guards.enter_context(patch.object(builtins,'open',read_open(open0)))
        guards.enter_context(patch.object(io,'open',read_open(io0)))
        guards.enter_context(patch.object(os,'open',os_open))
        for name in ('mkdir','makedirs','remove','unlink','rmdir','rename','replace','chmod','utime','truncate'):
            guards.enter_context(patch.object(os,name,forbidden))
        # Harmless positive sentinels prove each installed trap works, then reset.
        for module,name in targets:
            try:getattr(module,name)()
            except AssertionError:positive.append(module.__name__+'.'+name)
            else:raise AssertionError('inactive guard '+name)
        for mode in ('w','a','x','r+'):
            try:builtins.open(source,mode)
            except AssertionError:positive.append('open:'+mode)
            else:raise AssertionError('inactive write guard')
        calls.clear()
        try:
            result=(replay(saved) if saved['schema']=='archexplorer.compilation-search-result.v1' else api.evaluate_request(saved))
        except ValueError as error:
            if expected_error is None:raise
            assert re.search(expected_error,str(error)),str(error)
            semantic_error=dict(type='ValueError',diagnostic=str(error),expected_pattern=expected_error)
        else:
            assert expected_error is None,'Digest-valid semantic mutation was accepted'
            assert result==saved
    assert not calls and source.read_bytes()==raw
    if expected_error is None:
        assert all(pure_counts[key]>0 for key in ('validation_compile','taskgraph_reconstruction','provider_plan_reconstruction')),pure_counts
    else:assert semantic_error is not None
    destination.write_text(json.dumps(dict(schema='arch-explorer.val63.fresh.v1',
        executable=sys.executable,product_file=arch_explorer.__file__,source_sha256=hashlib.sha256(raw).hexdigest(),
        unchanged=True,fresh=True,positive_controls=positive,pure_validations=validations,pure_counts=pure_counts,
        semantic_error=semantic_error,forbidden_calls=len(calls),
        pure_reconstruction_allowed=True,global_profiler_used=False,
        prepare_engine_solver_native_network_writes_forbidden=True),indent=2)+'\n')

if __name__=='__main__':main()
