# Concrete Accel-Sim implementation; owned by the component provider.
"""Semantic replay of resource-bearing C++ API observations; no execution.

Objects are acquired from actual create outputs, including nonzero handles on
errors. Every use and cleanup must reference those same case/context objects.
Claims about remaining resources are compared only after this reconstruction.
"""
import math,re,struct
from dataclasses import dataclass
from arch_explorer.components.accelsim.trace import exact,digest

HOST='arch-explorer.measurement-state-host.v2'

@dataclass
class Object:
    kind: str
    name: str
    handle: int
    size: int
    created_at: int
    create_return_code: int
    live: bool = True
    uploaded: bool = False
    generation: int = -1
    synchronized: bool = False

def _require(test,message):
    if not test:raise ValueError('resource lifecycle: '+message)

def validate_lifecycle(bundle,cid,ledger):
    """Return derived ownership/cleanup; never use reported remaining as oracle."""
    _require(ledger.get('schema')==HOST,'v2 resource-bearing transcript required; old SIM33 evidence is insufficient')
    case=bundle.cases[cid];main,aux,expected=bundle.inputs(case)
    sizes={'input':len(main),'output':len(expected)}
    if case['family']=='chase':sizes['auxiliary']=len(aux)
    data={'input':digest(main),'output':digest(b'\xa5'*len(expected))}
    if 'auxiliary' in sizes:data['auxiliary']=digest(aux)
    scope={'run':ledger['run'],'case':cid,'context':ledger['context']}
    _require(all(isinstance(x,str) and x for x in scope.values()),'explicit run/case/context scope required')
    trace=ledger['trace'];slots=ledger['slots']
    _require(isinstance(trace,list),'ordered API observations required')
    initialization=[('allocate_input','allocation','input'),('allocate_output','allocation','output'),
                    ('upload_input','allocation','input')]
    if 'auxiliary' in sizes:initialization += [('allocate_auxiliary','allocation','auxiliary'),('upload_auxiliary','allocation','auxiliary')]
    initialization += [('event_create_start','event','start'),('event_create_stop','event','stop')]
    slot_ops=('write_started','reset_output','context_synchronize','event_record_start','launch',
              'event_record_stop','event_synchronize_stop','event_elapsed','write_event',
              'download_output','write_output','write_result')
    objects={};acquired=[];init_at=0;slot_at=0;op_at=0;fatal=False
    # A failed API normally jumps to write_result; elapsed retains its raw event
    # first. Output correctness/nonfinite/time errors are checked after download.
    tail=[];cleanup_started=False;cleanup_queue=[];cleanup_at=0;cleanup_errors=[]
    remaining_events=[];remaining_allocations=[];init_failure=None;slot_failures=[]

    def resource(row,kind,name,*,create=False,size=0,sha=None):
        _require(len(row['resources'])==1,'one explicit resource expected for '+row['operation'])
        r=row['resources'][0]
        _require(r['kind']==kind and r['name']==name,'wrong object/category for '+row['operation'])
        _require(r['bytes']==size and (sha is None or r['sha256']==sha),'request size/data differs from actual program: '+row['operation'])
        if row['operation']=='download_output':_require(isinstance(r['sha256'],str),'download result buffer identity missing')
        elif sha is None:_require(r['sha256'] is None,'unexpected data identity for '+row['operation'])
        if create:
            handle=r['handle']
            _require((kind,name) not in objects,'duplicate object creation')
            if handle:
                _require(not any(o.kind==kind and o.handle==handle for o in acquired),'aliased output handle')
                o=Object(kind,name,handle,size,row['index'],row['return_code']);objects[kind,name]=o;acquired.append(o)
            return handle
        _require((kind,name) in objects,'use without create: '+name)
        obj=objects[kind,name]
        _require(obj.live and obj.handle==r['handle'],'wrong/dead object handle: '+name)
        return obj

    def refs(row,names):
        _require(len(row['resources'])==len(names),'resource operand count: '+row['operation'])
        found=[]
        for r,(kind,name) in zip(row['resources'],names):
            _require(r['kind']==kind and r['name']==name and r['bytes']==0 and r['sha256'] is None,'wrong resource operand: '+row['operation'])
            _require((kind,name) in objects,'operand used before create: '+name)
            o=objects[kind,name];_require(o.live and o.handle==r['handle'],'wrong/dead operand: '+name);found.append(o)
        return found

    def timeout(slot=None):
        end=ledger['ended_ns'] if slot is None else slot['ended_realtime_ns']
        return type(end) is int and end>=ledger['deadline_ns']

    def finishable():
        return fatal or (init_at==len(initialization) and slot_at==len(slots)) or timeout()

    for index,row in enumerate(trace):
        exact(row,{'index','sequence','operation','return_code','scope','resources'},'resource API observation')
        _require(type(row['index']) is int and row['index']==index,'non-contiguous API order')
        _require(row['scope']==scope,'foreign run/case/context resource')
        _require(type(row['return_code']) is int and -2**31<=row['return_code']<2**31,'raw API return code required')
        _require(isinstance(row['resources'],list),'resource operands required')
        for r in row['resources']:
            exact(r,{'kind','name','handle','bytes','sha256'},'observed API resource')
            _require(r['kind'] in ('allocation','event') and isinstance(r['name'],str),'unknown resource kind/name')
            _require(type(r['handle']) is int and 0<=r['handle']<2**64 and type(r['bytes']) is int and r['bytes']>=0,'bounded resource output/request')
            _require(r['sha256'] is None or isinstance(r['sha256'],str) and re.fullmatch('[0-9a-f]{64}',r['sha256']),'resource data SHA')
        op=row['operation'];rc=row['return_code'];seq=row['sequence']
        _require(isinstance(op,str) and bool(op),'API operation name required')
        is_cleanup=op.startswith(('cleanup_','event_destroy_','free_'))
        if is_cleanup:
            if not cleanup_started:
                _require(finishable() and not tail,'cleanup before normal termination')
                cleanup_started=True
                events=[o for o in acquired if o.kind=='event'];allocs=[o for o in acquired if o.kind=='allocation']
                cleanup_queue=([('cleanup_context_synchronize',None)] if acquired else [])
                cleanup_queue += [('event_destroy_'+o.name,o) for o in reversed(events)]
                cleanup_queue += [('free_'+o.name,o) for o in reversed(allocs)]
            _require(seq is None and cleanup_at<len(cleanup_queue),'extra/duplicated cleanup attempt')
            want,obj=cleanup_queue[cleanup_at];_require(op==want,'missing/reordered cleanup: expected '+want)
            if obj is None:_require(not row['resources'],'cleanup synchronization has no object operands')
            else:
                resource(row,obj.kind,obj.name)
                if rc==0:obj.live=False
                elif obj.kind=='event':remaining_events.append(obj.handle)
                else:remaining_allocations.append(obj.handle)
            if rc!=0:cleanup_errors.append({'operation':op,'return_code':rc,'index':index})
            cleanup_at+=1;continue
        _require(not cleanup_started,'normal execution after cleanup')
        if init_at<len(initialization) and not fatal:
            want,kind,name=initialization[init_at]
            _require(seq is None and op==want,'missing/reordered initialization: expected '+want)
            create=op.startswith(('allocate_','event_create_'))
            out=resource(row,kind,name,create=create,size=sizes.get(name,0),sha=data[name] if op.startswith('upload_') else None)
            if op.startswith('upload_') and rc==0:out.uploaded=True
            init_at+=1
            if rc!=0 or create and not out:
                fatal=True;init_failure={'operation':op,'return_code':rc,'output_handle':out if create else None,'index':index}
            continue
        _require(slot_at<len(slots) and init_at==len(initialization),'normal call after initialization failure or after all slots')
        _require(not fatal or bool(tail),'normal execution continued after first failure')
        slot=slots[slot_at]
        _require(type(seq) is int and seq==slot_at and slot['sequence']==seq and slot['attempted'],'API call not attached to its actual attempted slot')
        if tail:
            _require(op==tail.pop(0),'normal execution continued after first failure')
        else:
            if op=='write_result' and op_at<len(slot_ops)-1:
                _require(slot['status']=='failed' and timeout(slot),'unexplained truncated slot execution')
                fatal=True;slot_failures.append({'sequence':seq,'reason':'deadline'})
            else:_require(op==slot_ops[op_at],'missing/reordered slot operation: expected '+slot_ops[op_at])
        _require(not op.startswith('write_') or rc==0 and not row['resources'],'sink operation has unexpected failure/resource operands')
        if op=='write_started':
            _require(slot['start_event']==str(objects['event','start'].handle) and slot['end_event']==str(objects['event','stop'].handle),
                     'sample event IDs differ from created pair')
        elif op=='reset_output':resource(row,'allocation','output',size=sizes['output'],sha=data['output'])
        elif op=='context_synchronize':_require(not row['resources'],'context synchronize cannot name objects')
        elif op in ('event_record_start','event_record_stop','event_synchronize_stop'):
            name='start' if op=='event_record_start' else 'stop';o=resource(row,'event',name)
            if op.startswith('event_record') and rc==0:o.generation=seq;o.synchronized=False
            if op=='event_synchronize_stop':
                _require(o.generation==seq,'synchronize stale/unrecorded event')
                if rc==0:o.synchronized=True
        elif op=='launch':
            names=[('allocation','input'),('allocation','output')]+([('allocation','auxiliary')] if 'auxiliary' in sizes else [])
            operands=refs(row,names)
            _require(all(o.uploaded for o in operands if o.name!='output'),'launch consumes input that was never uploaded')
            _require(objects['event','start'].live and objects['event','start'].generation==seq,'launch before this slot start event')
        elif op=='event_elapsed':
            a,b=refs(row,[('event','start'),('event','stop')])
            _require(a.generation==b.generation==seq and b.synchronized,'elapsed uses stale/unsynchronized event pair')
        elif op=='download_output':
            resource(row,'allocation','output',size=sizes['output'])
            if rc==0:
                _require(slot['actual_output'] is not None and row['resources'][0]['sha256']==slot['actual_output']['sha256'],'download bytes not bound to actual output')
        elif op.startswith('write_'):_require(not row['resources'],'file write cannot create/use Driver objects')
        else:_require(False,'unknown normal API operation')
        if not op.startswith('write_') and rc!=0:
            _require(slot['status']=='failed','API failure hidden by slot status')
            fatal=True;slot_failures.append({'sequence':seq,'operation':op,'return_code':rc})
            tail=['write_event','write_result'] if op=='event_elapsed' else ['write_result']
        if op=='write_output':
            bits=slot['event_elapsed_ms_float32_bits']
            value=struct.unpack('<f',struct.pack('<I',bits))[0] if type(bits) is int else float('nan')
            if slot['correct'] is not True or not math.isfinite(value) or value<=0 or timeout(slot):
                _require(slot['status']=='failed','output/time failure hidden by returned slot')
                fatal=True;slot_failures.append({'sequence':seq,'reason':'output/event/deadline'});tail=['write_result']
        if op=='write_result':
            _require(slot['status']==('failed' if fatal else 'returned'),'slot terminal contradicts observed API/output failure')
            # The slot API map is independently tied to every observed raw code.
            calls={t['operation']:t['return_code'] for t in trace[:index+1] if t['sequence']==seq and not t['operation'].startswith('write_')}
            _require(slot['returncodes']==calls,'slot API codes differ from actual observed calls')
            slot_at+=1;op_at=0
        else:op_at=slot_ops.index(op)+1

    _require(not tail,'required error-boundary original is missing')
    _require(finishable(),'missing necessary initialization/normal execution tail')
    _require(all(s['status']=='not_started' and not s['attempted'] for s in slots[slot_at:]),'execution after failure or missing attempted slot evidence')
    if acquired:
        _require(cleanup_started and cleanup_at==len(cleanup_queue),'missing cleanup attempt for acquired resources')
    else:_require(not cleanup_started,'cleanup claims objects that were never acquired')
    remaining=[o for o in acquired if o.live]
    _require(set(remaining_events)=={o.handle for o in remaining if o.kind=='event'} and set(remaining_allocations)=={o.handle for o in remaining if o.kind=='allocation'},'incomplete cleanup reconstruction')
    complete=not remaining and not cleanup_errors
    claim=ledger['cleanup']
    _require(claim['remaining_events']==remaining_events and claim['remaining_allocations']==remaining_allocations,'reported remaining differs from derived live objects')
    _require(claim['all_resources_released'] is complete and claim['status']==('closed' if complete else 'failed'),'reported cleanup contradicts derived lifecycle')
    byte_count=sum(o.size for o in acquired if o.kind=='allocation')
    _require(ledger['application_bytes']==byte_count and byte_count<=bundle.document['budget']['app_bytes'],'acquired byte accounting differs')
    return {'scope':scope,'acquired':[{'kind':o.kind,'name':o.name,'handle':o.handle,'bytes':o.size,
            'create_index':o.created_at,'create_return_code':o.create_return_code,'live':o.live} for o in acquired],
        'initialization_failure':init_failure,'slot_failures':slot_failures,'cleanup_errors':cleanup_errors,
        'remaining_events':remaining_events,'remaining_allocations':remaining_allocations,
        'cleanup_complete':complete,'application_bytes':byte_count,'initialized':init_at==len(initialization) and init_failure is None}
