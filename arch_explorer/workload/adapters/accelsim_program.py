"""Accel-Sim adapter program assembly; all CPU checks precede publication."""
from collections import Counter
from dataclasses import dataclass
import math
from pathlib import Path
import numpy as np
from arch_explorer.components.accelsim.lowering_limits import CAPABILITIES, LIMITS, canonical, sha
from arch_explorer.components.accelsim.emitter import Memory, Kernel, ProgramEmitter, Emission
TEMPLATE = 'accelsim_program'

def _selectors(graph,mapping):
    if mapping.template != TEMPLATE or set(mapping.params)-{'threads_per_block','implementations'}:
        raise ValueError('use accelsim_program with threads_per_block and optional implementations')
    threads = mapping.params.get('threads_per_block',32)
    if type(threads) is not int or threads not in (32,64,128):
        raise ValueError('threads_per_block must be 32, 64 or 128')
    selected = mapping.params.get('implementations',{})
    if (type(selected) is not dict or set(selected)-{op.name for op in graph.operators}
            or any(type(v) is not str or not v for v in selected.values())):
        raise ValueError('implementations must map actual operator node names to implementation IDs')
    return threads,selected

def _select(graph,mapping):
    from arch_explorer.components.accelsim.builtin_emitters import install
    from arch_explorer.workload.execution import select_implementation
    install()
    threads,selectors = _selectors(graph,mapping)
    sizes = [math.prod(t.shape) for t in graph.tensors.values()]
    if graph.iterations != 1:
        raise ValueError('native reference program requires one explicit DAG iteration')
    if any(t.dtype not in ('fp32','int32') for t in graph.tensors.values()):
        raise ValueError('scalar program storage supports fp32/int32 only')
    if (max(sizes,default=0)>LIMITS['tensor_elements'] or sum(sizes)>LIMITS['total_tensor_elements']
            or len(graph.operators)>LIMITS['kernels']):
        raise ValueError('complete program tensor/operator expansion bound exceeded')
    result=[]
    for op in graph.operators:
        try:
            result.append(select_implementation(op,'accelsim',
                dtype=graph.tensors[op.outputs[0]].dtype,tensors=graph.tensors,
                capabilities=CAPABILITIES,implementation=selectors.get(op.name),
                options={'threads_per_block':threads}))
        except (ValueError,TypeError,NotImplementedError) as error:
            raise ValueError('operator '+op.name+' on accelsim: '+str(error)) from error
    return result

def inspect_graph(graph,mapping):
    try:
        from arch_explorer.workload.execution import implementation_identity
        definitions=_select(graph,mapping)
        selected=[{'node':op.name,**implementation_identity(d)}
                  for op,d in zip(graph.operators,definitions)]
        gaps=[]
    except (ValueError,TypeError,KeyError,NotImplementedError) as error:
        gaps,selected=[{'kind':'implementation','reason':str(error)}],[]
    return {'status':'unsupported' if gaps else 'supported','gaps':gaps,
            'operator_coverage':dict(Counter(op.op for op in graph.operators)),
            'operators':len(graph.operators),'limits':dict(LIMITS),'implementations':selected}

def _check_numeric(name,result,expected,dtype):
    result,expected=np.asarray(result,dtype=dtype),np.asarray(expected,dtype=dtype)
    if result.shape!=expected.shape:
        raise ValueError('emitted result shape differs: '+name)
    if np.issubdtype(np.dtype(dtype),np.integer):
        valid,error=np.array_equal(result,expected),0
    else:
        valid=(np.array_equal(np.isneginf(result),np.isneginf(expected))
               and np.allclose(result,expected,rtol=3e-4,atol=3e-5,equal_nan=False))
        finite=np.isfinite(result)&np.isfinite(expected)
        error=float(np.max(np.abs(result[finite]-expected[finite]),initial=0))
    if not valid:
        raise ValueError('lowered numeric result differs: '+name+' max_abs='+str(error))
    return {'tensor':name,'max_abs_error':error,'actual_sha256':sha(result.tobytes()),
            'reference_sha256':sha(expected.tobytes()),'rtol':3e-4,'atol':3e-5}

@dataclass
class PreparedProgram:
    document:dict
    files:dict
    source_objects:tuple

    def publish(self,directory):
        directory=Path(directory)
        directory.mkdir(exist_ok=False)
        (directory/'inputs').mkdir()
        for name,raw in self.files.items():
            with (directory/name).open('xb') as f:
                f.write(raw)
        from arch_explorer.components.accelsim.trace_stream import facts
        for kernel in self.document['kernels']:
            summary,_=facts(directory/kernel['file'])
            kernel['facts']=summary
        return self.document

def prepare_program(reference,mapping):
    """Pure CPU in-memory emission and complete numerical/ISA checks; no files."""
    from arch_explorer.workload.execution import ExecutionContext,lower_implementation
    from arch_explorer.workload.extensions import source_trace
    from arch_explorer.components.accelsim.trace_stream import instruction
    graph=reference['graph']
    definitions=_select(graph,mapping)
    assessment=inspect_graph(graph,mapping)
    threads,_=_selectors(graph,mapping)
    memory,files=Memory(),{}
    for name in reference['input_tensors']:
        t=graph.tensors[name]
        memory.allocate(name,t.shape,np.float32 if t.dtype=='fp32' else np.int32,
                        reference['tensors'][name],'external-input-or-weight')
    kernels,operators,checks=[],[],[]
    records=0
    subjects=[prepare_program,*reference.get('_source_objects',())]
    subjects += [fn for d in definitions for fn in (d.lower,d.supports) if fn is not None]
    with source_trace() as used:
        for op,definition in zip(graph.operators,definitions):
            emitter=ProgramEmitter(memory,graph,op,threads)
            context=ExecutionContext(target='accelsim',dtype=graph.tensors[op.outputs[0]].dtype,
                tensors=graph.tensors,capabilities=CAPABILITIES,
                options={'threads_per_block':threads,'emitter':emitter})
            emitted=lower_implementation(definition,op,context)
            if not isinstance(emitted,Emission) or not emitter._finished:
                raise TypeError('accelsim implementation must return emitter.finish_operator()')
            if (emitted.kernels!=tuple(emitter.kernels) or emitted.aliases!=tuple(emitter.aliases)
                    or emitted.temporaries!=tuple(emitter.temporaries)):
                raise ValueError('implementation returned a foreign emission')
            for name in op.outputs:
                t=graph.tensors[name];dtype=np.float32 if t.dtype=='fp32' else np.int32
                value=memory.read(memory.addresses[name]).view(dtype).reshape(t.shape)
                checks.append(_check_numeric(name,value,reference['tensors'][name],dtype))
            operator_kernels=[]
            for k in emitted.kernels:
                if len(kernels)>=LIMITS['kernels']:
                    raise ValueError('lowered kernel count bound exceeded')
                uid=len(kernels)+1;k.name='ae_'+str(uid)+'_'+op.op
                if len(k.name)>4096 or any(ord(c)<32 or ord(c)>126 for c in k.name):
                    raise ValueError('native kernel label outside bounded ASCII domain')
                raw,count=k.render_bytes(uid);records+=count
                if records>LIMITS['trace_records']:
                    raise ValueError('complete program trace record bound exceeded')
                for line in raw.decode('ascii').splitlines():
                    if line and line[0] in '0123456789abcdef' and not line.startswith(('insts','thread')):
                        instruction(line,{'nregs':64,'lineinfo':0,'version':3})
                name='kernel-'+str(uid)+'.traceg';files[name]=raw
                kernels.append({'file':name,'sha256':sha(raw),'operator':op.name,
                    'instruction_template':[{'pc':(j if j<k.prologue else j+1)*16,
                        **{key:value for key,value in ins.items() if key!='addresses'}}
                        for j,ins in enumerate(k.code)],
                    'logical_output_elements':k.size,'requested_threads_per_block':threads})
                operator_kernels.append(name)
            operators.append({'name':op.name,'op':op.op,'inputs':list(op.inputs),'outputs':list(op.outputs),
                'kernel':operator_kernels[0] if len(operator_kernels)==1 else None,
                'kernels':operator_kernels,'aliases':list(emitted.aliases),'temporaries':list(emitted.temporaries),
                'implementation':{'id':definition.id,'version':definition.version,'target':definition.target},
                'attrs':op.attrs})
        subjects+=used
    outputs={}
    for number,name in enumerate(reference['outputs']):
        t=graph.tensors[name];dtype=np.float32 if t.dtype=='fp32' else np.int32
        value=memory.read(memory.addresses[name]).view(dtype).reshape(t.shape)
        _check_numeric(name,value,reference['outputs'][name],dtype)
        file='output-'+str(number)+'.bin';files[file]=value.tobytes()
        outputs[name]={'shape':list(value.shape),'dtype':t.dtype,'file':file,'sha256':sha(files[file]),
                      'values':value.tolist(),'reference_sha256':sha(np.asarray(reference['outputs'][name],dtype=dtype).tobytes())}
    buffers=[]
    for number,b in enumerate(memory.buffers):
        row={'name':b.name,'address':b.base,'bytes':int(b.bits.nbytes),'kind':b.kind,
             'initialized_after_program':bool(np.all(b.initialized))}
        if b.kind!='activation':
            file='inputs/buffer-'+str(number)+'.bin';files[file]=b.bits.tobytes()
            row.update(file=file,sha256=sha(files[file]))
        buffers.append(row)
    if not kernels:
        raise ValueError('complete program must contain a real instruction kernel')
    return PreparedProgram({'schema':'arch-explorer.reference-trace-program.v1',
        'kind':'reference-generated-synthetic','isa':'sm90','stream':0,'kernels':kernels,
        'operators':operators,'buffers':buffers,'outputs':outputs,'mapping':mapping.to_dict(),
        'numeric_checks':checks,'reference':reference['reference'],'assessment':assessment,
        'execution_sources':list(used.identities),
        'scope':{'implementation':'registered target implementations emit concrete scalar SM90 instructions',
            'initialization':'bound external inputs/weights; activation/temporary storage uninitialized until stores',
            'addresses':'synthetic consistent 32-bit global virtual addresses; not captured GPU pointers',
            'numerics':'CPU interpreter executes actual emitted register/memory operations with per-output reference checks',
            'native':'timing-only trace execution; no functional GPU output or hardware accuracy claim',
            'cache':'one ordered stream; every emitted step is a real native kernel'}},files,tuple(subjects))

def lower(reference,mapping,directory):
    return prepare_program(reference,mapping).publish(directory)

