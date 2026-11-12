# Concrete Accel-Sim implementation; owned by the component provider.
"""Workload graph -> mapped reference trace program -> native Evaluation.

This API never estimates GPU cycles. Numerical validation executes the explicit
lowered implementation on CPU; timing comes from one external native simulator
process consuming all kernels in the same stream.
"""
from dataclasses import asdict
import json
import math
from pathlib import Path
import shutil
import tempfile

from arch_explorer.adf import Evaluation, MappingSpec, WorkloadSpec, SpacePoint, from_dict, walk
from arch_explorer.components.accelsim.lowering import canonical, inspect_graph, lower, prepare_program, sha
from arch_explorer.workload.reference_api import execute_reference
from arch_explorer.components.accelsim.config import compile_adf
from arch_explorer.components.accelsim.profile import ACCELSIM_SHA, GPGPUSIM_SHA
from arch_explorer.components.accelsim.program_stats import parse_program_stats
from arch_explorer.components.accelsim.runner import load_engine, run_native
from arch_explorer.components.accelsim.trace import exact
from arch_explorer.components.accelsim.trace_stream import facts

PROGRAM_REQUEST_SCHEMA = 'arch-explorer.accelsim-program-request.v1'
PROGRAM_RESULT_SCHEMA = 'arch-explorer.accelsim-program-result.v1'
_FIELDS = {'schema','adf','workload','mapping','options','program'}
_PROGRAM_SOURCE = 'components/sm/implementations/accelsim/program.py'
_REPLAY_POLICY = 'arch-explorer.accelsim-component-migration.v1'


def _read(path, maximum=32*2**20):
    path = Path(path)
    if not path.is_file() or path.stat().st_size > maximum: raise ValueError('missing/oversized program artifact '+str(path))
    return path.read_bytes()


def _write(path, data):
    with Path(path).open('xb') as f: f.write(canonical(data)+b'\n')
    return {'path':str(Path(path).resolve()),'sha256':sha(Path(path).read_bytes())}


def _sources(subjects=()):
    from arch_explorer.components.accelsim.program_sources import sources
    return sources(subjects)


def _graph(graph):
    return {'name':graph.name,'dtype':graph.dtype,'tensors':{n:asdict(t) for n,t in graph.tensors.items()},
            'operators':[asdict(op) for op in graph.operators],'meta':graph.meta}


def _binding(adf, workload, mapping, config, subjects=()):
    return {'adf':sha(canonical(adf.to_dict())), 'workload':sha(canonical(workload.to_dict())),
            'mapping':sha(canonical(mapping.to_dict())), 'config':config.sha256,'sources':_sources(subjects)}


def _check_binding(producer, consumer, *, read_only=False):
    exact(producer, set(consumer), 'complete program binding')
    if any(producer[k] != consumer[k] for k in consumer if k != 'sources'):
        raise ValueError('program bound to different ADF/workload/mapping/configuration')
    if producer['sources'] != consumer['sources']:
        if not read_only:
            raise ValueError('new native execution requires current semantic/target sources')
        from arch_explorer.components.accelsim.source_compat import program_source_era
        program_source_era(producer['sources'], consumer['sources'])


def _comparison_view(document, binding):
    """A transient comparison view; producer documents/identities stay intact."""
    _check_binding(document.get('binding'), binding, read_only=True)
    if document['binding']['sources'] != binding['sources']:
        from arch_explorer.components._support.accelsim.layout_history import callable_view
        document = callable_view(document)
    reference = document.get('reference')
    if not isinstance(reference, dict):
        raise ValueError('complete numerical reference required for program regeneration')
    version = reference.get('numpy_version')
    if (not isinstance(version, str) or not 1 <= len(version) <= 128
            or version != version.strip() or any(not 32 <= ord(c) < 127 for c in version)):
        raise ValueError('producer NumPy provenance must be a bounded nonempty version label')
    # Only this environment label is non-semantic. Tensor/weight/output hashes,
    # numeric checks, source records, coverage, steps and limits remain in view.
    reference = dict(reference, numpy_version='<producer-environment-provenance>')
    return dict(document, reference=reference,
                binding=dict(document['binding'], sources=dict(binding['sources'])))


def _difference(a, b, path=''):
    """First differing JSON pointer, for honest numerical/semantic diagnostics."""
    if type(a) is not type(b): return path or '/'
    if isinstance(a, dict):
        for key in sorted(set(a) | set(b)):
            here = path+'/'+key.replace('~','~0').replace('/','~1')
            if key not in a or key not in b: return here
            found = _difference(a[key], b[key], here)
            if found is not None: return found
    elif isinstance(a, list):
        if len(a) != len(b): return path+'/length'
        for i, (x, y) in enumerate(zip(a, b)):
            found = _difference(x, y, path+'/'+str(i))
            if found is not None: return found
    elif a != b:
        return path or '/'
    return None


def _validation_provenance(document):
    import numpy as np
    from arch_explorer.components.accelsim.source_compat import program_source_era
    reference = document.get('reference')
    producer = document['binding']['sources']
    consumer = _sources()
    # _load already revalidated the full actual target/semantic source vector.
    era = 'current-registered-execution' if producer.get(_PROGRAM_SOURCE) == consumer.get(_PROGRAM_SOURCE) else program_source_era(producer,consumer)
    source = next((key for key in (_PROGRAM_SOURCE, 'components/accelsim/program.py', 'simulator/accelsim/program.py') if key in producer), None)
    if source is None:
        raise ValueError('exact producer program implementation identity missing')
    return {'policy':_REPLAY_POLICY,
        'producer':{'numpy_version':reference.get('numpy_version') if isinstance(reference,dict) else None,
                    'program_source_sha256':producer[source], 'source_era':era},
        'consumer':{'numpy_version':np.__version__, 'program_source_sha256':consumer[_PROGRAM_SOURCE]},
        'semantic_comparison':'exact registered semantic/target sources and all graph/numerical/instruction content; only NumPy provenance label excluded'}


def _compile(adf):
    from arch_explorer.components.native_bindings import native_adf
    if any(isinstance(node, SpacePoint) and node.component == 'accelsim_sm120_mechanism' for _,node in walk(native_adf(adf))):
        raise ValueError('complete FP32 programs require ordinary SM90 components; the SM120 integer mechanism is a separate contract')
    return compile_adf(adf)


def _workload_graph(workload):
    from arch_explorer.workload.api import workload_from_spec
    return workload_from_spec(workload)


def prepare_request(adf, workload, mapping, *, output_dir, engine=None, options=None):
    if options is None:
        options = {}
    if type(options) is not dict or set(options)-{'timeout_s','max_cycles','memory_limit_mb'}:
        raise ValueError('program options only select native runtime bounds')
    from arch_explorer.components.accelsim.component_execution import options as checked_options
    checked_options({'engine':'preflight-only','timeout_s':120,'max_cycles':10000000,
                     'memory_limit_mb':4096,**options})
    root = Path(output_dir).resolve()
    if root.exists() and any(root.iterdir()):
        raise ValueError('program preparation requires a new artifact namespace')
    config = _compile(adf)
    reference = execute_reference(workload)
    graph = reference['graph']
    planned = prepare_program(reference,mapping)
    binding = _binding(adf,workload,mapping,config,planned.source_objects)
    # Missing/ambiguous implementation, dtype/layout/resource/ISA/numeric errors
    # all occur before this first output namespace side effect.
    root.mkdir(parents=True,exist_ok=True)
    document = planned.publish(root/'program')
    document.update(binding=binding,graph=_graph(graph))
    program = _write(root/'PROGRAM.json',document)
    request = {'schema':PROGRAM_REQUEST_SCHEMA,'adf':adf.to_dict(),'workload':workload.to_dict(),
               'mapping':mapping.to_dict(),'program':program,
               'options':{'engine':None if engine is None else str(Path(engine).resolve()),
                          'output_dir':str(root/'native'),'timeout_s':120,
                          'max_cycles':10000000,'memory_limit_mb':4096,**options}}
    _write(root/'REQUEST.json',request)
    return request


def _load(request, *, regenerate=True, read_only=False):
    exact(request, _FIELDS, 'complete program request')
    if request['schema'] != PROGRAM_REQUEST_SCHEMA: raise ValueError('unsupported complete program request schema')
    exact(request['program'], {'path','sha256'}, 'complete program artifact reference')
    path = Path(request['program']['path'])
    if not path.is_absolute(): raise ValueError('program artifact path must be absolute')
    raw = _read(path)
    if sha(raw) != request['program']['sha256']: raise ValueError('program manifest bytes differ')
    document = json.loads(raw)
    if document.get('schema') != 'arch-explorer.reference-trace-program.v1': raise ValueError('unsupported trace program version')
    options = request['options']
    if not isinstance(options, dict) or set(options)-{'engine','output_dir','timeout_s','max_cycles','memory_limit_mb'}:
        raise ValueError('unsupported native runtime options')
    if not {'engine','output_dir'} <= set(options) or not Path(options['output_dir']).is_absolute():
        raise ValueError('explicit engine and absolute native evidence directory required')
    adf, workload, mapping = from_dict(request['adf']), WorkloadSpec(**request['workload']), MappingSpec(**request['mapping'])
    config = _compile(adf)
    reference = execute_reference(workload)
    graph = reference['graph']
    planned = prepare_program(reference,mapping)
    binding = _binding(adf,workload,mapping,config,planned.source_objects)
    _check_binding(document.get('binding'),binding, read_only=read_only)
    assessment = planned.document['assessment']
    observed_assessment = document.get('assessment')
    if read_only and document['binding']['sources'] != binding['sources']:
        from arch_explorer.components._support.accelsim.layout_history import callable_view
        observed_assessment = callable_view(observed_assessment)
    if canonical(observed_assessment) != canonical(assessment) or canonical(document.get('graph')) != canonical(_graph(graph)):
        raise ValueError('program complete graph/assessment differs from public workload and mapping')
    if assessment['status'] != 'supported': return document, config, path.parent/'program'
    if regenerate:
        with tempfile.TemporaryDirectory(prefix='ae-program-verify-') as tmp:
            rebuilt = planned.publish(Path(tmp)/'program')
            rebuilt.update(binding=binding, graph=_graph(graph))
            expected, observed = _comparison_view(rebuilt, binding), _comparison_view(document, binding)
            expected_bytes, observed_bytes = canonical(expected), canonical(observed)
            if expected_bytes != observed_bytes:
                provenance = _validation_provenance(document)
                difference = _difference(json.loads(expected_bytes), json.loads(observed_bytes))
                raise ValueError('program semantic regeneration differs at '+str(difference)
                    +'; producer NumPy '+provenance['producer']['numpy_version']
                    +', consumer NumPy '+provenance['consumer']['numpy_version']
                    +'; only the NumPy provenance label is excluded from semantic comparison')
    program_root = path.parent/'program'
    names = set()
    for item in document['kernels']:
        name = item['file']
        if name in names or name != 'kernel-'+str(len(names)+1)+'.traceg': raise ValueError('kernel program order/filename differs')
        names.add(name)
        file = program_root/name
        if sha(_read(file, 64*2**20)) != item['sha256']: raise ValueError('kernel trace bytes differ')
        summary, _ = facts(file)
        if summary != item['facts']: raise ValueError('kernel trace geometry/instruction closure differs')
    for item in [*document['buffers'], *document['outputs'].values()]:
        if 'file' in item:
            relative = Path(item['file'])
            if relative.is_absolute() or '..' in relative.parts: raise ValueError('nonlocal program buffer')
            if sha(_read(program_root/relative)) != item['sha256']: raise ValueError('program input/output bytes differ')
    return document, config, program_root


def assess_request(request):
    """Read/execute only bounded CPU validation; no device or native launch."""
    try:
        document, config, _ = _load(request)
        assessment = dict(document['assessment'])
        assessment['gaps'] = list(assessment['gaps'])
        if assessment['status'] == 'supported':
            try: load_engine(request['options']['engine'])
            except (ValueError, ImportError, OSError, TypeError) as error:
                assessment['gaps'].append({'kind':'engine','reason':str(error)})
        assessment['status'] = 'unsupported' if assessment['gaps'] else 'supported'
        assessment.update(schema='arch-explorer.accelsim-program-assessment.v1',
            kernel_count=len(document['kernels']), stream=0, instruction_set='ordinary SM90 scalar FP32',
            config_sha256=config.sha256, hardware_accuracy='not validated',
            validation_provenance=_validation_provenance(document))
        return assessment
    except (ValueError, TypeError, KeyError, OSError, ImportError) as error:
        return {'schema':'arch-explorer.accelsim-program-assessment.v1','status':'unsupported',
                'gaps':[{'kind':'artifact','reason':str(error)}]}


class _ProgramFiles:
    def __init__(self, document, directory): self.document, self.directory = document, directory
    def stage(self, target):
        names = []
        for item in self.document['kernels']:
            name = item['file']; raw = _read(self.directory/name, 64*2**20)
            if sha(raw) != item['sha256']: raise ValueError('trace changed before native staging')
            with (target/name).open('xb') as f: f.write(raw)
            names.append(name)
        return names


def _consume(request, document, config, directory, run):
    directory = Path(directory)
    from arch_explorer.components.accelsim.source_compat import adapter_source_era
    adapter_source_era(run.get('adapter_sources_sha256'))
    engine, binary, _, engine_hash = load_engine(request['options']['engine'])
    if (run.get('engine_manifest_sha256') != engine_hash or run.get('engine') != engine
        or run.get('config_sha256') != config.sha256 or run.get('adf_sha256') != config.adf_sha256
        or run.get('cwd') != str(directory) or directory.parent != Path(request['options']['output_dir'])
        or run.get('exit_code') != 0
        or run.get('environment',{}).get('CUDA_VISIBLE_DEVICES') != ''):
        raise ValueError('native program evidence engine/config/process scope differs')
    expected_argv = [str(binary),'-config','gpgpusim.config','-config','limits.config','-trace','./kernelslist.g']
    if run.get('argv') != expected_argv: raise ValueError('native program invocation differs')
    limits = {key:request['options'].get(key,default) for key,default in
              (('timeout_s',120),('max_cycles',1000000),('memory_limit_mb',4096))}
    if run.get('limits') != limits or _read(directory/'limits.config') != ('-gpgpu_max_cycle '+str(limits['max_cycles'])+'\n').encode():
        raise ValueError('native program runtime limits differ from request')
    for name, expected in run['files'].items():
        if Path(name).name != name or sha(_read(directory/name,64*2**20)) != expected: raise ValueError('staged native input changed')
    for name, content in config.artifacts.items():
        if _read(directory/name) != content.encode(): raise ValueError('native staged config differs')
    names = [k['file'] for k in document['kernels']]
    if _read(directory/'kernelslist.g') != ''.join(n+'\n' for n in names).encode(): raise ValueError('native kernel list incomplete')
    for item in document['kernels']:
        if sha(_read(directory/item['file'],64*2**20)) != item['sha256']: raise ValueError('native kernel does not match mapped trace')
    stdout, stderr = _read(directory/'stdout.log',256*2**20), _read(directory/'stderr.log')
    if sha(stdout) != run['stdout_sha256'] or sha(stderr) != run['stderr_sha256']: raise ValueError('native stdout/stderr bytes differ')
    stdout, stderr = stdout.decode(errors='replace'), stderr.decode(errors='replace')
    if ACCELSIM_SHA[:7] not in stdout or GPGPUSIM_SHA[:7] not in stdout: raise ValueError('native source build identity absent')
    core_mhz = next(r['value'] for r in config.component_map if r['param']=='clock_mhz' and r['clock_domain']=='sm')
    stats = parse_program_stats(stdout, stderr, [k['facts'] for k in document['kernels']], core_mhz)
    provenance = [{k:r[k] for k in ('path','param','value','unit','source')} for r in config.component_map]
    return Evaluation(stats['latency_ns'], backend='accelsim', provenance=provenance,
        breakdown=[{'kernel':k['name'],'operator':op['operator'],'cycles':k['cycles'],
                    'latency_ns':k['cycles']*1000/core_mhz} for k,op in zip(stats['kernels'],document['kernels'])],
        warnings=['Reference-generated synthetic scalar FP32 program with synthetic weights and addresses.',
                  'Native software execution; hardware accuracy and framework compiler/kernel equivalence are unvalidated.'],
        details={'schema':'arch-explorer.accelsim-program-evaluation.v1','program_sha256':request['program']['sha256'],
            'complete_operator_count':len(document['operators']),'operator_coverage':document['assessment']['operator_coverage'],
            'kernel_count':len(document['kernels']),'mapping':request['mapping'],'native':stats,
            'config_sha256':config.sha256,'adf_sha256':config.adf_sha256,'component_map':config.component_map,
            'numeric_checks':document['numeric_checks'],'outputs':document['outputs'],'scope':document['scope'],
            'hardware_accuracy':{'status':'not validated','mape':None,'worst':None},
            'evidence':str(directory),'run':run}).to_dict()


def evaluate_request(request_or_result):
    """Execute a new program, or read-only validate/replay its existing result."""
    replay = request_or_result.get('schema') == PROGRAM_RESULT_SCHEMA
    if replay:
        exact(request_or_result, {'schema','request','evaluation'}, 'complete program result')
    request = request_or_result['request'] if replay else request_or_result
    document, config, root = _load(request, read_only=replay)
    if document['assessment']['status'] != 'supported':
        raise ValueError('unsupported complete program: '+json.dumps(document['assessment']['gaps']))
    if replay:
        old = request_or_result['evaluation']; directory = Path(old['details']['evidence'])
        raw = json.loads(_read(directory/'run.json'))
        if raw != old['details']['run'] or raw.get('status') != 'complete': raise ValueError('native replay run receipt differs')
        result = _consume(request, document, config, directory, raw)
        if canonical(result) != canonical(old): raise ValueError('native program result differs from original counters')
        return request_or_result
    options = dict(request['options'])
    directory, run = run_native(config, _ProgramFiles(document, root), **options)
    try:
        run['status'] = 'complete'
        evaluation = _consume(request, document, config, directory, run)
    except ValueError as error:
        run.update(status='invalid_result',error=str(error))
        (directory/'run.json').write_bytes(canonical(run)+b'\n')
        raise ValueError(str(error)+'; native evidence: '+str(directory)) from error
    (directory/'run.json').write_bytes(canonical(run)+b'\n')
    result = {'schema':PROGRAM_RESULT_SCHEMA,'request':request,'evaluation':evaluation}
    _write(directory/'RESULT.json', result)
    return result

