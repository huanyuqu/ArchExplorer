# Concrete Accel-Sim implementation; owned by the component provider.
"""V2 binding of complete dynamic trace facts; request schema stays stable."""
from pathlib import Path
import json
import re
import shutil
from arch_explorer.components.accelsim.profile import ACCELSIM_SHA,GPGPUSIM_SHA,PROFILE
from arch_explorer.components.accelsim.trace import exact,_text,_integer,_geometry,_sha256,_mapping_spec,_bind_facts,identity,same_json,digest,WORKLOAD_FIELDS
from arch_explorer.components.accelsim.trace_stream import facts
from arch_explorer.components.accelsim.native_isa import require_adapter_isa

SCHEMA='arch-explorer.accelsim-trace.v2'

def _params(p,where,reference=False):
    exact(p,WORKLOAD_FIELDS|({'accelsim'} if reference else set()),where)
    if p['op']!='trace_kernel':raise ValueError(where+': requires trace_kernel')
    _text(p['kernel'],where+'.kernel')
    for key in ['grid','block']:_geometry(p[key],where+'.'+key)


def load_v2(workload,mapping,path,manifest,raw):
    _params(workload.params,'workload.params',True);_mapping_spec(mapping.to_dict(),'mapping')
    exact(manifest,{'schema','format','isa','tracer_version','source','engine','workload','mapping','commands','kernels','scope'},'trace v2 manifest')
    require_adapter_isa(manifest['isa'])
    if manifest['format'] not in ('traceg','tracez') or type(manifest['tracer_version']) is not int or manifest['tracer_version'] not in (3,6):raise ValueError('adapter: v2 validates ordinary traceg/tracez v3 or v6 only')
    if manifest['engine']!={'accelsim':ACCELSIM_SHA,'gpgpusim':GPGPUSIM_SHA,'profile':PROFILE}:raise ValueError('trace v2 fixed engine/profile mismatch')
    exact(manifest['workload'],{'kind','params'},'manifest.workload');_params(manifest['workload']['params'],'manifest.workload.params');_mapping_spec(manifest['mapping'],'manifest.mapping')
    if manifest['workload']['kind']!='operator' or manifest['scope']!='single-kernel-device-cycles' or not same_json(manifest['workload'],identity(workload)) or not same_json(manifest['mapping'],mapping.to_dict()):raise ValueError('trace v2 workload/mapping/scope identity mismatch')
    exact(manifest['source'],{'kind','gpu','toolchain','description'},'source')
    for k,v in manifest['source'].items():_text(v,'source.'+k)
    if manifest['source']['kind'] not in ('hardware','synthetic') or (manifest['source']['kind']=='synthetic' and manifest['source']['gpu']!='none') or (manifest['source']['kind']=='hardware' and manifest['source']['gpu']=='none'):raise ValueError('trace source GPU/kind inconsistent')
    kernels=manifest['kernels']
    if type(kernels) is not list or len(kernels)!=1:raise ValueError('v2 supports one complete kernel, not application/concurrency/copies')
    k=kernels[0];exact(k,{'file','sha256','original_sha256','name','grid','block','stream','thread_instructions','ctas','facts'},'kernel v2')
    _text(k['name'],'kernel.name')
    for key in ('grid','block'):_geometry(k[key],'kernel.'+key)
    for key in ('stream','ctas','thread_instructions'):_integer(k[key],'kernel.'+key,minimum=0 if key=='stream' else 1)
    for key in ('sha256','original_sha256'):_sha256(k[key],'kernel.'+key)
    if type(k['file']) is not str or not re.fullmatch(r'kernel-[A-Za-z0-9_-]+\.'+manifest['format'],k['file']):raise ValueError('v2 requires a plain original kernel basename')
    actual,artifact=facts(path.parent/k['file'])
    if artifact.sha256!=k['sha256'] or k['sha256']!=k['original_sha256']:raise ValueError('trace v2 hash/original identity mismatch')
    if not same_json(actual,k['facts']):raise ValueError('trace v2 actual dynamic facts differ from manifest')
    for key in ('name','grid','block','stream','ctas','thread_instructions'):
        if type(k[key]) is not type(actual[key]) or k[key]!=actual[key]:raise ValueError('trace v2 actual '+key+' differs from kernel claim')
    if actual['tracer_version']!=manifest['tracer_version'] or actual['isa']!=manifest['isa']:raise ValueError('trace v2 actual version/ISA mismatch')
    for p,fields,where in [(workload.params,{'kernel':'name','grid':'grid','block':'block'},'workload.params'),(manifest['workload']['params'],{'kernel':'name','grid':'grid','block':'block'},'manifest.workload.params'),(mapping.params,{'grid':'grid','block':'block'},'mapping.params'),(manifest['mapping']['params'],{'grid':'grid','block':'block'},'manifest.mapping.params')]:_bind_facts(p,fields,actual,where)
    exact(manifest['commands'],{'file','sha256'},'commands');_sha256(manifest['commands']['sha256'],'commands.sha256')
    if manifest['commands']['file']!='kernelslist.g':raise ValueError('commands must be kernelslist.g')
    commands=path.parent/'kernelslist.g'
    if commands.stat().st_size>4096:raise ValueError('command list exceeds single-kernel bound')
    data=commands.read_bytes()
    if digest(data)!=manifest['commands']['sha256'] or data!=(k['file']+'\n').encode():raise ValueError('commands do not name exactly the complete kernel')
    return manifest,artifact,digest(raw),actual


def create_request(trace_path, directory, *, source, trace_id, compiler, engine=None, adf=None):
    """Bind an external complete trace's observed facts to a stable public request.

    ``source`` is supplied provenance, never a claim independently certified by
    this helper. Real traces are external resources and are not packaged here.
    """
    from arch_explorer.adf import MappingSpec, WorkloadSpec
    from arch_explorer.components.accelsim.scenarios import sm90_adf
    trace_path, directory = Path(trace_path).resolve(), Path(directory).resolve()
    actual, artifact = facts(trace_path)
    if not re.fullmatch(r'kernel-[A-Za-z0-9_-]+\.(traceg|tracez)', trace_path.name):
        raise ValueError('external trace needs a plain kernel basename')
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / trace_path.name
    if destination != trace_path:
        with trace_path.open('rb') as src, destination.open('xb') as dst:
            shutil.copyfileobj(src, dst, 2**20)
        from arch_explorer.components.accelsim.trace_stream import file_digest
        if file_digest(destination) != artifact.sha256:
            raise ValueError('external trace changed during request creation')
    params = dict(op='trace_kernel', kernel=actual['name'], grid=actual['grid'], block=actual['block'])
    workload = WorkloadSpec('operator', params)
    mapping = MappingSpec('accelsim_trace', dict(trace_id=trace_id, compiler=compiler,
                                               grid=actual['grid'], block=actual['block']))
    commands = (trace_path.name+'\n').encode()
    manifest = dict(schema=SCHEMA, format=trace_path.suffix[1:], isa=actual['isa'],
        tracer_version=actual['tracer_version'], source=source,
        engine=dict(accelsim=ACCELSIM_SHA, gpgpusim=GPGPUSIM_SHA, profile=PROFILE),
        workload=identity(workload), mapping=mapping.to_dict(),
        commands=dict(file='kernelslist.g', sha256=digest(commands)),
        kernels=[dict(file=trace_path.name, sha256=artifact.sha256, original_sha256=artifact.sha256,
                      **{key: actual[key] for key in ('name','grid','block','stream','ctas','thread_instructions')},
                      facts=actual)], scope='single-kernel-device-cycles')
    raw = (json.dumps(manifest, indent=2)+'\n').encode()
    (directory/'kernelslist.g').write_bytes(commands)
    manifest_path = directory/'trace-manifest.json'
    manifest_path.write_bytes(raw)
    workload.params['accelsim'] = dict(manifest=str(manifest_path), sha256=digest(raw))
    request = dict(schema='arch-explorer.accelsim-request.v1',
        adf=(adf if adf is not None else sm90_adf()).to_dict(), workload=workload.to_dict(),
        mapping=mapping.to_dict(), options=dict(engine=str(Path(engine).resolve()) if engine else None,
                                               output_dir=str(directory/'runs')))
    (directory/'request.json').write_text(json.dumps(request, indent=2)+'\n')
    return request
