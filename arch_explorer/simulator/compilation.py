"""Compilation envelope composition over the existing provider lifecycle."""
from pathlib import Path

from arch_explorer.workload.compilation import canonical, seal, check, validate_compilation

REQUEST_SCHEMA = 'arch-explorer.compiled-request.v3'
RESULT_SCHEMA = 'arch-explorer.compiled-result.v3'


def prepare_compiled(compilation, *, output_dir, options=None, registry=None):
    from arch_explorer.simulator.api import prepare_request
    from arch_explorer.adf.model import from_dict
    from arch_explorer.adf.evaluation import WorkloadSpec, MappingSpec
    compilation = validate_compilation(compilation,registry=registry)
    lower = compilation['lowered']
    request = prepare_request(from_dict(compilation['source']['adf']),
        WorkloadSpec(**lower['workload']),MappingSpec(**lower['mapping']),
        output_dir=output_dir,provider=lower['provider'],options=options)
    result = seal(dict(schema=REQUEST_SCHEMA,compilation=compilation,request=request))
    path = Path(output_dir).resolve()/'COMPILATION-REQUEST.json'
    path.write_bytes(canonical(result)+b'\n')
    return result


def _load_request(value):
    check(value,{'schema','compilation','request','binding'},REQUEST_SCHEMA)
    compilation = validate_compilation(value['compilation'])
    request,lower = value['request'],compilation['lowered']
    for key,expected in (('adf',compilation['source']['adf']),('workload',lower['workload']),('mapping',lower['mapping'])):
        if request.get(key) != expected:
            raise ValueError('compiled request '+key+' differs from bound target input')
    if request.get('provider',{}).get('id') != lower['provider']:
        raise ValueError('compiled request provider differs')
    return request


def assess_compiled(value):
    from arch_explorer.simulator.api import assess_request
    try:
        if value.get('schema') == RESULT_SCHEMA:
            check(value,{'schema','request','result','evaluation','binding'},RESULT_SCHEMA)
            _load_request(value['request'])
            if value['result'].get('request') != value['request']['request']:
                raise ValueError('compiled result belongs to another request')
            if value['evaluation'] != value['result'].get('evaluation'):
                raise ValueError('compiled result evaluation differs')
            return assess_request(value['result'])
        return assess_request(_load_request(value))
    except (ValueError,TypeError,KeyError,NotImplementedError,OSError) as error:
        return {'supported':False,'gaps':[f'{type(error).__name__}: {error}']}


def evaluate_compiled(value):
    from arch_explorer.simulator.api import evaluate_request
    if value.get('schema') == RESULT_SCHEMA:
        assessment = assess_compiled(value)
        if not assessment['supported']:
            raise ValueError('unsupported compiled result: '+str(assessment['gaps']))
        read = evaluate_request(value['result'])
        if canonical(read) != canonical(value['result']):
            raise ValueError('compiled replay differs from original result')
        return value
    request = _load_request(value)
    result = evaluate_request(request)
    return seal(dict(schema=RESULT_SCHEMA,request=value,result=result,evaluation=result['evaluation']))
