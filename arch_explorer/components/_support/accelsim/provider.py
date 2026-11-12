"""Public joint-domain provider; descriptor imports never load an engine."""
from copy import deepcopy
import json
import math
from pathlib import Path
from arch_explorer.components.accelsim.component_provider import ORACLE_REQUEST, ORACLE_RESULT

REQUEST_SCHEMA = 'arch-explorer.accelsim-provider-request.v1'
RESULT_SCHEMA = 'arch-explorer.accelsim-provider-result.v1'
PROGRAM_REQUEST = 'arch-explorer.accelsim-program-request.v1'
PROGRAM_RESULT = 'arch-explorer.accelsim-program-result.v1'
TRACE_REQUEST = 'arch-explorer.accelsim-request.v1'
TRACE_RESULT = 'arch-explorer.accelsim-result.v1'
_REQUEST_FIELDS = {'schema', 'adf', 'workload', 'mapping', 'options', 'output_dir', 'native_request'}
_RESULT_FIELDS = {'schema', 'request', 'evaluation', 'native_result'}


def _copy(value):
    return json.loads(json.dumps(value, sort_keys=True, allow_nan=False))


def _same(left, right):
    return json.dumps(left, sort_keys=True, allow_nan=False) == json.dumps(right, sort_keys=True, allow_nan=False)


def _options(value):
    if not isinstance(value, dict) or set(value) - {'engine', 'timeout_s', 'max_cycles', 'memory_limit_mb'}:
        raise ValueError('native options only select engine and bounded runtime limits')
    value = _copy(value)
    timeout = value.get('timeout_s', 120)
    if type(timeout) not in (int, float) or not math.isfinite(timeout) or not 0 < timeout <= 3600:
        raise ValueError('timeout_s must be finite in (0, 3600]')
    for key, default, minimum, maximum in (
        ('max_cycles', 10000000, 1, 1000000000),
        ('memory_limit_mb', 4096, 512, 16384),
    ):
        item = value.get(key, default)
        if type(item) is not int or not minimum <= item <= maximum:
            raise ValueError(key+' outside native runtime bounds')
    if value.get('engine') is not None and (not isinstance(value['engine'], str) or not value['engine']):
        raise ValueError('engine must be a manifest path or null')
    return value


def _native_options(options, output, *, program):
    return {
        'engine': None if options.get('engine') is None else str(Path(options['engine']).resolve()),
        'output_dir': str(output),
        'timeout_s': options.get('timeout_s', 120),
        'max_cycles': options.get('max_cycles', 10000000 if program else 1000000),
        'memory_limit_mb': options.get('memory_limit_mb', 4096),
    }


def _checked_request(payload):
    from arch_explorer.components.accelsim.trace import exact
    exact(payload, _REQUEST_FIELDS, 'native provider request')
    if payload['schema'] != REQUEST_SCHEMA:
        raise ValueError('unsupported native provider request version')
    options = _options(payload['options'])
    root = Path(payload['output_dir'])
    if not root.is_absolute() or str(root.resolve()) != str(root):
        raise ValueError('absolute canonical native provider output directory required')
    native = payload['native_request']
    if not isinstance(native, dict) or native.get('schema') not in (PROGRAM_REQUEST, TRACE_REQUEST):
        raise ValueError('explicit native request required')
    program = native['schema'] == PROGRAM_REQUEST
    exact(native, {'schema', 'adf', 'workload', 'mapping', 'options'} | ({'program'} if program else set()),
          'nested native request')
    for key in ('adf', 'workload', 'mapping'):
        if not _same(native[key], payload[key]):
            raise ValueError('nested native '+key+' differs from public input')
    if not _same(native['options'], _native_options(options, root/'native', program=program)):
        raise ValueError('nested native runtime options differ from original options')
    if program:
        if native.get('program', {}).get('path') != str(root/'PROGRAM.json'):
            raise ValueError('native program belongs to a different provider output directory')
    return native


def _native_evaluate(payload):
    if payload.get('schema') in (PROGRAM_REQUEST, PROGRAM_RESULT):
        from arch_explorer.components.accelsim.program import evaluate_request
    elif payload.get('schema') in (TRACE_REQUEST, TRACE_RESULT):
        from arch_explorer.components.accelsim.backend import evaluate_request
    else:
        raise ValueError('unsupported registered native schema')
    return evaluate_request(payload)


def evaluate_backend(adf, workload, mapping, **options):
    """Compatibility backend selector; public provider registration owns it."""
    from arch_explorer.components.accelsim.backend import evaluate_accelsim
    return evaluate_accelsim(adf, workload, mapping, **options)


class AccelSimProvider:
    id = 'accelsim'
    version = '1'
    component_types = (
        'accelsim_sm', 'accelsim_sm120_mechanism', 'accelsim_l1_shared',
        'accelsim_icnt', 'accelsim_l2', 'accelsim_dram',
    )
    execution_domain = 'joint'
    capabilities = {
        'execution': 'one complete native program in one external CPU process',
        'components': 'SM/L1/shared/interconnect/L2/DRAM form one indivisible execution domain',
        'frontend': ['registered workload graph/reference and accelsim target implementation', 'single-trace-kernel'],
        'program_dtype': ['fp32', 'int32'],
        'cross_domain_composition': False,
        'hardware_accuracy': 'not validated',
    }
    request_schemas = (REQUEST_SCHEMA, PROGRAM_REQUEST, TRACE_REQUEST, ORACLE_REQUEST)
    result_schemas = (RESULT_SCHEMA, PROGRAM_RESULT, TRACE_RESULT, ORACLE_RESULT)

    def prepare_request(self, adf, workload, mapping, *, output_dir, options=None):
        from arch_explorer.components.accelsim.config import compile_adf
        from arch_explorer.components.accelsim.trace import load_trace
        raw_options = _options({} if options is None else options)
        from arch_explorer.components.accelsim.component_execution import is_component_engine
        if is_component_engine(raw_options.get('engine')):
            from arch_explorer.components.accelsim.component_provider import ORACLE
            return ORACLE.prepare_request(adf, workload, mapping, output_dir=output_dir, options=raw_options)
        # Validate the complete joint topology and requested role before writing.
        compile_adf(adf)
        root = Path(output_dir).resolve()
        program = mapping.template == 'accelsim_program'
        if program:
            from arch_explorer.components.accelsim.program import prepare_request
            native = prepare_request(adf, workload, mapping, output_dir=root,
                engine=raw_options.get('engine'),
                options={k: v for k, v in raw_options.items() if k != 'engine'})
        else:
            load_trace(workload, mapping)
            if root.exists() and any(root.iterdir()):
                raise ValueError('native provider preparation requires a new artifact namespace')
            root.mkdir(parents=True, exist_ok=True)
            native = {'schema': TRACE_REQUEST, 'adf': adf.to_dict(),
                'workload': workload.to_dict(), 'mapping': mapping.to_dict(),
                'options': _native_options(raw_options, root/'native', program=False)}
        payload = {'schema': REQUEST_SCHEMA, 'adf': adf.to_dict(), 'workload': workload.to_dict(),
            'mapping': mapping.to_dict(), 'options': raw_options, 'output_dir': str(root),
            'native_request': native}
        _checked_request(payload)
        return _copy(payload)

    def assess_request(self, payload):
        if payload.get('schema') in (ORACLE_REQUEST, ORACLE_RESULT):
            from arch_explorer.components.accelsim.component_provider import ORACLE
            return ORACLE.assess_request(payload)
        try:
            if payload.get('schema') == RESULT_SCHEMA:
                self._replay(payload)
                return {'supported': True, 'gaps': [], 'execution_domain': self.execution_domain}
            if payload.get('schema') in (PROGRAM_RESULT, TRACE_RESULT):
                _native_evaluate(payload)  # Strict read-only result branch.
                return {'supported': True, 'gaps': [], 'execution_domain': self.execution_domain}
            native = _checked_request(payload) if payload.get('schema') == REQUEST_SCHEMA else payload
            if native.get('schema') == PROGRAM_REQUEST:
                from arch_explorer.components.accelsim.program import assess_request
                result = assess_request(native)
                result['supported'] = result['status'] == 'supported'
                return result
            if native.get('schema') == TRACE_REQUEST:
                from arch_explorer.components.accelsim.protocol import consume_request
                from arch_explorer.components.accelsim.capabilities import assess
                data = consume_request(native)
                result = assess(data['config'], data['manifest'], data['actual'],
                                engine=native['options']['engine'])
                return {'supported': result['status'] == 'passed',
                        'gaps': list(result['blockers']), 'preflight': result}
            raise ValueError('unsupported registered native request schema')
        except (ValueError, KeyError, TypeError, OSError, ImportError) as error:
            return {'supported': False, 'gaps': [{'kind': 'native-domain', 'reason': str(error)}]}

    def _replay(self, payload):
        from arch_explorer.components.accelsim.trace import exact
        exact(payload, _RESULT_FIELDS, 'native provider result')
        native = _checked_request(payload['request'])
        result = payload['native_result']
        if not isinstance(result, dict) or not _same(result.get('request'), native):
            raise ValueError('native result differs from complete provider request')
        if result.get('schema') != (PROGRAM_RESULT if native['schema'] == PROGRAM_REQUEST else TRACE_RESULT):
            raise ValueError('native result/request schema mismatch')
        if not _same(result.get('evaluation'), payload['evaluation']):
            raise ValueError('provider evaluation differs from native result')
        replayed = _native_evaluate(result)
        if not _same(replayed, result):
            raise ValueError('native read-only replay changed the retained result')
        return payload

    def evaluate_request(self, payload):
        if payload.get('schema') in (ORACLE_REQUEST, ORACLE_RESULT):
            from arch_explorer.components.accelsim.component_provider import ORACLE
            return ORACLE.evaluate_request(payload)
        if payload.get('schema') == RESULT_SCHEMA:
            return self._replay(payload)
        if payload.get('schema') == REQUEST_SCHEMA:
            native = _checked_request(payload)
            result = _native_evaluate(native)
            if not _same(result.get('request'), native):
                raise ValueError('native execution did not retain the complete input request')
            return {'schema': RESULT_SCHEMA, 'request': deepcopy(payload),
                    'evaluation': result['evaluation'], 'native_result': result}
        return _native_evaluate(payload)


PROVIDER = AccelSimProvider()
