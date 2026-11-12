"""Public provider for actual component assembly and the original-cycle oracle."""
from arch_explorer.components._support.paths import package_root
from copy import deepcopy
from pathlib import Path

from arch_explorer.adf import Evaluation, from_dict
from arch_explorer.components.accelsim.component_topology import compile_topology
from arch_explorer.components.accelsim import component_execution as execution

REQUEST = 'arch-explorer.accelsim-components-request.v1'
RESULT = 'arch-explorer.accelsim-components-result.v1'
ORACLE_REQUEST = 'arch-explorer.accelsim-component-oracle-request.v1'
ORACLE_RESULT = 'arch-explorer.accelsim-component-oracle-result.v1'
_FIELDS = {'schema', 'mode', 'adf', 'workload', 'mapping', 'options', 'output_dir',
           'program_request', 'topology', 'engine_identity', 'sources'}
_SOURCE_PATHS = (
    'adf/model.py', 'adf/dynamic.py', 'simulator/api.py', 'simulator/dynamic/runtime.hpp',
    'components/accelsim/component_topology.py', 'components/accelsim/componentization.py',
    'components/accelsim/component_execution.py', 'components/accelsim/component_provider.py',
    'components/accelsim/standalone_engine.py', 'components/accelsim/native_sdk.py',
    'components/accelsim/component_comparison.py', 'components/accelsim/program_stats.py',
    'components/accelsim/profile.py', 'components/accelsim/variant.py',
)


def sources():
    root = package_root()
    from arch_explorer.components.accelsim.program_sources import sources as program_sources
    result = program_sources()
    result.update({name: execution.digest((root/name).read_bytes()) for name in _SOURCE_PATHS})
    return result


def _same(left, right):
    return execution.canonical(left) == execution.canonical(right)


def _frontend_options(options, output_dir):
    return dict(engine=None, output_dir=str(Path(output_dir)/'frontend'/'native'),
                **execution.limits(options))


def _checked(payload, mode, *, read_only=False):
    if not isinstance(payload, dict) or set(payload) != _FIELDS:
        raise ValueError('exact component provider request fields required')
    if payload['schema'] != (REQUEST if mode == 'components' else ORACLE_REQUEST) or payload['mode'] != mode:
        raise ValueError('component provider request schema/mode differs')
    options = execution.options(payload['options'])
    root = Path(payload['output_dir'])
    if not root.is_absolute() or str(root.resolve()) != str(root):
        raise ValueError('canonical absolute component output directory required')
    engine, engine_ref = execution.load_engine(options['engine'], read_only=read_only)
    if mode == 'components' and engine['schema'] != 'arch-explorer.accelsim-component-engine.v2':
        raise ValueError('component execution requires independently created standalone factories; legacy borrowed-host images are read-only artifacts')
    from arch_explorer.components._support.accelsim.layout_history import source_era
    if payload['engine_identity'] != engine_ref:
        raise ValueError('component engine or provider source binding differs')
    source_era(payload['sources'], sources(), 'component', read_only=read_only)
    topology = compile_topology(from_dict(payload['adf']))
    if execution.check_ref(payload['topology']).decode() != topology:
        raise ValueError('component topology file differs from explicit ADF connections')
    front = payload['program_request']
    if not isinstance(front, dict) or front.get('schema') != 'arch-explorer.accelsim-program-request.v1':
        raise ValueError('complete reference program frontend required')
    for key in ('adf', 'workload', 'mapping'):
        if not _same(front.get(key), payload[key]):
            raise ValueError('component frontend differs from public '+key)
    if not _same(front.get('options'), _frontend_options(options, root)):
        raise ValueError('component frontend runtime scope differs')
    from arch_explorer.components.accelsim.program import _load
    document, config, directory = _load(front, read_only=read_only)
    if document['assessment']['status'] != 'supported' or not document['kernels']:
        raise ValueError('unsupported/incomplete graph cannot execute as components')
    return document, config, directory, engine


def _evaluation(payload, document, config, stats, metrics, directory, receipt):
    from arch_explorer.components.accelsim.program import _comparison_view
    semantic = execution.digest(execution.canonical(_comparison_view(document, document['binding'])))
    provenance = [{key: row[key] for key in ('path', 'param', 'value', 'unit', 'source')}
                  for row in config.component_map]
    breakdown = [{'kernel': kernel['name'], 'operator': operator['operator'],
                  'cycles': kernel['cycles'], 'latency_ns': kernel['cycles']*1000/stats['core_mhz']}
                 for kernel, operator in zip(stats['kernels'], document['kernels'])]
    if stats['drain_core_cycles']:
        breakdown.append({'phase': 'physical_system_drain', 'cycles': stats['drain_core_cycles'],
                          'latency_ns': stats['drain_core_cycles']*1000/stats['core_mhz']})
    return Evaluation(stats['system_drained_latency_ns'], backend='accelsim_components' if payload['mode'] == 'components' else 'accelsim',
        provenance=provenance,
        breakdown=breakdown,
        warnings=['Complete reference-generated synthetic FP32 forward with synthetic weights and addresses.',
                  'Native software consistency comparison; hardware accuracy is not validated.'],
        details={'schema': 'arch-explorer.accelsim-component-evaluation.v1', 'mode': payload['mode'],
                 'program_sha256': payload['program_request']['program']['sha256'],
                 'program_semantic_sha256': semantic,
                 'complete_operator_count': len(document['operators']),
                 'operator_coverage': document['assessment']['operator_coverage'],
                 'kernel_count': len(document['kernels']), 'mapping': payload['mapping'],
                 'native': stats, 'metrics': metrics, 'engine_identity': payload['engine_identity'],
                 'config_sha256': config.sha256, 'adf_sha256': config.adf_sha256,
                 'component_map': config.component_map, 'numeric_checks': document['numeric_checks'],
                 'outputs': document['outputs'], 'scope': document['scope'],
                 'hardware_accuracy': {'status': 'not validated', 'mape': None, 'worst': None},
                 'evidence': str(directory), 'run': receipt}).to_dict()


class ComponentProvider:
    id = 'accelsim_components'
    version = '1'
    execution_domain = 'joint'
    component_types = ('accelsim_sm', 'accelsim_l1_shared', 'accelsim_icnt', 'accelsim_l2', 'accelsim_dram')
    capabilities = {
        'execution': 'explicit native component instances, dynamic transaction routing and clock phases',
        'frontend': 'registered workload graph/reference and accelsim target implementation',
        'program_dtype': ['fp32', 'int32'],
        'component_protocol': 'transaction.v1', 'hardware_accuracy': 'not validated',
        'topology': 'one chiplet, LRC disabled; full explicit six-edge connections',
        'runtime_bounds': {'timeout_s': 600, 'memory_limit_mb': 8192},
    }
    request_schemas = (REQUEST,)
    result_schemas = (RESULT,)
    mode = 'components'

    def prepare_request(self, adf, workload, mapping, *, output_dir, options=None):
        value = execution.options({} if options is None else options)
        engine, identity = execution.load_engine(value['engine'])
        if self.mode == 'components' and engine['schema'] != 'arch-explorer.accelsim-component-engine.v2':
            raise ValueError('new component requests require a standalone-factories-v1 image')
        topology = compile_topology(adf)
        root = Path(output_dir).resolve()
        if root.exists() and any(root.iterdir()):
            raise ValueError('component preparation requires a fresh artifact namespace')
        from arch_explorer.components.accelsim.program import prepare_request
        front = prepare_request(adf, workload, mapping, output_dir=root/'frontend', engine=None,
                                options=execution.limits(value))
        topology_ref = root/'topology.tsv'
        with topology_ref.open('xb') as stream:
            stream.write(topology.encode())
        payload = {'schema': REQUEST if self.mode == 'components' else ORACLE_REQUEST,
                   'mode': self.mode, 'adf': adf.to_dict(), 'workload': workload.to_dict(),
                   'mapping': mapping.to_dict(), 'options': value, 'output_dir': str(root),
                   'program_request': front, 'topology': execution.file_ref(topology_ref),
                   'engine_identity': identity, 'sources': sources()}
        _checked(payload, self.mode)
        return payload

    def assess_request(self, payload):
        try:
            if payload.get('schema') == (RESULT if self.mode == 'components' else ORACLE_RESULT):
                self._replay(payload)
            else:
                _checked(payload, self.mode)
            return {'supported': True, 'gaps': [], 'execution_domain': self.execution_domain,
                    'component_protocol': 'transaction.v1'}
        except (KeyError, ValueError, TypeError, OSError, ImportError) as error:
            return {'supported': False, 'gaps': [{'kind': 'component-domain', 'reason': str(error)}]}

    def _replay(self, result):
        if not isinstance(result, dict) or set(result) != {'schema', 'request', 'evaluation'}:
            raise ValueError('exact component result fields required')
        if result['schema'] != (RESULT if self.mode == 'components' else ORACLE_RESULT):
            raise ValueError('component result mode/version differs')
        payload = result['request']
        document, config, program_root, _ = _checked(payload, self.mode, read_only=True)
        directory = Path(result['evaluation']['details']['evidence'])
        receipt = execution.json_read(directory/'run.json')
        if receipt != result['evaluation']['details']['run']:
            raise ValueError('component retained process receipt changed')
        stats, metrics = execution.consume(payload, document, config, program_root, directory, receipt, read_only=True)
        rebuilt = _evaluation(payload, document, config, stats, metrics, directory, receipt)
        if not _same(rebuilt, result['evaluation']):
            raise ValueError('component evaluation differs from raw native facts')
        return result

    def evaluate_request(self, payload):
        if payload.get('schema') in (RESULT, ORACLE_RESULT):
            return self._replay(payload)
        document, config, program_root, _ = _checked(payload, self.mode)
        directory, receipt = execution.run(config, document, program_root,
            execution.check_ref(payload['topology']).decode(), payload['options']['engine'],
            Path(payload['output_dir'])/'native', execution.limits(payload['options']), self.mode)
        try:
            stats, metrics = execution.consume(payload, document, config, program_root, directory, receipt)
            evaluation = _evaluation(payload, document, config, stats, metrics, directory, receipt)
        except (ValueError, TypeError, KeyError, OSError) as error:
            execution.write(directory/'INVALID-RESULT.json', {'status': 'failed', 'reason': str(error)})
            raise ValueError(str(error)+'; component native evidence: '+str(directory)) from error
        result = {'schema': RESULT if self.mode == 'components' else ORACLE_RESULT,
                  'request': deepcopy(payload), 'evaluation': evaluation}
        execution.write(directory/'RESULT.json', result)
        return result


class OriginalCycleOracle(ComponentProvider):
    id = 'accelsim'
    request_schemas = (ORACLE_REQUEST,)
    result_schemas = (ORACLE_RESULT,)
    mode = 'oracle'


PROVIDER = ComponentProvider()
ORACLE = OriginalCycleOracle()


def evaluate_backend(adf, workload, mapping, **options):
    from arch_explorer.simulator.api import prepare_request, evaluate_request
    output_dir = options.pop('output_dir', None)
    if output_dir is None:
        raise ValueError('component native evaluation requires explicit output_dir')
    request = prepare_request(adf, workload, mapping, output_dir=output_dir,
                              provider=PROVIDER.id, options=options)
    return Evaluation(**evaluate_request(request)['evaluation'])
