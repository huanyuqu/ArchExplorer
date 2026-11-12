"""Read-only comparison of two independently executed native program results."""
from arch_explorer.components.accelsim import component_execution as execution

COMPARISON_SCHEMA = 'arch-explorer.accelsim-component-comparison.v1'


def _differences(a, b, path=''):
    if type(a) is not type(b):
        return [{'path': path, 'components': a, 'oracle': b}]
    if isinstance(a, dict):
        if set(a) != set(b):
            return [{'path': path, 'components_keys': sorted(a), 'oracle_keys': sorted(b)}]
        return [row for key in a for row in _differences(a[key], b[key], path+'/'+key)]
    if isinstance(a, list):
        if len(a) != len(b):
            return [{'path': path, 'components_count': len(a), 'oracle_count': len(b)}]
        return [row for i, (left, right) in enumerate(zip(a, b))
                for row in _differences(left, right, path+'/'+str(i))]
    return [] if a == b else [{'path': path, 'components': a, 'oracle': b}]


def compare_results(component_result, oracle_result):
    from arch_explorer.simulator.api import RESULT_SCHEMA, evaluate_request
    for value, provider in ((component_result, 'accelsim_components'), (oracle_result, 'accelsim')):
        if (not isinstance(value, dict) or value.get('schema') != RESULT_SCHEMA
                or value.get('provider') != {'id': provider, 'version': '1'}):
            raise ValueError('comparison requires complete public component and oracle results')
        evaluate_request(value)  # Strict raw reconstruction; the result branch never launches.
    a, b = component_result['evaluation']['details'], oracle_result['evaluation']['details']
    ar, br = component_result['request'], oracle_result['request']
    if a.get('mode') != 'components' or b.get('mode') != 'oracle':
        raise ValueError('comparison mode mismatch')
    for key in ('adf', 'workload', 'mapping', 'options'):
        if execution.canonical(ar[key]) != execution.canonical(br[key]):
            raise ValueError('comparison inputs differ: '+key)
    for key in ('engine_identity', 'config_sha256', 'adf_sha256', 'program_semantic_sha256',
                'outputs', 'numeric_checks', 'kernel_count', 'complete_operator_count'):
        if execution.canonical(a[key]) != execution.canonical(b[key]):
            raise ValueError('comparison program/source/config/numerical identity differs: '+key)
    if (a['evidence'] == b['evidence']
            or (a['run']['pid'], a['run']['started_monotonic_ns']) ==
               (b['run']['pid'], b['run']['started_monotonic_ns'])
            or a['run']['files'] != b['run']['files']):
        raise ValueError('comparison requires distinct processes/evidence and identical complete staged inputs')
    counter_differences = _differences(a['native'], b['native'], '/native')
    comparison_fields = ('kernels', 'foreground_final', 'drain_ticks', 'drain_core_cycles',
                         'final', 'ports', 'lifecycle', 'source', 'clock_periods')
    def mechanism_view(details):
        view = {key: details['metrics'][key] for key in comparison_fields}
        if details['metrics']['schema']=='arch-explorer.accelsim-component-metrics.v2':
            view['ports']=details['metrics']['oracle_observations']['ports']
        # Port directory order is not transaction order. Duplicate/unknown
        # wires have already been rejected by the raw-evidence reader.
        view['ports'] = sorted(view['ports'], key=lambda p: tuple(p[k] for k in
            ('source', 'source_port', 'destination', 'destination_port')))
        return view
    mechanism_differences = _differences(
        mechanism_view(a), mechanism_view(b), '/mechanisms')
    return {'schema': COMPARISON_SCHEMA,
            'status': 'failed' if counter_differences or mechanism_differences else 'passed',
            'inputs': {key: ar[key] for key in ('adf', 'workload', 'mapping', 'options')},
            'identity': {key: a[key] for key in ('engine_identity', 'config_sha256', 'adf_sha256',
                                               'program_semantic_sha256')},
            'results': {'components': component_result['binding'], 'oracle': oracle_result['binding']},
            'evidence': {'components': a['evidence'], 'oracle': b['evidence']},
            'native': {'components': a['native'], 'oracle': b['native']},
            'counter_differences': counter_differences, 'mechanism_differences': mechanism_differences,
            'observation_contract': 'v2 retains actual wire lengths and all allocations; common v1 port observations and wire lifecycles compare exactly, without changing oracle artifacts',
            'wire_observation_differences': _differences(a['metrics']['ports'],b['metrics']['ports'],'/ports'),
            'system_drained': True,
            'scope': 'exact native software consistency for this complete bound reference program',
            'hardware_accuracy': 'not validated'}
