"""Observe existing public search with CORE28 config; no search implementation."""
import json
from pathlib import Path
import sys

from operator_attention_search_consumer import main


def consume(directory):
    def configuration():
        from arch_explorer.archs.multilayer import encoder_search_config
        return encoder_search_config(layers=2,dtype='bf16')
    main(directory,'multilayer',config_factory=configuration,mapping_template='compile_graph_v3')
    validate_saved_search(directory)


def validate_saved_search(directory,*,observation_dir=None):
    root=Path(directory)
    observations=json.loads((root/'OBSERVATIONS.json').read_text())
    best=json.loads((root/'BEST.json').read_text())
    calls=observations['calls']['compile'];assert calls
    for call in calls:
        assert call['workload']['kind']=='transformer_encoder'
        assert len(call['workload']['params']['inputs']['layers'])==2
        params=call['mapping']['params']
        assert call['mapping']['template']=='compile_graph_v3'
        assert set(params)=={'compute','memory','rules','operator_rules'}
        assert len({json.dumps(binding,sort_keys=True) for binding in params['operator_rules'].values()})>=2
    for attempt in best['evaluation_attempts']:
        assert 'requested_mapping' in attempt
        if attempt['phase']=='completed':assert attempt['effective_mapping']
    capacity_failures=[row for row in best['evaluation_attempts'] if not row['valid']
        and 'storage allocation' in str(row.get('error','')).lower()
        and '> available 1 bytes' in str(row.get('error','')).lower()]
    assert capacity_failures
    assert all(row['phase']=='compile' and not row['execution_attempted'] for row in capacity_failures)
    assert best['evaluations_used']==len(best['evaluation_attempts'])==6
    for row in best['best']['workloads']:
        compilation=row['compiled_result']['request']['compilation']
        assert row['mapping']==compilation['source']['mapping']
        effective=row['mapping']['params']['operator_rules']
        assert set(effective)==set(compilation['plan']['rules'])
        selected=[value for value in effective.values() if value['id']=='matmul.tiled']
        assert len({json.dumps(value,sort_keys=True) for value in selected})>=2
    (Path(observation_dir or root)/'INSTANCE-MAPPING-OBSERVATION.json').write_text(json.dumps(dict(
        existing_public_search_used=True,same_kind_instance_options_distinct=True,
        requested_and_effective_attempt_fields_preserved=True),indent=2)+'\n')


if __name__=='__main__':consume(sys.argv[1])
