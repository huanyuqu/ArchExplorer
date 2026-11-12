"""Observe the existing public searcher using frozen CORE27 block config."""
from pathlib import Path
import json
import sys

from operator_attention_search_consumer import main


def consume(directory,dtype):
    def configuration():
        # Imported only after the observer patches the public APIs, preventing
        # module-level aliases from hiding real compile/prepare/evaluate calls.
        from arch_explorer.archs.transformer import block_search_config
        return block_search_config(dtype=dtype)

    main(directory,'block-'+dtype,config_factory=configuration)
    path=Path(directory)
    observed=json.loads((path/'OBSERVATIONS.json').read_text())
    best=json.loads((path/'BEST.json').read_text())
    calls=observed['calls']['compile']
    assert calls and all(row['workload']['kind']=='transformer_block' and
                         row['workload']['params']['dtype']==dtype for row in calls)
    for row in best['best']['workloads']:
        saved=row['compiled_result']
        actual=saved['request']['compilation']['source']
        assert actual['workload']['params']['dtype']==dtype
        assert row['mapping']==actual['mapping']
    (path/'BLOCK-SOURCE-OBSERVATION.json').write_text(json.dumps(dict(
        dtype=dtype,actual_public_workload_kind='transformer_block',
        observed_real_compile_calls=len(calls),search_is_existing_public_entry=True),indent=2)+'\n')


if __name__=='__main__':
    consume(sys.argv[1],sys.argv[2])
