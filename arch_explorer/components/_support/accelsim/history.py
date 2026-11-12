"""Read-only native trace result replay using the common fact validator."""
import json
from pathlib import Path

from arch_explorer.components.accelsim.protocol import consume_request, validate_prediction_data
from arch_explorer.components.accelsim.source_compat import adapter_source_era
from arch_explorer.components.accelsim.trace import digest, exact
from arch_explorer.components.accelsim.result_contract import RESULT_SCHEMAS


def replay_result(payload):
    exact(payload, {'schema', 'request', 'evaluation'}, 'native trace result')
    if payload['schema'] not in RESULT_SCHEMAS:
        raise ValueError('unsupported native trace result schema')
    current = consume_request(payload['request'])
    details = payload['evaluation']['details']
    directory = Path(details['evidence'])
    if (not directory.is_absolute()
            or directory.parent != Path(payload['request']['options']['output_dir']).resolve()):
        raise ValueError('native trace result has a different absolute output scope')
    file = directory/'run.json'
    if not file.is_file() or file.stat().st_size > 2**20:
        raise ValueError('missing/oversized native run receipt')
    run = json.loads(file.read_bytes())
    if run != details['run']:
        raise ValueError('embedded native trace run differs from retained original')
    # Direct result replay has no archive argument. It accepts the current full
    # adapter vector or an exact frozen full producer vector, never a subset.
    adapter_source_era(run.get('adapter_sources_sha256'))

    def retain(role, file, expected):
        if not file.is_file() or file.stat().st_size > 64*2**20:
            raise ValueError('missing/oversized retained native original: '+role)
        if digest(file.read_bytes()) != expected:
            raise ValueError('retained native original bytes differ: '+role)

    validate_prediction_data(payload, run, current, None,
                             run['adapter_sources_sha256'], directory, retain)
    return payload
