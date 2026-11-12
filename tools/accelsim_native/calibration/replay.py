"""Replay a sealed offline plan and synthetic/empty observations, file reads only."""
import argparse
import json
from pathlib import Path
from common import read_json, write_json
from dryrun import external, validate_bundle
from solver import seal, solve
from arch_explorer.components.accelsim.capabilities import preflight_cubin


def replay(plan, observations):
    if seal(plan) != plan.get('design_sha256'):
        raise ValueError('design freeze changed')
    for row in plan['source_files'] + plan['offline_toolchain_dependencies'] + [plan['cpu_oracle_binary'], plan['engine'], plan['raw_record_schema']]:
        external(row)
    build = external(plan['build'])
    b, cases = validate_bundle(build.parent)
    if sorted(cases, key=lambda c: c['id']) != plan['cases']:
        raise ValueError('registered compiled/input/oracle closure differs from plan')
    native = preflight_cubin(build.parent / b['candidates'][0]['cubin']['path'], engine=plan['engine']['path'])
    if native != plan['public_static_preflight']:
        raise ValueError('fixed native identity/capability changed')
    external(plan['history']['history_check'])
    for name in ('closed_history', 'static_history'):
        row = plan['history'][name]
        from common import sha
        if sha(Path(row['path'])) != row['sha256']:
            raise ValueError('closed history changed')
    return solve(plan, observations)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('plan', type=Path)
    parser.add_argument('observations', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('new output required; preserve refusals')
    try:
        result = replay(read_json(args.plan), read_json(args.observations))
    except (ValueError, OSError, KeyError) as error:
        result = {'status': 'refused', 'reason': str(error), 'hardware_accuracy': 'blocked'}
    write_json(args.output, result)
    print(json.dumps(result))
    return 2 if result['status'] == 'refused' else 0


if __name__ == '__main__':
    raise SystemExit(main())
