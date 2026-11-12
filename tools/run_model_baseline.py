"""Bounded, auditable non-GPU experiments using ADF tools and catalog proposals.

No model execution, cost formula, or hardware-family selection lives here.
Run from an exported source/sdist root with the ArchExplorer wheel installed.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import random
import subprocess
import sys
import time

from arch_explorer.adf import SpacePoint, from_dict, walk
from arch_explorer.agent_env import ExplorationEnv
from arch_explorer.search.candidates import proposals


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def ok(reply):
    if 'error' in reply:
        raise ValueError(reply['error'])
    return reply['result']


def require_non_gpu(spec):
    """Reject removed predictor configurations before creating experiment output."""
    def visit(value):
        if isinstance(value, dict):
            if (value.get('preset') == 'gsm'
                    or value.get('component') in ('gpu_sm', 'gpu_kernel_service')):
                raise ValueError('The in-house GPU predictor was removed; use archexplorer demo for native GPU programs')
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)
    visit(spec)


def resources(adf):
    """Count physical multiplicities, including ancestor repeats (not tree nodes)."""
    nodes = list(walk(from_dict(adf)))
    multiplicities, rows = {}, []
    for path, node in nodes:
        count = node.repeat * multiplicities.get(path.rsplit('/', 1)[0], 1)
        multiplicities[path] = count
        if isinstance(node, SpacePoint):
            rows.append({'path': path, 'component': node.component, 'instances': count,
                         'params': {k: p.to_dict() for k, p in node.params.items()}})
    return rows


def identity(value, *, topology=False):
    """Ignore provenance text for duplicates; retain physical values."""
    if isinstance(value, list):
        return [identity(v, topology=topology) for v in value]
    if isinstance(value, dict):
        return {k: identity(v, topology=topology) for k, v in value.items()
                if k != 'source' and not (topology and k == 'params')}
    return value


def violations(state, limits):
    rows = resources(state['adf'])
    by_path = {r['path']: r for r in rows}
    errors = []
    if sum(r['instances'] for r in rows) > limits['max_point_instances']:
        errors.append('max_point_instances')
    capacity = sum(r['instances'] * r['params'].get('capacity', {}).get('value', 0) for r in rows)
    if capacity > limits['max_capacity_bytes']:
        errors.append('max_capacity_bytes')
    for path in limits['required_paths']:
        if path not in by_path:
            errors.append('required_path:' + path)
    for bound in limits.get('parameter_maxima', []):
        value = by_path.get(bound['path'], {}).get('params', {}).get(bound['param'])
        if value is None or value['unit'] != bound['unit'] or value['value'] > bound['max']:
            errors.append('parameter_bound:' + bound['path'] + '.' + bound['param'])
    return errors


def subprocess_replay(out, winner, timeout):
    checks = []
    commands = [('state', ['env', 'replay', str(out/'tools.jsonl'), '--output', str(out/'state-replay.json')])]
    if winner:
        commands.insert(0, ('best', ['evaluate', str(out/'best.json'), '--output', str(out/'best-replay.json')]))
    for name, args in commands:
        command = [sys.executable, '-m', 'arch_explorer', *args]
        start = time.monotonic()
        result = subprocess.run(command, cwd=out, capture_output=True, text=True, timeout=timeout)
        (out/f'{name}-stdout.txt').write_text(result.stdout)
        (out/f'{name}-stderr.txt').write_text(result.stderr)
        equal = False
        if result.returncode == 0:
            expected = json.loads((out/('best.json' if name == 'best' else 'state.json')).read_text())
            actual = json.loads((out/f'{name}-replay.json').read_text())
            equal = (expected['best'] == actual['best'] and expected.get('resource_constraints') == actual.get('resource_constraints')) if name == 'best' else expected == actual
        checks.append({'target': name, 'command': command, 'returncode': result.returncode,
                       'entire_object_equal': equal, 'wall_seconds': time.monotonic()-start})
        write(out/'replay.json', checks)
        if not equal:
            raise RuntimeError(f'{name} fresh-process replay failed; see {out}')
    return checks


def run(spec, output, *, max_evaluations=None, replay_timeout=3600):
    require_non_gpu(spec)
    if 'proposal_method' in spec:
        return run_bounded(spec, output, max_evaluations=max_evaluations, replay_timeout=replay_timeout)
    start = time.monotonic()
    out = Path(output).resolve()
    out.mkdir(parents=True, exist_ok=False)  # Never overwrite prior evidence.
    cfg = deepcopy(spec['config'])
    if len(cfg['workloads']) != 1 or cfg['workloads'][0].get('weight', 1) != 1:
        raise ValueError('Each baseline requires one unweighted complete workload')
    if max_evaluations is not None:
        if type(max_evaluations) is not int or not 1 <= max_evaluations <= cfg['max_evaluations']:
            raise ValueError('Debug budget must be positive and cannot increase frozen budget')
        cfg['max_evaluations'] = max_evaluations
    cfg['trace_path'] = str(out/'tools.jsonl')
    write(out/'experiment.json', spec)
    write(out/'config.json', cfg)
    write(out/'runtime.json', {'python': sys.version, 'platform': platform.platform(),
        'packages': sorted((d.metadata['Name'], d.version) for d in importlib.metadata.distributions()),
        'source_baseline': spec['source_baseline'], 'execution_scope': 'non-GPU ADF event simulation; not measured device latency',
        'runner_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
    env = ExplorationEnv(cfg)
    description = ok(env.describe_environment())
    ok(env.set_mappings(mappings=spec['initial_mappings']))
    initial = ok(env.get_state())
    write(out/'initial.json', initial)
    if violations(initial, spec['limits']):
        raise ValueError('Initial ADF violates frozen resource bounds')
    ok(env.checkpoint(name='initial'))
    best = None
    rows, seen = [], set()

    def trial(label, calls):
        nonlocal best
        if env.remaining == 0:
            return
        ok(env.restore(name='initial'))
        row = {'label': label, 'calls': calls, 'used_before': env.used}
        tick = time.monotonic()
        draft = deepcopy(spec['initial_mappings'])
        for call in calls:
            reply = env.call(call['tool'], call.get('args', {}))
            if 'error' in reply:
                row['error'] = reply['error']
                break
            if call['tool'] == 'set_mappings':
                draft = deepcopy(call['args']['mappings'])
        # Edits invalidate the Env draft. Restore the experiment's explicit
        # mapping through the public tool, never silently re-sample it.
        if 'error' not in row:
            reply = env.set_mappings(mappings=draft)
            if 'error' in reply:
                row['error'] = reply['error']
        state = ok(env.get_state())
        row['resource_violations'] = violations(state, spec['limits'])
        key = json.dumps(identity([state['adf'], state['mappings']]), sort_keys=True)
        row['structure_changed'] = identity(state['adf'], topology=True) != identity(initial['adf'], topology=True)
        row['adf_values_changed'] = identity(state['adf']) != identity(initial['adf'])
        row['mapping_changed'] = state['mappings'] != initial['mappings']
        row.update(fingerprint=state['fingerprint'], duplicate=key in seen,
                   resources=resources(state['adf']), mappings=state['mappings'], costs=state['costs'])
        write(out/f'candidate-{len(rows):02d}.json', {'adf':state['adf'], 'mappings':state['mappings']})
        if 'error' not in row and not row['resource_violations'] and not row['duplicate']:
            seen.add(key)
            reply = env.evaluate(optimize='none')
            if 'error' in reply:
                row['error'] = reply['error']
            else:
                candidate = reply['result']
                row['score'] = candidate['score']
                evaluation = candidate['workloads'][0]['evaluation']
                row['latency_ns'] = evaluation['latency_ns']
                trace = evaluation['details'].get('trace', [])
                row['accumulated_task_ns'] = sum(max(0, t['end_ns']-t['start_ns']) for t in trace
                                                if t['start_ns'] is not None and t['kind'] != 'storage')
                row['top_resource_utilization'] = sorted(evaluation['utilization'].items(), key=lambda x:x[1], reverse=True)[:10]
                value = candidate['score']['objective']
                if candidate['score']['feasible'] and value is not None and (best is None or value < best):
                    best = value
                    ok(env.checkpoint(name='best-feasible'))
        row.update(used_after=env.used, wall_seconds=time.monotonic()-tick, best_feasible_objective=best)
        rows.append(row)
        write(out/'trajectory.json', rows)
        print(json.dumps({k:row[k] for k in ('label','used_after','best_feasible_objective','wall_seconds')}), flush=True)

    trial('baseline', [])
    for experiment in spec['experiments']:
        trial(experiment['label'], experiment['calls'])
    # Use the framework's DEFAULT catalog/basic-edit generator unchanged. This
    # experiment enforces resource bounds before evaluation, never family quotas.
    rng = random.Random(cfg.get('random_seed', 0))
    for index in range(spec['generic_steps']):
        if env.remaining == 0:
            break
        choices = proposals(initial, description['components'])
        if not choices:
            break
        selected = rng.choice(choices)
        edits = selected if isinstance(selected, list) else [selected]
        trial(f'generic-{index}', [{'tool':'apply_edits', 'args':{'edits':edits}}])
    if best is not None:
        ok(env.restore(name='best-feasible'))
        ok(env.submit(path=str(out/'best.json'), selection='current'))
    else:
        write(out/'no-winner.json', {'reason':'No evaluated candidate satisfies every hard constraint',
                                    'backend_attempts':env.used})
    write(out/'attempts.json', env.attempts)
    write(out/'state.json', ok(env.get_state()))
    elapsed = time.monotonic()-start
    summary = {'status':'feasible_winner' if best is not None else 'no_feasible_winner',
        'official_graph_audit':'pending SIM18', 'source_baseline':spec['source_baseline'],
        'backend_attempts':env.used, 'budget':env.budget, 'remaining':env.remaining,
        'failed_backend_attempts':sum(not a['valid'] for a in env.attempts),
        'duplicate_proposals':sum(r['duplicate'] for r in rows),
        'resource_rejections':sum(bool(r['resource_violations']) for r in rows),
        'best_feasible_objective':best, 'experiment_wall_seconds':elapsed,
        'timing_note':'task durations overlap; accumulated task time is not end-to-end latency; wall time is Python simulation, not hardware measurement'}
    write(out/'summary.json', summary)
    summary['replay'] = subprocess_replay(out, best is not None, replay_timeout)
    summary['total_wall_seconds'] = time.monotonic()-start
    write(out/'summary.json', summary)
    write(out/'SHA256.json', {str(p.relative_to(out)):hashlib.sha256(p.read_bytes()).hexdigest()
                            for p in sorted(out.rglob('*')) if p.is_file() and p.name!='SHA256.json'})
    return summary


def run_bounded(spec, output, *, max_evaluations=None, replay_timeout=3600):
    """Paired finite generic proposals, always enforced by the real Env."""
    require_non_gpu(spec)
    from arch_explorer.search.candidates import catalog_proposals, constrained_proposals
    start = time.monotonic()
    out = Path(output).resolve()
    out.mkdir(parents=True, exist_ok=False)
    cfg = deepcopy(spec['config'])
    if len(cfg['workloads']) != 1 or cfg['workloads'][0].get('weight', 1) != 1:
        raise ValueError('Use one unweighted complete workload')
    if max_evaluations is not None:
        if type(max_evaluations) is not int or not 1 <= max_evaluations <= cfg['max_evaluations']:
            raise ValueError('Debug can only lower budget')
        cfg['max_evaluations'] = max_evaluations
    method, cap = spec['proposal_method'], spec['max_proposals']
    if method not in ('legacy', 'constrained') or type(cap) is not int or cap < 1:
        raise ValueError('Expected legacy/constrained and positive max_proposals')
    cfg['trace_path'] = str(out/'tools.jsonl')
    write(out/'experiment.json', spec)
    write(out/'config.json', cfg)
    write(out/'runtime.json', {'python':sys.version, 'source_baseline':spec['source_baseline'],
          'execution_scope':'non-GPU ADF event simulation; not measured device latency',
          'runner_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
    env = ExplorationEnv(cfg)
    description = ok(env.describe_environment())
    ok(env.set_mappings(mappings=spec['initial_mappings']))
    initial = ok(env.get_state())
    write(out/'initial.json', initial)
    ok(env.checkpoint(name='initial'))
    rows, seen, best = [], set(), None

    def trial(label, edits=None):
        nonlocal best
        ok(env.restore(name='initial'))
        row = {'label':label, 'edits':edits, 'used_before':env.used}
        tick = time.monotonic()
        reply = env.apply_edits(edits=edits) if edits is not None else env.get_state()
        if 'error' in reply:
            row.update(error=reply['error'], status='preflight_rejected')
        state = ok(env.get_state())
        row.update(resources=resources(state['adf']), costs=state['costs'],
                   structure_changed=identity(state['adf'],topology=True)!=identity(initial['adf'],topology=True))
        # Mapping is sampled by Env from the official request_space after a
        # structural edit. Explicit correlated multi-device drafts stay intact.
        key = json.dumps(identity(state['adf']), sort_keys=True)
        if 'error' not in row and key in seen:
            row['status'] = 'duplicate'
        elif 'error' not in row:
            seen.add(key)
            result = env.evaluate(optimize='mapping' if edits is not None else 'none')
            if 'error' in result:
                row.update(error=result['error'],status='backend_failed' if env.used>row['used_before'] else 'evaluation_preflight_rejected')
            else:
                c=result['result'];e=c['workloads'][0]['evaluation']
                row.update(status='evaluated',score=c['score'],mappings=c['mappings'],latency_ns=e['latency_ns'],
                           mapping_changed=c['mappings']!=spec['initial_mappings'],
                           top_resource_utilization=sorted(e['utilization'].items(),key=lambda x:x[1],reverse=True)[:10])
                if c['score']['feasible'] and c['score']['objective'] is not None and (best is None or c['score']['objective']<best):
                    best=c['score']['objective'];ok(env.checkpoint(name='best-feasible'))
        final=ok(env.get_state())
        write(out/f'candidate-{len(rows):03d}.json',{'adf':final['adf'],'mappings':final['mappings'],'resource_constraints':cfg['resource_constraints']})
        row.update(used_after=env.used,wall_seconds=time.monotonic()-tick,best_feasible_objective=best)
        rows.append(row);write(out/'trajectory.json',rows)
        print(json.dumps({k:row[k] for k in ('label','status','used_after','best_feasible_objective')}),flush=True)

    trial('baseline')
    rng=random.Random(cfg.get('random_seed',0));audit=[]
    if method=='constrained':
        choices=constrained_proposals(initial,description['components'],max_attempts=cap,rng=rng,audit=audit,registry=env.components)
    else:
        choices=catalog_proposals(initial,description['components']);rng.shuffle(choices);choices=choices[:cap]
    for i,choice in enumerate(choices):
        if env.remaining==0:
            break
        edits=choice if isinstance(choice,list) else [choice]
        trial(f'proposal-{i}',edits)
        if method=='legacy':
            audit.append({'edits':edits,'status':rows[-1]['status'],'error':rows[-1].get('error')})
    write(out/'proposal-audit.json',audit)
    if best is not None:
        ok(env.restore(name='best-feasible'));ok(env.submit(path=str(out/'best.json'),selection='current'))
    else:
        write(out/'no-winner.json',{'reason':'No feasible evaluation','backend_attempts':env.used})
    write(out/'attempts.json',env.attempts);write(out/'state.json',ok(env.get_state()))
    from collections import Counter
    summary={'status':'feasible_winner' if best is not None else 'no_feasible_winner','proposal_method':method,
             'termination':'backend_budget_exhausted' if env.remaining==0 else ('no_candidate' if not choices else 'proposal_space_exhausted'),
             'seed':cfg.get('random_seed',0),'source_baseline':spec['source_baseline'],
             'backend_attempts':env.used,'budget':env.budget,'remaining':env.remaining,
             'failed_backend_attempts':sum(not a['valid'] for a in env.attempts),
             'proposal_attempts':len(audit),'proposal_budget':cap,'proposal_statuses':dict(Counter(a['status'] for a in audit)),
             'trial_statuses':dict(Counter(r['status'] for r in rows)),
             'evaluated_structures':sum(r['status']=='evaluated' and r['structure_changed'] for r in rows),
             'best_feasible_objective':best,'experiment_wall_seconds':time.monotonic()-start}
    write(out/'summary.json',summary)
    summary['replay']=subprocess_replay(out,best is not None,replay_timeout)
    summary['total_wall_seconds']=time.monotonic()-start;write(out/'summary.json',summary)
    write(out/'SHA256.json',{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in out.iterdir() if p.is_file() and p.name!='SHA256.json'})
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('experiment', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--max-evaluations', type=int, help='Lower budget for debugging; does not shrink model')
    parser.add_argument('--replay-timeout', type=int, default=3600)
    parser.add_argument('--method', choices=['legacy','constrained'])
    parser.add_argument('--seed', type=int)
    args = parser.parse_args()
    spec = json.loads(args.experiment.read_text())
    if args.method is not None:
        spec['proposal_method'] = args.method
    if args.seed is not None:
        if args.seed not in spec.get('paired_seeds', []):
            parser.error('seed must be in frozen paired_seeds')
        spec['config']['random_seed'] = args.seed
    print(json.dumps(run(spec, args.output,
                         max_evaluations=args.max_evaluations, replay_timeout=args.replay_timeout), indent=2))


if __name__ == '__main__':
    main()
