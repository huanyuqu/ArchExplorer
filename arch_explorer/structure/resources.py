"""Serializable, name-independent resource checks; no simulator calls.

Selectors match catalog roles/component types, never tree paths. Totals count
all ancestor repeats. Units are exact declaration units, not implicit conversions.
"""
import math
from arch_explorer.adf import SpacePoint, walk


def validate_constraints(spec, registry=None):
    if not isinstance(spec, dict) or set(spec) - {'totals', 'parameters'}:
        raise ValueError('resource_constraints accepts only totals and parameters')
    for mode in ('totals', 'parameters'):
        rules = spec.get(mode, [])
        if not isinstance(rules, list):
            raise ValueError(f'{mode} must be a list')
        for rule in rules:
            if (not isinstance(rule, dict) or set(rule) - {'select', 'param', 'unit', 'min', 'max', 'value'}
                    or not {'select', 'unit'} <= rule.keys()):
                raise ValueError('Resource rules require select/unit and explicit bounds')
            sel = rule['select']
            if not isinstance(sel, dict) or set(sel) - {'component', 'role'}:
                raise ValueError('Selectors use component/role; path/name selectors cannot enforce physical budgets')
            if any(not isinstance(v, str) or not v for v in sel.values()) or (
                    'role' in sel and sel['role'] not in {'compute', 'memory', 'network'}):
                raise ValueError('Invalid resource selector')
            if not isinstance(rule['unit'], str) or not rule['unit']:
                raise ValueError('Resource unit must be explicit')
            if registry is not None:
                types = [registry.get(n) for n in registry.names() if
                         ('component' not in sel or n == sel['component']) and
                         ('role' not in sel or registry.get(n).role == sel['role'])]
                if not types:
                    raise ValueError('Resource selector matches no registered component type')
                if 'param' in rule and isinstance(rule['param'], str):
                    declarations = [c.params[rule['param']] for c in types if rule['param'] in c.params]
                    if not declarations or all(d.unit != rule['unit'] for d in declarations):
                        raise ValueError('Resource parameter/unit matches no selected declaration')
            if 'param' in rule and (not isinstance(rule['param'], str) or not rule['param']):
                raise ValueError('param must be a nonempty name')
            if mode == 'parameters' and 'param' not in rule:
                raise ValueError('Per-instance rules require param')
            if mode == 'totals' and ('value' in rule or ('param' not in rule and rule['unit'] != 'count')):
                raise ValueError('Totals require min/max; instance totals use count')
            if not any(k in rule for k in ('min', 'max', 'value')):
                raise ValueError('Resource bound missing')
            for key in ('min', 'max'):
                if key in rule and (type(rule[key]) not in (int, float) or not math.isfinite(rule[key]) or rule[key] < 0):
                    raise ValueError('Resource bounds must be finite nonnegative numbers')
            if 'min' in rule and 'max' in rule and rule['min'] > rule['max']:
                raise ValueError('Resource min exceeds max')
            if 'value' in rule and (set(rule) & {'min', 'max'} or type(rule['value']) not in (int, float, str, bool)
                    or (type(rule['value']) in (int, float) and not math.isfinite(rule['value']))):
                raise ValueError('Fixed value must be a finite scalar without min/max')
    return spec


def inventory(adf, registry):
    counts, rows = {}, []
    for path, node in walk(adf):
        parent = path.rsplit('/', 1)[0] if '/' in path else None
        count = node.repeat * counts.get(parent, 1)
        counts[path] = count
        if isinstance(node, SpacePoint):
            rows.append({'path': path, 'component': node.component,
                         'role': registry.get(node.component).role, 'instances': count,
                         'params': {k: p.to_dict() for k, p in node.params.items()}})
    return rows


def resource_report(adf, constraints, registry, *, cost=None, limits=None):
    rows = inventory(adf, registry)
    violations, checks = [], []
    for mode in ('totals', 'parameters'):
        for i, rule in enumerate(constraints.get(mode, [])):
            selected = [r for r in rows if all(r[k] == v for k, v in rule['select'].items())]
            values = []
            for row in selected:
                if 'param' not in rule:
                    values.append((row['path'], row['instances']))
                    continue
                key = rule['param']
                # A wildcard key applies to every type that declares it.
                if key not in registry.get(row['component']).params:
                    continue
                p = row['params'].get(key)
                value = None if p is None or p['unit'] != rule['unit'] else p['value']
                if mode == 'totals':
                    value = value * row['instances'] if type(value) in (int, float) else None
                values.append((row['path'], value))
            if mode == 'totals':
                values = [(None, None if any(v is None for _, v in values) else sum(v for _, v in values))]
            for path, value in values:
                valid = value is not None
                if 'value' in rule:
                    valid = valid and value == rule['value'] and (isinstance(value, bool) == isinstance(rule['value'], bool))
                else:
                    valid = valid and type(value) in (int, float)
                    valid = valid and ('min' not in rule or value >= rule['min']) and ('max' not in rule or value <= rule['max'])
                item = {'rule': f'{mode}[{i}]', 'path': path, 'actual': value, 'unit': rule['unit'], 'valid': valid}
                checks.append(item)
                if not valid:
                    violations.append(item)
    for key, bound in (limits or {}).items():
        if bound is not None:
            value = cost.get(key) if cost else None
            item = {'rule': key, 'actual': value, 'max': bound, 'valid': value is not None and value <= bound}
            checks.append(item)
            if not item['valid']:
                violations.append(item)
    return {'valid': not violations, 'violations': violations, 'checks': checks, 'resources': rows}
