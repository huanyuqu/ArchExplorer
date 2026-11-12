"""Bounded generic proposals from declarations and values visible in an ADF.

This is a proposal generator, not a hardware support whitelist. Required fixed
parameters without witnessed values cannot be invented: callers can supply
explicit component specs. Validation and execution remain authoritative.
"""
from copy import deepcopy

from arch_explorer.adf import SpaceMatrix, SpacePoint, from_dict, walk


def _name(names, preferred):
    while preferred in names:
        preferred += "_copy"
    return preferred


def _prototype(component, points):
    params = {}
    for key, declaration in component["parameters"].items():
        kinds = {"float": (int, float), "int": (int,), "str": (str,), "bool": (bool,)}
        witnessed = [p.params[key] for p in points if key in p.params
                     and p.params[key].unit == declaration["unit"]
                     and type(p.params[key].value) in kinds.get(declaration["kind"], ())]
        domain = declaration["domain"]
        if witnessed:
            value = witnessed[0].value
        elif domain and declaration["domain_kind"] in {"choices", "range"}:
            value = domain[0]
        elif declaration["kind"] == "bool":
            value = False
        elif not declaration["required"]:
            continue
        else:
            return None
        # Catalog bounds/choices have precedence over values borrowed from other types.
        if domain:
            if declaration["domain_kind"] == "choices" and value not in domain:
                value = domain[0]
            elif declaration["domain_kind"] == "range":
                value = min(domain[1], max(domain[0], value))
        origin = witnessed[0].source if witnessed else "declared parameter domain"
        params[key] = {"value": value, "unit": declaration["unit"],
                       "source": f"synthetic catalog proposal from {origin}; not a device measurement"}
    return {"type": "point", "component": component["name"], "name": component["name"], "params": params}


def catalog_proposals(state, catalog, basic_only=False):
    """Generic graph edits; named transforms are opt-in through search.proposals.

    basic_only restricts explicit transforms in the search client. All default
    proposals already consist exclusively of basic edits.
    """
    root = from_dict(state["adf"])
    nodes = list(walk(root))
    points = [n for _, n in nodes if isinstance(n, SpacePoint)]
    components = {c["name"]: c for c in catalog}
    prototypes = [p for c in catalog if (p := _prototype(c, points)) is not None]
    choices = []
    for path, node in nodes:
        if isinstance(node, SpacePoint):
            if "/" not in path:
                continue
            if components[node.component]["role"] != "network":
                choices.append({"op": "split", "point": path, "num": 2})
            choices.append({"op": "delete", "component": path})
            # Categorical features (including topology flags) use the same edit
            # construction for every component, without preset/tree-name tests.
            for key, declaration in components[node.component]["parameters"].items():
                if declaration["kind"] == "bool":
                    values = [False, True]
                elif declaration["domain_kind"] == "choices":
                    values = declaration["domain"]
                else:
                    continue
                for value in values:
                    if key in node.params and value == node.params[key].value:
                        continue
                    spec = node.to_dict()
                    spec["params"][key] = {"value": value, "unit": declaration["unit"],
                                           "source": "synthetic categorical graph proposal"}
                    choices.append([{"op": "delete", "component": path},
                                    {"op": "create", "container": path.rsplit("/", 1)[0], "spec": spec}])
            continue
        if "/" in path:
            choices.append({"op": "remove", "level": path})
        names = {n.name for n in (*node.elements, *node.networks)}
        # Every catalog type with constructible parameters can enter any container.
        for prototype in prototypes:
            spec = deepcopy(prototype)
            spec["name"] = _name(names, spec["name"])
            choices.append({"op": "create", "container": path, "spec": spec})
        for child in (*node.elements, *node.networks):
            spec = child.to_dict()
            spec["name"] = _name(names, child.name + "_copy")
            choices.append({"op": "create", "container": path, "spec": spec})
        for source, child in nodes:
            if "/" in source and source.rsplit("/", 1)[0] != path and child.name not in names:
                if not path.startswith(source + "/") and path != source:
                    choices.append({"op": "create", "container": path, "component": source})
        for child in node.elements:
            for size in range(1, child.repeat):
                if child.repeat % size == 0:
                    choices.append({"op": "insert", "parent": path, "children": [f"{path}/{child.name}"],
                                    "name": _name(names, "group"), "group_size": size})
        if len(node.elements) > 1:
            choices.append({"op": "insert", "parent": path,
                            "children": [f"{path}/{n.name}" for n in node.elements[:2]],
                            "name": _name(names, "group")})
        for collection in (node.elements, node.networks):
            for i, child in enumerate(collection):
                for other in collection[i + 1:]:
                    if (isinstance(child, SpacePoint) and isinstance(other, SpacePoint)
                            and child.component == other.component and child.repeat == other.repeat):
                        choices.append({"op": "merge", "points": [f"{path}/{child.name}", f"{path}/{other.name}"]})
    return choices


def _exchanges(state, catalog):
    """Resource-neutral alternatives derived from roles/extensive declarations."""
    root = from_dict(state['adf'])
    nodes = list(walk(root))
    points = [n for _, n in nodes if isinstance(n, SpacePoint)]
    components = {c['name']: c for c in catalog}
    choices = []
    for path, point in nodes:
        if not isinstance(point, SpacePoint) or '/' not in path:
            continue
        role = components[point.component]['role']
        for component in catalog:
            if component['role'] != role or component['name'] == point.component:
                continue
            spec = _prototype(component, points)
            if spec is not None:
                spec.update(name=point.name, repeat=point.repeat)
                choices.append([{'op': 'delete', 'component': path},
                                {'op': 'create', 'container': path.rsplit('/', 1)[0], 'spec': spec}])
    # Two siblings can exchange an extensive quantity while conserving the
    # physical total. Replace their containing subtree atomically (two edits).
    for path, container in nodes:
        if not isinstance(container, SpaceMatrix) or '/' not in path:
            continue
        siblings = [n for n in container.elements if isinstance(n, SpacePoint)]
        for i, left in enumerate(siblings):
            for right in siblings[i + 1:]:
                for key, decl in components[left.component]['parameters'].items():
                    other = components[right.component]['parameters'].get(key, {})
                    if not (decl.get('extensive') and other.get('extensive') and decl['unit'] == other['unit'] and decl['kind'] == other['kind']
                            and key in left.params and key in right.params):
                        continue
                    for donor, receiver in ((left, right), (right, left)):
                        value = donor.params[key].value
                        delta = value / 2
                        increase = delta * donor.repeat / receiver.repeat
                        if decl['kind'] == 'int' and (delta != int(delta) or increase != int(increase)):
                            continue
                        subtree = container.to_dict()
                        for child in subtree['elements']:
                            if child['name'] == donor.name:
                                child['params'][key]['value'] -= int(delta) if decl['kind'] == 'int' else delta
                            if child['name'] == receiver.name:
                                child['params'][key]['value'] += int(increase) if other['kind'] == 'int' else increase
                        choices.append([{'op': 'delete', 'component': path},
                                        {'op': 'create', 'container': path.rsplit('/', 1)[0], 'spec': subtree}])
    return choices


def constrained_proposals(state, catalog, *, max_attempts=100, rng=None, audit=None, registry=None):
    """Finite preflight. Each considered edit is reported; no backend is called.

    The same resource_constraints and cost inputs carried by get_state govern
    direct edits and sampling. Empty output is an explicit exhausted space.
    """
    import random
    from arch_explorer.adf import COMPONENTS
    from arch_explorer.cost import estimate
    from arch_explorer.structure import apply_edit, StructureError
    from arch_explorer.structure.resources import resource_report
    if type(max_attempts) is not int or max_attempts < 1:
        raise ValueError('max_attempts must be a positive integer')
    registry = registry or COMPONENTS
    choices = catalog_proposals(state, catalog) + _exchanges(state, catalog)
    (rng or random.Random(0)).shuffle(choices)
    root = from_dict(state['adf'])
    context = state.get('constraint_context', {})
    accepted, seen = [], set()
    for proposal in choices[:max_attempts]:
        row = {'edits': proposal if isinstance(proposal, list) else [proposal]}
        try:
            adf = root
            for edit in row['edits']:
                adf = apply_edit(adf, edit, registry=registry).adf
            # Provenance-only differences do not make a new physical proposal.
            def physical(value):
                if isinstance(value, dict):
                    return {k: physical(v) for k, v in value.items() if k != 'source'}
                return [physical(v) for v in value] if isinstance(value, list) else value
            import json
            key = json.dumps(physical(adf.to_dict()), sort_keys=True)
            if key in seen:
                row['status'] = 'duplicate'
            else:
                seen.add(key)
                cost = estimate(adf, options=context.get('cost_model', {})).to_dict()
                report = resource_report(adf, state.get('resource_constraints') or {}, registry,
                                         cost=cost, limits=context.get('limits', {}))
                row.update(status='eligible' if report['valid'] else 'resource_rejected',
                           violations=report['violations'])
                if report['valid']:
                    accepted.append(proposal)
        except (StructureError, ValueError, KeyError, TypeError) as error:
            row.update(status='invalid_edit', error=str(error))
        if audit is not None:
            audit.append(row)
    return accepted




def proposals(state, catalog, basic_only=False, *, max_attempts=100, rng=None, audit=None, registry=None):
    if state.get('resource_constraints') is None:
        return catalog_proposals(state, catalog, basic_only)
    return constrained_proposals(state, catalog, max_attempts=max_attempts, rng=rng, audit=audit, registry=registry)
