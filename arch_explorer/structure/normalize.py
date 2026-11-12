"""Conservative normalization: preserve names, parameters and physical multiplicity."""
from copy import deepcopy

from arch_explorer.adf import COMPONENTS, SpaceMatrix, fingerprint
from .primitives import check


def normalize(root, *, registry=COMPONENTS):
    result = deepcopy(root)

    def visit(node):
        if not isinstance(node, SpaceMatrix):
            return
        for child in node.elements:
            visit(child)
        names = {n.name for n in (*node.elements, *node.networks)}
        children = []
        for child in node.elements:
            if (isinstance(child, SpaceMatrix) and child.repeat == 1 and not child.networks
                    and len(child.elements) == 1
                    and child.elements[0].name not in names - {child.name}):
                child = child.elements[0]
            children.append(child)
        node.elements = sorted(children, key=lambda n: n.name)
        node.networks.sort(key=lambda n: n.name)

    visit(result)
    check(result, registry)
    return result


def normalized_fingerprint(root, *, registry=COMPONENTS):
    return fingerprint(normalize(root, registry=registry))
