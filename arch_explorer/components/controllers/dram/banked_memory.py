"""DRAM striping and colocated PIM-bank assembly, owned by these components."""
from dataclasses import dataclass
from arch_explorer.components.api import MemoryGroup


@dataclass
class BankedMemoryGroup(MemoryGroup):
    pims: list


def assemble_banked_memory(simulator):
    """Preserve maximal pure DRAM/PIM groups and single-bank PIM bindings."""
    def container(model):
        return hasattr(model, 'children')

    def bank(model):
        return (not container(model) and model.component.name == 'dram'
                and model.technology != 'bonded')

    def pim(model):
        return not container(model) and model.component.name == 'pim_unit'

    def scan(node):
        pure, banks, pims = True, [], []
        for child in node.children:
            if container(child):
                ok, bs, ps = scan(child)
                pure &= ok
                banks.extend(bs)
                pims.extend(ps)
            elif bank(child):
                banks.append(child)
            elif pim(child):
                pims.append(child)
            else:
                pure = False
        return pure, banks, pims

    def groups(node):
        pure, banks, pims = scan(node)
        if pure and len(banks) >= 2:
            paths = [p.path for p in pims]
            simulator.add_memory_group(BankedMemoryGroup(node.path, banks, paths), paths)
            return
        for child in node.children:
            if container(child):
                groups(child)

    def bind(node):
        banks = [child for child in node.children if bank(child)]
        for child in node.children:
            if container(child):
                bind(child)
            elif pim(child):
                child.bank = banks[0] if len(banks) == 1 else None

    groups(simulator.root)
    bind(simulator.root)
