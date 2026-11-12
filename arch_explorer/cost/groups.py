"""Cost views of homogeneous striped DRAM groups; never change the input ADF."""
from dataclasses import replace

from arch_explorer.adf import Param, SpaceMatrix, SpacePoint, instance_count, walk
from arch_explorer.mapping.base import is_memory_group


def memory_groups(adf):
    """Collapse the maximal homogeneous groups recognized by the events API.

    The aggregate is one accounting row over all expanded instances. Group
    overrides address that row; per-part overrides are deliberately not silently
    consumed. Heterogeneous groups retain per-part costing instead.
    """
    groups = {}

    def visit(node, path):
        if not isinstance(node, SpaceMatrix):
            return
        if is_memory_group(node):
            entries = [(path + p[len(node.name):], n) for p, n in walk(node)]
            drams = [(p, n) for p, n in entries if isinstance(n, SpacePoint) and n.component == "dram"]
            pims = [(p, n) for p, n in entries if isinstance(n, SpacePoint) and n.component == "pim_unit"]
            # These values have a single meaning for a homogeneous memory group.
            keys = ("technology", "dram_usd_gib", "dram_density", "interface_bytes")
            signatures = {tuple((k, n.params[k].value if k in n.params else None) for k in keys) for _, n in drams}
            dram_parents = {p.rsplit("/", 1)[0] for p, _ in drams}
            pim_parents = {p.rsplit("/", 1)[0] for p, _ in pims}
            # Preserve bank-local ownership; do not spread a partial PIM retrofit
            # over unmodified banks or hide ambiguous multi-DRAM siblings.
            unambiguous = (not pims or (pim_parents == dram_parents and len(dram_parents) == len(drams)))
            if len(signatures) == 1 and unambiguous:
                capacity = sum(n.params["capacity"].value * instance_count(adf, p) for p, n in drams)
                params = dict(drams[0][1].params)
                params["capacity"] = Param(capacity, "B", f"sum of expanded DRAM capacities in {path}; see member paths")
                units = sum(instance_count(adf, p) for p, _ in pims)
                ops = sum(n.params["ops_per_cycle"].value * n.params["clock_ghz"].value * instance_count(adf, p)
                          for p, n in pims)
                # interface_count on a part describes that part's allocation.
                if any("interface_count" in n.params for _, n in drams):
                    if not all("interface_count" in n.params for _, n in drams):
                        return  # cannot infer missing allocations
                    params["interface_count"] = replace(drams[0][1].params["interface_count"],
                        value=sum(n.params["interface_count"].value * instance_count(adf, p) for p, n in drams),
                        source=f"sum of expanded interface_count in {path}")
                groups[path] = {"point":SpacePoint(node.name, "dram", params),
                    "member_paths":[p for p, _ in drams], "part_instances":sum(instance_count(adf, p) for p, _ in drams),
                    "pim_units":units, "pim_ops_ghz":ops,
                    "provenance":[{"path":p, "parameter":k, **v.to_dict()} for p, n in drams + pims
                                  for k, v in n.params.items() if k in (*keys, "capacity", "interface_count", "ops_per_cycle", "clock_ghz")]}
                return
        for child in node.elements:
            visit(child, f"{path}/{child.name}")

    visit(adf, adf.name)
    return groups


def merge_unsupported(items):
    """Coalesce equal causes/metrics without burying users in one error per bank."""
    combined = {}
    for item in items:
        key = (item["reason"], tuple(item["metrics"]))
        if key not in combined:
            combined[key] = {**item, "affected_path_count":0, "example_paths":[]}
        result = combined[key]
        result["affected_path_count"] += item.get("affected_path_count", 1)
        for path in item.get("example_paths", [item["path"]]):
            if path not in result["example_paths"] and len(result["example_paths"]) < 4:
                result["example_paths"].append(path)
    return list(combined.values())
