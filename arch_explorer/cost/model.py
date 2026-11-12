"""ADF area/cost estimates with explicit reference assumptions and omissions."""
from dataclasses import asdict, dataclass
import math

from arch_explorer.adf import Param, SpacePoint, instance_count, walk
from .constants import CONSTANTS as C, DEFAULTS
from .formulas import fpu_area, manufacturing, raw_die_cost, sram_area
from .groups import memory_groups, merge_unsupported

CATEGORIES = ("arithmetic", "control_rf", "sram", "interfaces_network")
NETWORKS = {"mesh2d", "bus", "crossbar", "p2p_switch"}
COMPUTE = {"systolic_array", "vector_unit", "pim_unit", "cgra_pe"}


@dataclass
class CostEstimate:
    area_mm2: float | None
    cost_usd: float | None
    raw_cost_usd: float | None
    full_cost_usd: float | None
    by_category: dict
    by_path: list
    provenance: list
    unsupported: list
    omitted: list
    assumptions: list
    manufacturing: dict | None
    memory: dict
    basis: str
    model: str = "analytical-v1"

    def to_dict(self):
        return asdict(self)


def _param(value):
    result = value if isinstance(value, Param) else Param.from_dict(value)
    if not result.source.strip():
        raise ValueError("Cost model parameters require a nonempty source")
    return result


def estimate(adf, *, options=None):
    """Estimate logic-die area and raw-die+memory (or full RE scenario) cost.

    options: basis=raw|full, process_nm=Param, die_count=Param,
    overrides={ADF path: {cost-only parameter: Param}}. No ADF mutation.
    DRAM silicon is separately reported, never added to the logic-die area.
    """
    options = options or {}
    if set(options) - {"basis", "process_nm", "die_count", "overrides", "scope_notes"}:
        raise ValueError("Unknown cost_model option")
    basis = options.get("basis", "raw")
    if basis not in {"raw", "full"}:
        raise ValueError("cost_model.basis must be raw or full")
    nodes = dict(walk(adf))
    groups = memory_groups(adf)
    for path, group in groups.items():
        nodes = {p: n for p, n in nodes.items() if not p.startswith(path + "/")}
        nodes[path] = group["point"]
    siblings_by_parent = {}
    for path, point in nodes.items():
        if isinstance(point, SpacePoint):
            siblings_by_parent.setdefault(path.rsplit("/", 1)[0], []).append((path, point))
    overrides = options.get("overrides", {})
    if not isinstance(overrides, dict):
        raise ValueError("cost_model.overrides must map paths to sourced parameters")
    provenance = [{"path":"$model_catalog", "parameter":key, **value.to_dict()} for key, value in C.items()]
    assumptions, unsupported, omitted, rows = [], [], [], []
    notes = options.get("scope_notes", [])
    if not isinstance(notes, list) or any(not isinstance(note, str) or not note.strip() for note in notes):
        raise ValueError("cost_model.scope_notes must be a list of nonempty strings")
    assumptions.extend({"scope":"cost configuration", "reason":note} for note in notes)
    used_overrides = set()

    def omit(path, reason, metrics=("area_mm2", "cost_usd")):
        members = groups[path]["member_paths"] if path in groups else [path]
        unsupported.append({"path":path, "reason":reason, "metrics":list(metrics),
                            "affected_path_count":len(members), "example_paths":members[:4]})

    def read(path, key, default=None, unit=None):
        node = nodes.get(path)
        if key in overrides.get(path, {}):
            value = _param(overrides[path][key])
            used_overrides.add((path, key))
        elif isinstance(node, SpacePoint) and key in node.params:
            value = node.params[key]
        elif default is not None:
            value = default
            assumptions.append({"path":path, "parameter":key, **value.to_dict()})
        else:
            raise ValueError(f"Missing cost parameter {key}; provide a sourced cost_model.overrides entry")
        if unit is not None and value.unit != unit:
            raise ValueError(f"{key}: expected unit {unit}, got {value.unit}")
        provenance.append({"path":path, "parameter":key, **value.to_dict()})
        if isinstance(value.value, bool) or (isinstance(value.value, (int, float)) and
                (not math.isfinite(value.value) or value.value < 0)):
            raise ValueError(f"{key} must be nonnegative and finite")
        return value.value

    node_param = _param(options.get("process_nm", C["process_nm"]))
    die_param = _param(options.get("die_count", DEFAULTS["die_count"]))
    if node_param.unit != "nm" or die_param.unit != "count":
        raise ValueError("process_nm uses nm and die_count uses count")
    node, dies = node_param.value, die_param.value
    if type(node) is not int or node not in (5, 7):
        raise ValueError("Cost model supports process_nm 5 or 7; no inferred node scaling")
    if type(dies) is not int or dies < 1:
        raise ValueError("die_count must be a positive integer")
    provenance.extend([{"path":"$model", "parameter":k, **v.to_dict()}
                       for k, v in (("process_nm", node_param), ("die_count", die_param))])
    assumptions.append({"scope":"all logic", "process_nm":node_param.to_dict(), "equal_area_dies":die_param.to_dict()})
    if "process_nm" not in options:
        assumptions.append({"scope":"all logic", "parameter":"process_nm",
            "status":"unverified_default", "reason":"Default 7 nm node is not confirmed for this device; "
            "area and cost are reference-scenario values, not device estimates. Only 5/7 nm are supported; "
            "provide a sourced process_nm to select a supported device node."})
    density = C[f"density_{node}"].value * 1e6
    memory_cost, memory_silicon, footprints = [], [], []
    for path, point in nodes.items():
        if not isinstance(point, SpacePoint):
            continue
        group = groups.get(path)
        copies = 1 if group else instance_count(adf, path)
        category = {key:0.0 for key in CATEGORIES}
        row = {"path":path, "component":point.component, "instances":copies, "by_category":category}
        parent = path.rsplit("/", 1)[0]
        siblings = siblings_by_parent.get(parent, [])
        if group:
            row.update(member_paths=group["member_paths"], part_instances=group["part_instances"],
                       pim_units=group["pim_units"])
            provenance.extend(group["provenance"])
            omitted.append({"path":path, "items":"DRAM-group bank buses are on DRAM, not independent logic-die networks; "
                "PIM logic/internal routing covered only by the aggregate 20% overhead assumption"})
        try:
            # Cost-only physical counts never change the execution inventory.
            if "physical_instances" in overrides.get(path, {}) or "physical_instances" in point.params:
                physical = read(path, "physical_instances", unit="count")
                if type(physical) is not int or physical < copies:
                    raise ValueError("physical_instances must be an integer at least the expanded active instance count")
                copies = physical
                row["instances"] = copies
            get = lambda key, default=None, unit=None: read(path, key, default, unit)
            kind = point.component
            if kind == "systolic_array":
                category["arithmetic"] = fpu_area(get("rows", unit="count") * get("cols", unit="count"), get("precision"), node)
                row["note"] = "MAC array only; control/RF must be represented or supplied separately"
                omitted.append({"path":path, "items":"systolic control, RF and array wires outside the MAC-only reference model"})
            elif kind == "vector_unit":
                lanes = get("lanes", unit="count")
                category["arithmetic"] = fpu_area(lanes, get("precision"), node)
                category["control_rf"] = lanes * C["control_per_lane"].value / density
                row["note"] = "Vector FMA + reference lane control; SFU-specific logic/RF not independently characterized"
                omitted.append({"path":path, "items":"vector SFU-specific logic and RF"})
            elif kind == "sram":
                category["sram"] = sram_area(get("capacity", unit="B"), node)
            elif kind in NETWORKS:
                # Communication proxy per expanded immediate endpoint. Internal
                # router buffers/wires are not inferred from simulated bandwidth.
                container = nodes[parent]
                endpoints = sum(n.repeat for n in container.elements)
                if "endpoint_count" in overrides.get(path, {}) or "endpoint_count" in point.params:
                    endpoints = get("endpoint_count", unit="count")
                if kind == "p2p_switch":
                    units = endpoints * get("links_per_endpoint", unit="count")
                    category["interfaces_network"] = units * (C["nvlink_phy"].value + C["nvlink_controller"].value / density)
                    row["note"] = "NVLink3 link proxy; not technology detection"
                else:
                    category["interfaces_network"] = endpoints * C["network_per_endpoint"].value / density
                    row["note"] = "LLMCompass communication-per-endpoint proxy; mesh/bus/crossbar topology and bandwidth not area-calibrated"
                    omitted.append({"path":path, "items":"topology-specific buffers and physical routing; using endpoint proxy"})
                # Explicit on-die external interfaces missing from an execution
                # fabric. Counts are per fabric instance, never inferred from BW.
                for key, prefix in (("nvlink_links", "nvlink"), ("pcie4_lanes", "pcie4")):
                    if key in overrides.get(path, {}) or key in point.params:
                        count = get(key, unit="count")
                        if type(count) is not int:
                            raise ValueError(f"{key} must be a nonnegative integer")
                        if kind == "p2p_switch" and key == "nvlink_links":
                            raise ValueError("p2p_switch already charges NVLink; nvlink_links would double count")
                        category["interfaces_network"] += count * (C[prefix + "_phy"].value + C[prefix + "_controller"].value / density)
            elif kind == "dram":
                capacity = get("capacity", unit="B")
                tech = get("technology")
                pims = [(p, n) for p, n in siblings if n.component == "pim_unit"]
                has_pim = bool(group["pim_units"]) if group else bool(pims)
                overhead = 1 + C["pim_overhead"].value if has_pim else 1
                if not group and pims and len([n for _, n in siblings if n.component == "dram"]) != 1:
                    raise ValueError("PIM costing requires exactly one DRAM sibling per PIM bank group")
                charged_capacity = capacity
                if tech in {"ddr", "gddr"} and has_pim:
                    units = group["pim_units"] if group else sum(n.repeat for _, n in pims) / point.repeat
                    ops = group["pim_ops_ghz"] if group else sum(read(p, "ops_per_cycle", unit="ops/cycle") * read(p, "clock_ghz", unit="GHz") * n.repeat
                              for p, n in pims) / point.repeat
                    equivalent = max(capacity / C["aim_die_bytes"].value, units / C["aim_units"].value,
                                     ops / (C["aim_units"].value * C["aim_ops"].value * C["aim_clock"].value))
                    charged_capacity = equivalent * C["aim_die_bytes"].value
                    row["aim_equivalent_dies"] = copies * equivalent
                    row["note"] = "AiM-equivalent capacity/PU/compute sizing; 16 PU/die, 16 MAC/PU/cycle at 1 GHz. Bandwidth is not assumed off-chip."
                if tech == "bonded":
                    layers = get("bonded_layers", DEFAULTS["bonded_layers"], "count")
                    if type(layers) is not int or not 1 <= layers <= C["bonded_max_layers"].value:
                        raise ValueError("bonded_layers must be an integer from 1 to 4")
                    silicon = capacity / 2**20 / C["bonded_density"].value * overhead
                    footprint = silicon / layers
                    price = C["hbm_usd_gib"].value / C["bond_yield"].value ** layers
                elif tech in {"hbm", "ddr", "gddr"}:
                    if tech == "gddr":
                        assumptions.append({"path":path, "reason":"GDDR logic-side interface uses the DDR5 32-bit "
                            "reference proxy; GDDR PHY/controller area is not independently characterized."})
                    default_price = C["hbm_usd_gib"] if tech == "hbm" else C["ddr_usd_gib"] if tech == "ddr" else None
                    # Price evidence is independent of logic-side interface area.
                    # Do not let a missing/invalid price erase the whole row.
                    try:
                        price = get("dram_usd_gib", default_price, "USD/GiB")
                    except (ValueError, KeyError) as error:
                        price = None
                        omit(path, str(error), ("cost_usd",))
                    prefix = "hbm" if tech == "hbm" else "ddr"
                    unit_bytes = get("interface_bytes", DEFAULTS["hbm_stack_bytes"] if tech == "hbm" else DEFAULTS["ddr_channel_bytes"], "B")
                    if unit_bytes <= 0:
                        raise ValueError("interface_bytes must be positive")
                    interfaces = get("interface_count", unit="count") if "interface_count" in overrides.get(path, {}) or "interface_count" in point.params else capacity / unit_bytes
                    row["interface_count"] = copies * interfaces
                    category["interfaces_network"] = interfaces * (C[prefix + "_phy"].value + C[prefix + "_controller"].value / density)
                    silicon, footprint = None, None
                    if "dram_density" in overrides.get(path, {}) or "dram_density" in point.params:
                        dram_density = get("dram_density", unit="MiB/mm2")
                        if dram_density <= 0:
                            raise ValueError("dram_density must be positive")
                        silicon = charged_capacity / 2**20 / dram_density * overhead
                    else:
                        omit(path, "DRAM die density/area absent; AiM die capacity does not determine silicon area",
                             ("memory_die_area_mm2",))
                else:
                    raise ValueError(f"Unsupported DRAM technology {tech!r}")
                row.update(memory_usd=None if price is None else copies * charged_capacity / 2**30 * price * overhead,
                           memory_die_area_mm2=None if silicon is None else copies * silicon,
                           bonded_footprint_mm2=None if footprint is None else copies * footprint,
                           pim_overhead=overhead - 1)
                memory_cost.append(row["memory_usd"])
                memory_silicon.append(row["memory_die_area_mm2"])
                if footprint is not None:
                    footprints.append(row["bonded_footprint_mm2"])
            elif kind == "pim_unit":
                if not any(n.component == "dram" for _, n in siblings):
                    raise ValueError("pim_unit requires a DRAM sibling for the in-bank overhead model")
                row["note"] = "In-memory logic charged once as the sibling DRAM's 20% area/price overhead; not on logic die"
            else:
                raise ValueError(f"Unsupported component {kind!r}; no area constants (including CGRA until characterized)")
            row["by_category"] = {k: v * copies for k, v in category.items()}
            row["area_mm2"] = sum(row["by_category"].values())
        except (ValueError, KeyError) as error:
            omit(path, str(error))
            row["area_mm2"] = None
            row["by_category"] = {k:None for k in CATEGORIES}
        rows.append(row)
    for path, params in overrides.items():
        for key in params:
            if (path, key) not in used_overrides:
                omit(path, f"Unused cost override {key!r}; path or parameter is not used by this candidate")
    categories = {key:sum(r["by_category"][key] or 0 for r in rows) for key in CATEGORIES}
    area = None if any("area_mm2" in item["metrics"] for item in unsupported) else sum(categories.values())
    raw, full, fab, die_cost = None, None, None, None
    if area is not None:
        try:
            die_cost = dies * raw_die_cost(area / dies, node)
        except ValueError as error:
            omit(adf.name, str(error), ("cost_usd",))
    if area is not None and not any("cost_usd" in item["metrics"] for item in unsupported):
        try:
            raw = die_cost + sum(memory_cost)
            fab = manufacturing(area, sum(memory_cost), node, dies)
            full = fab["total_usd"]
        except ValueError as error:
            omit(adf.name, str(error), ("cost_usd",))
    for row in rows:
        # Allocate the shared raw wafer charge by each path's logic area. This
        # is an accounting allocation, not independent fabrication of every node.
        row["raw_die_usd"] = None if die_cost is None else (die_cost * row["area_mm2"] / area if area else 0.0)
        memory_usd = row.get("memory_usd", 0.0)
        row["raw_cost_usd"] = None if row["raw_die_usd"] is None or memory_usd is None else row["raw_die_usd"] + memory_usd
    return CostEstimate(area, raw if basis == "raw" else full, raw, full, categories, rows, provenance,
        merge_unsupported(unsupported), omitted, assumptions, fab,
        {"cost_usd":None if any(x is None for x in memory_cost) else sum(memory_cost), "die_area_mm2":None if any(x is None for x in memory_silicon) else sum(memory_silicon),
         "bonded_footprint_mm2":sum(footprints), "note":"DRAM silicon separate from logic area; sums exclude unsupported paths"}, basis)
