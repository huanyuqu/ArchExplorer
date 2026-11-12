"""Contract-only test components. Deliberately not a predictive hardware model."""
from arch_explorer.adf import (COMPONENTS, MAPPING_TEMPLATES, ComponentType, Evaluation, MappingTemplate,
                               ParamSpec, SpacePoint, instance_count, register_backend, walk)


class FakeComponent(ComponentType):
    def build(self, point, context):
        return None

    def area_mm2(self, point):
        return point.params["capacity"].value / 2**25 if self.role == "memory" else 1.0

    def cost_usd(self, point):
        return self.area_mm2(point) * 2.0


def component_types():
    integer = lambda unit="count", extensive=False: ParamSpec(unit, "int", extensive, (1, 2**40))
    number = lambda unit, extensive=False: ParamSpec(unit, "float", extensive, (0.0001, 1e6))
    precision = ParamSpec("1", "str", domain=["fp16", "fp32"])
    memory = {"capacity": integer("B", True), "bandwidth": number("B/ns", True), "latency": number("ns")}
    definitions = {
        "sram": ("memory", memory),
        "dram": ("memory", {**memory, "technology": ParamSpec("1", "str", domain=["ddr", "gddr", "hbm", "bonded"])}),
        "vector_unit": ("compute", {"lanes": integer(), "sfu_lanes": integer(), "clock_ghz": number("GHz"), "precision": precision}),
        "systolic_array": ("compute", {"rows": integer(), "cols": integer(), "clock_ghz": number("GHz"),
                           "dataflow": ParamSpec("1", "str", domain=["os", "ws", "is"]), "precision": precision}),
        "mesh2d": ("network", {"rows": integer(), "cols": integer(), "link_bandwidth": number("B/ns"), "hop_latency": number("ns")}),
        "bus": ("network", {"bandwidth": number("B/ns"), "latency": number("ns")}),
        "crossbar": ("network", {"port_bandwidth": number("B/ns"), "latency": number("ns")}),
        "p2p_switch": ("network", {"link_bandwidth": number("B/ns"), "links_per_endpoint": integer(), "latency": number("ns")}),
    }
    return {name: type("Fake_" + name, (FakeComponent,), dict(name=name, role=role, params=params))
            for name, (role, params) in definitions.items()}


class FakeMapping(MappingTemplate):
    name = "core-test-map"

    def applicable(self, adf, workload):
        return any(isinstance(n, SpacePoint) and COMPONENTS.get(n.component).role == "compute" for _, n in walk(adf))

    def space(self, adf, workload):
        return {"partitions": [1, 2, 4], "storage": ["shared", "local"]}

    def apply(self, simulator, workload, params):
        pass


def fake_backend(adf, workload, mapping, missing_metrics=False, fail=False):
    if fail:
        raise ValueError("synthetic backend failure")
    compute, bandwidth, area, cost, provenance = 0.0, 0.0, 0.0, 0.0, []
    resources = []
    for path, node in walk(adf):
        if not isinstance(node, SpacePoint):
            continue
        count = instance_count(adf, path)
        component = COMPONENTS.get(node.component)
        area += count * component.area_mm2(node)
        cost += count * component.cost_usd(node)
        if component.role == "compute":
            compute += count * node.value("clock_ghz") * (node.value("lanes") if node.component == "vector_unit" else
                       node.value("rows") * node.value("cols"))
            resources.append(path)
        if component.role == "memory":
            bandwidth += count * node.value("bandwidth")
        provenance.extend({"path": path, "param": key, **param.to_dict()} for key, param in node.params.items())
    if compute <= 0 or bandwidth <= 0:
        raise ValueError("Synthetic evaluation needs compute and memory")
    latency = workload.params.get("operations", 4096) / compute + 1024 / bandwidth
    latency *= 1 + 0.1 / mapping.params["partitions"] + (0.1 if mapping.params["storage"] == "shared" else 0)
    return Evaluation(latency_ns=latency, area_mm2=None if missing_metrics else area,
                      cost_usd=None if missing_metrics else cost, provenance=provenance,
                      breakdown=[{"operator": "synthetic", "task_type": "compute", "resource": resources[0],
                                  "start_ns": 0.0, "end_ns": latency}], utilization={resources[0]: 1.0},
                      warnings=["Test double: no physical prediction"])


def install():
    COMPONENTS._items = component_types()
    MAPPING_TEMPLATES._items = {FakeMapping.name: FakeMapping}
    register_backend("core-test", fake_backend)
