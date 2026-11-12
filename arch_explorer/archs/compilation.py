"""Explicit synthetic semantics and numerical inputs; no target lowering here."""
from arch_explorer.adf import MappingSpec, Param, SpaceMatrix, SpacePoint, WorkloadSpec
from arch_explorer.workload.api import Workload


def _input(rows, cols, offset):
    # Small signed powers-of-two fractions are exactly representable in FP32.
    return [[[((row * cols + col + offset) % 17 - 8) / 8.0
              for col in range(cols)] for row in range(rows)]]


def matmul_workload(*, m=3, k=5, n=7, fanout=False):
    """One batched matmul, optionally read by two further matmuls.

    Default dimensions deliberately leave boundary tiles. Numerical reference
    execution belongs to the public workload API; this example supplies inputs
    and tensor/compute relations, never a second matrix multiplication routine.
    """
    if any(type(size) is not int or size <= 0 for size in (m, k, n)):
        raise ValueError("matmul example dimensions must be positive integers")
    if type(fanout) is not bool:
        raise TypeError("fanout must be a boolean")
    graph = Workload("matmul_fanout" if fanout else "matmul", dtype="fp32")
    a = graph.tensor("A", (1, m, k))
    b = graph.tensor("B", (1, k, n), kind="weight")
    product = graph.tensor("product", (1, m, n))
    graph.add("multiply", "matmul", [a, b], [product], batch=1, m=m, k=k, n=n)
    inputs = {a: _input(m, k, 0), b: _input(k, n, 3)}
    if fanout:
        for name, offset in (("left", 5), ("right", 11)):
            weight = graph.tensor(name + "_weight", (1, n, n), kind="weight")
            output = graph.tensor(name + "_output", (1, m, n))
            graph.add(name, "matmul", [product, weight], [output],
                      batch=1, m=m, k=n, n=n)
            inputs[weight] = _input(n, n, offset)
    graph.meta["inputs"] = inputs
    return WorkloadSpec("graph", graph.to_dict())


def compile_mapping(*, compute, memory, tile_m=2, tile_k=3, tile_n=4):
    """Public placement and tiles, independent of adapter-specific options."""
    return MappingSpec("compile_graph_v2", {
        "compute": compute, "memory": memory,
        "rules": {"matmul": {"id": "matmul.tiled", "version": "1", "options": {
            "tiles": {"m": tile_m, "k": tile_k, "n": tile_n}}}},
    })


_SOURCE = "explicit teaching hardware; synthetic, not measured GPU parameters"


def paired_board(hardware=None):
    """Copy one teaching board and expose two compute points in its domain.

    Hardware factories keep their public ``(adf, workload, mapping)`` contract.
    This helper changes structure only; component models still own service.
    """
    from copy import deepcopy
    from arch_explorer.adf import find
    source, _, placement = (events_scene if hardware is None else hardware)()
    adf = deepcopy(source)
    compute, memory = placement.params['compute'], placement.params['memory']
    if not isinstance(compute, str) or '/' not in compute:
        raise ValueError('Teaching hardware needs one compute point with a parent matrix')
    parent, name = compute.rsplit('/', 1)
    peer_name = name + '_peer'
    matrix = find(adf, parent)
    if any(element.name == peer_name for element in matrix.elements):
        raise ValueError('Teaching hardware peer name is already occupied')
    peer = deepcopy(find(adf, compute))
    peer.name = peer_name
    matrix.elements.append(peer)
    return adf, (compute, parent + '/' + peer_name), memory


def _ordinary_board(binding):
    sm = SpacePoint("sm", "sm", {"precision": Param("fp32", "1", _SOURCE)},
                    implementation=binding)
    def memory(name, capacity, latency, technology=None):
        params = {"capacity": Param(capacity, "B", _SOURCE),
                  "bandwidth": Param(16.0, "B/ns", _SOURCE),
                  "latency": Param(latency, "ns", _SOURCE)}
        if technology is not None:
            params["technology"] = Param(technology, "1", _SOURCE)
        return SpacePoint(name, "dram" if technology else "sram", params)
    def bus(name):
        return SpacePoint(name, "bus", {"bandwidth": Param(16.0, "B/ns", _SOURCE),
                                         "latency": Param(1.0, "ns", _SOURCE)})
    core = SpaceMatrix("core", [sm], [bus("core_bus")])
    return SpaceMatrix("board", [core, memory("sram", 65536, 1.0),
                                  memory("dram", 2**20, 4.0, "hbm")], [bus("device_bus")])


def events_scene(*, m=3, k=5, n=7, fanout=False, **mapping_options):
    """Same hardware identity with an explicitly sourced events SM binding."""
    from arch_explorer.components.api import ImplementationBinding
    binding = ImplementationBinding("sm.events", "1", {
        "lanes": Param(8, "count", _SOURCE), "sfu_lanes": Param(1, "count", _SOURCE),
        "clock_ghz": Param(1.0, "GHz", _SOURCE)})
    return (_ordinary_board(binding), matmul_workload(m=m, k=k, n=n, fanout=fanout),
            compile_mapping(compute="board/core/sm", memory="board/sram", **mapping_options))


def roofline_scene(*, m=3, k=5, n=7, fanout=False, **mapping_options):
    """Explicit ideal-overlap component model; no external duplicate IO timing."""
    from arch_explorer.components.api import ImplementationBinding
    binding = ImplementationBinding("sm.roofline", "1", {
        "peak_ops_per_ns": Param(128.0, "op/ns", _SOURCE),
        "bandwidth_bytes_per_ns": Param(64.0, "B/ns", _SOURCE),
        "boundary": Param("component_io", "1", _SOURCE)})
    return (_ordinary_board(binding), matmul_workload(m=m, k=k, n=n, fanout=fanout),
            compile_mapping(compute="board/core/sm", memory="board/sram", **mapping_options))


def heterogeneous_scene(*, m=3, k=5, n=7, fanout=True, **mapping_options):
    """Two explicit SM implementations sharing one memory and real graph edges."""
    from arch_explorer.adf import find
    if fanout is not True:
        raise ValueError("heterogeneous teaching scene requires the two-reader graph")
    adf, workload, _ = events_scene(m=m, k=k, n=n, fanout=True)
    roofline_adf, _, _ = roofline_scene(m=m, k=k, n=n)
    other = find(roofline_adf, "board/core/sm")
    other.name = "ideal_sm"
    find(adf, "board/core").elements.append(other)
    return adf, workload, compile_mapping(
        compute={"multiply": "board/core/sm", "left": "board/core/ideal_sm",
                 "right": "board/core/sm"}, memory="board/sram", **mapping_options)


def native_scene(*, m=3, k=5, n=7, fanout=False, **mapping_options):
    """Component-owned assembly; callers explicitly bind native adapter options.

    The mapping contains no thread count. Bind the selected compute path to
    accelsim@1 with an explicit threads_per_block in AdapterBinding.options.
    Native capability/topology validation belongs to that adapter.
    """
    from arch_explorer.components.assemblies.gpu import gpu_adf
    return (gpu_adf(sm_clusters=1), matmul_workload(m=m, k=k, n=n, fanout=fanout),
            compile_mapping(compute="component_gpu/cluster/sm",
                            memory="component_gpu/partition/dram", **mapping_options))
