"""Synthetic seeds and exploration adapters for sourced device presets."""
from arch_explorer.adf import Param, SpaceMatrix, SpacePoint

def point(name, component, values, *, source="synthetic seed", repeat=1):
    return SpacePoint(name, component, {k: Param(v, unit, source) for k, (v, unit) in values.items()}, repeat)


def memory(name="sram", *, technology=None, capacity=2**20, bandwidth=64.0, latency=1.0):
    values = {"capacity": (capacity, "B"), "bandwidth": (bandwidth, "B/ns"), "latency": (latency, "ns")}
    if technology is not None:
        values["technology"] = (technology, "1")
    return point(name, "dram" if technology else "sram", values)


def mesh(name="noc", *, rows=1, cols=4, bandwidth=64.0, latency=1.0):
    return point(name, "mesh2d", {"rows": (rows, "count"), "cols": (cols, "count"),
                                "link_bandwidth": (bandwidth, "B/ns"), "hop_latency": (latency, "ns")})


def bus(name="bus", *, bandwidth=64.0, latency=1.0):
    return point(name, "bus", {"bandwidth": (bandwidth, "B/ns"), "latency": (latency, "ns")})


def dmc_adf(num_cores=4):
    if type(num_cores) is not int or num_cores < 1:
        raise ValueError("num_cores must be a positive integer")
    compute = point("vector", "vector_unit", {"lanes": (16, "count"), "sfu_lanes": (4, "count"),
                    "clock_ghz": (1.0, "GHz"), "precision": ("fp16", "1")})
    core = SpaceMatrix("core", [compute, memory()], [bus()], repeat=num_cores)
    return SpaceMatrix("chip", [core, memory("dram", technology="ddr", capacity=2**30,
                                           bandwidth=64.0, latency=100.0)], [mesh(cols=num_cores + 1)])


def non_gpu_adf(*, num_cores=1, matrix_precision="fp16", dram_capacity_bytes=64 * 2**30):
    """Explicit synthetic systolic/vector hardware, with no GPU timing meaning.

    The array accepts the selected matrix operand precision. Each core also
    has FP32 matrix/vector/SFU capability for mixed-precision operators and
    reductions. Capacity includes FP32 initialization and converted weights;
    workload/operator precision remains owned by the workload contract.
    """
    if type(num_cores) is not int or num_cores < 1:
        raise ValueError("num_cores must be a positive integer")
    if type(dram_capacity_bytes) is not int or dram_capacity_bytes < 1:
        raise ValueError("dram_capacity_bytes must be a positive integer")
    if type(matrix_precision) is not str or matrix_precision not in ("fp16", "bf16", "fp32"):
        raise ValueError("matrix_precision must be fp16, bf16 or fp32")
    source = "synthetic non-GPU systolic/vector scenario; not measured hardware or a GPU model"

    def declared(name, component, values):
        return point(name, component, values, source=source)

    array = declared("array", "systolic_array", {
        "rows": (32, "count"), "cols": (32, "count"), "clock_ghz": (1.0, "GHz"),
        "dataflow": ("ws", "1"), "stationary_buffering": ("double", "1"),
        "precision": (matrix_precision, "1")})
    vector = declared("vector", "vector_unit", {
        "lanes": (16, "count"), "sfu_lanes": (4, "count"), "clock_ghz": (1.0, "GHz"),
        "precision": ("fp32", "1"), "matrix_kernels": (True, "1")})
    sram = declared("sram", "sram", {"capacity": (4 * 2**20, "B"),
                                      "bandwidth": (128.0, "B/ns"), "latency": (1.0, "ns")})
    local = declared("bus", "bus", {"bandwidth": (128.0, "B/ns"), "latency": (1.0, "ns")})
    core = SpaceMatrix("core", [array, vector, sram], [local], repeat=num_cores)
    dram = declared("dram", "dram", {"technology": ("ddr", "1"), "capacity": (dram_capacity_bytes, "B"),
                                      "bandwidth": (64.0, "B/ns"), "latency": (100.0, "ns")})
    network = declared("noc", "mesh2d", {"rows": (1, "count"), "cols": (num_cores + 1, "count"),
                                          "link_bandwidth": (64.0, "B/ns"), "hop_latency": (1.0, "ns")})
    return SpaceMatrix("chip", [core, dram], [network])
