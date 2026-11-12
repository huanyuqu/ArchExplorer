"""Public ADF/workload/mapping objects; implementation choice survives JSON."""
from arch_explorer.adf import MappingSpec, Param, SpaceMatrix, SpacePoint, WorkloadSpec
from arch_explorer.components.api import ImplementationBinding


SOURCE = "explicit synthetic external SM example; not measured GPU parameters"


def serial_scene(*, ops_per_ns=4.0, startup_ns=3.0, m=2, k=2, n=2):
    for value in (m, k, n):
        if type(value) is not int or value < 1:
            raise ValueError("GEMM dimensions must be positive integers")
    sm = SpacePoint("compute", "sm", {"precision": Param("fp32", "1", SOURCE)},
                    implementation=ImplementationBinding("example.sm.serial", "1", {
                        "ops_per_ns": Param(ops_per_ns, "op/ns", SOURCE),
                        "startup_ns": Param(startup_ns, "ns", SOURCE),
                    }))
    memory = SpacePoint("sram", "sram", {
        "capacity": Param(4096, "B", SOURCE),
        "bandwidth": Param(16.0, "B/ns", SOURCE),
        "latency": Param(1.0, "ns", SOURCE),
    })
    core_bus = SpacePoint("core_bus", "bus", {
        "bandwidth": Param(16.0, "B/ns", SOURCE), "latency": Param(1.0, "ns", SOURCE)})
    core = SpaceMatrix("core", [sm], [core_bus])
    dram = SpacePoint("dram", "dram", {
        "technology": Param("ddr", "1", SOURCE),
        "capacity": Param(65536, "B", SOURCE),
        "bandwidth": Param(16.0, "B/ns", SOURCE),
        "latency": Param(10.0, "ns", SOURCE),
    })
    device_bus = SpacePoint("device_bus", "bus", {
        "bandwidth": Param(16.0, "B/ns", SOURCE), "latency": Param(1.0, "ns", SOURCE)})
    adf = SpaceMatrix("device", [core, memory, dram], [device_bus])
    workload = WorkloadSpec("operator", {"op": "matmul", "m": m, "k": k, "n": n, "dtype": "fp32"})
    mapping = MappingSpec("gsm_tiled", {"tile_m": 64, "tile_n": 64, "rows_per_cta": 4})
    return adf, workload, mapping
