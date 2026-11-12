"""A user-authored ADF using only existing component types."""
import math

from arch_explorer.adf import Param, SpaceMatrix, SpacePoint


def custom_adf(*, memory_bytes=32768, lanes=4, clock_ghz=1.0):
    if type(memory_bytes) is not int or memory_bytes < 1:
        raise ValueError("memory_bytes must be a positive integer")
    if type(lanes) is not int or lanes not in (2, 4, 8, 16, 32):
        raise ValueError("example lanes must be one of 2, 4, 8, 16, 32")
    if type(clock_ghz) not in (int, float) or not math.isfinite(clock_ghz) or clock_ghz <= 0:
        raise ValueError("clock_ghz must be finite and positive")
    source = "synthetic external-operator teaching assumption; not measured hardware"
    def param(value, unit):
        return Param(value, unit, source)
    units = [SpacePoint(name, "vector_unit", {
        "lanes": param(lanes, "count"), "sfu_lanes": param(1, "count"),
        "clock_ghz": param(clock_ghz, "GHz"), "precision": param("fp32", "1"),
        "matrix_kernels": param(False, "1")}) for name in ("vector_a", "vector_b")]
    memory = SpacePoint("shared", "sram", {"capacity": param(memory_bytes, "B"),
        "bandwidth": param(32.0, "B/ns"), "latency": param(1.0, "ns")})
    host = SpacePoint("host", "dram", {"capacity": param(2**20, "B"),
        "bandwidth": param(16.0, "B/ns"), "latency": param(4.0, "ns"),
        "technology": param("ddr", "1")})
    bus = SpacePoint("fabric", "bus", {"bandwidth": param(64.0, "B/ns"), "latency": param(0.0, "ns")})
    return SpaceMatrix("operator_board", [*units, memory, host], [bus])
