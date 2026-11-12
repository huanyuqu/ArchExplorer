"""Sourced single-chip scenarios, not an assertion of benchmark machine identity."""
from arch_explorer.adf import Param, SpaceMatrix, SpacePoint

HC = "[vendor peak] Knowles, Hot Chips 2021, slides 8-16 https://hc33.hotchips.org/assets/program/conference/day2/HC2021.Graphcore.SimonKnowles.v04.pdf"
TIAN = "[primary paper] Ma et al. Sci Robot 2022, Table1, Fig5 and p7 https://doi.org/10.1126/scirobotics.abk2948"
FORMAT = "[primary implementation study] Wang et al. 2025 Sec4.1/Fig7: int8 operands, int32 partial sums https://doi.org/10.3389/fnins.2025.1512926"


def _p(value, unit, source):
    return Param(value, unit, source)


def _tile(name, kind, count, compute, capacity, bandwidth):
    return SpaceMatrix(name, [SpacePoint("compute", kind, compute), SpacePoint("sram", "sram", {
        "capacity": capacity, "bandwidth": bandwidth,
        "latency": _p(0.0, "ns", "[not modeled] SRAM startup/instruction latency; throughput-only bound"),
    })], repeat=count)


def ipu_gc200(remote_bandwidth=None, remote_capacity=None):
    """One GC200 at 1.325GHz; remote attachment requires two explicitly sourced Params.

    No default remote throughput is inferred from M2000 or Bow metadata.
    Host loading/compilation are excluded; all ordinary inputs start in tile SRAM.
    """
    clock, count = 1.325, 1472
    compute = {
        "clock_ghz": _p(clock, "GHz", HC + "; slide9"),
        "precision": _p("fp16", "1", HC + "; slide13 IEEE f16 operands"),
        "accumulation": _p("fp32", "1", HC + "; slide13 internal datapath"),
        "output_precision": _p("fp16", "1", HC + "; slide13 memory f16, burst250TFLOP/s"),
        "macs_per_cycle": _p(64, "MAC/cycle", "[vendor peak] GC200 AMP up to64 MAC/cycle; FP16 stored output, FP32 internal accumulation https://docs.graphcore.ai/projects/ipu-programmers-guide/en/latest/about_ipu.html"),
        "vector_ops_per_cycle": _p(2, "ops/cycle", "[vendor] 2 FP32 vector arithmetic lanes/cycle (not SFU rate) https://www.graphcore.ai/posts/exploiting-tile-level-parallelism-on-graphcore-ipus"),
        "sync_cycles": _p(150, "cycle", HC + "; slide15 approximately150cycles, used once per modeled BSP boundary"),
        "identity": _p("GC200; M2000 contains four GC200; target measurement SKU unknown", "1", HC),
    }
    tile = _tile("tile", "ipu_tile", count, compute, _p(624 * 1024, "B", HC + "; slide8 624KiB/tile"),
                 _p(47000 / count, "B/ns", HC + "; slide12 rounded47TB/s chip data-side /1472 uniform tiles; combined service ceiling"))
    elements = [tile]
    if (remote_bandwidth is None) != (remote_capacity is None):
        raise ValueError("remote_bandwidth and remote_capacity must both be sourced Params")
    if remote_bandwidth is not None:
        for p, unit in ((remote_bandwidth, "B/ns"), (remote_capacity, "B")):
            if not isinstance(p, Param) or not p.source or p.unit != unit:
                raise ValueError(f"remote attachment requires a sourced Param in {unit}")
        elements.append(SpacePoint("remote", "dram", {
            "technology": _p("ddr", "1", "[scenario] explicitly attached shared streaming memory; not fabricated on-die DRAM"),
            "capacity": remote_capacity, "bandwidth": remote_bandwidth,
            "latency": _p(0.0, "ns", "[not modeled] remote command/startup delay and gateway topology"),
        }))
    return SpaceMatrix("ipu_gc200", elements, [SpacePoint("exchange", "tile_exchange", {
        "tiles": _p(count, "count", HC), "topology": _p("all_to_all", "1", HC + "; slide16 stateless all-to-all"),
        "endpoint_bandwidth": _p(4 * clock, "B/ns", HC + "; slide16 32bit/cycle send AND receive, separate endpoint resources"),
    })])


def tianjicx(capacity_unit="decimal"):
    """Int8/int32 scenario. Units are unresolved: decimal (default conservative) or binary.

    Data allocation uses the two64kB blocks from the implementation study;
    total physical144kB includes other blocks. Mesh link timing is omitted.
    """
    if capacity_unit not in ("decimal", "binary"):
        raise ValueError("capacity_unit must be decimal or binary")
    unit = 1000 if capacity_unit == "decimal" else 1024
    compute = {
        "clock_ghz": _p(0.4, "GHz", TIAN), "precision": _p("int8", "1", FORMAT),
        "accumulation": _p("int32", "1", FORMAT), "output_precision": _p("int32", "1", FORMAT),
        "macs_per_cycle": _p(128, "MAC/cycle", TIAN + ";128 MAC/core;2*128*160*0.4=16.384TOP/s approximate reported16"),
        "identity": _p("TianjicX 28nm,160 cores; target dtype unknown", "1", TIAN),
    }
    core = _tile("core", "tianjic_core", 160, compute,
                 _p(128 * unit, "B", FORMAT + f";[capacity scenario]2x64kB data blocks, {capacity_unit} unit; total144kB includes other blocks"),
                 _p(5120 / 160, "B/ns", TIAN + ";p7 local SRAM access5.12TB/s /160 uniform cores; not mesh link throughput"))
    return SpaceMatrix("tianjicx", [core], [SpacePoint("mesh", "tile_exchange", {
        "tiles": _p(160, "count", TIAN), "topology": _p("mesh_unknown", "1", TIAN + ";link widths/rates not established; no fabricated link capacity"),
        "rows": _p(16, "count", TIAN), "cols": _p(10, "count", TIAN),
    })])


PRESETS = {"ipu_gc200": ipu_gc200, "tianjicx": tianjicx}
