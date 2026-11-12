"""Pinned VectorCGRA single-array scenarios, not the unpublished paper configuration."""
import math

from arch_explorer.adf.model import Param, SpaceMatrix, SpacePoint

RTL = "github.com/tancheng/VectorCGRA@20ea53bfd54980acf5bc6eef4b0959a710a4dd1f"
TEST = f"{RTL}/cgra/test/CgraRTL_gemm_test_from_yaml.py:166-186"


def vectorcgra_4x4(*, diagonal=False, clock_ghz=None, words_per_bank=32):
    """Two 32-word banks as in the pinned test; larger resident workloads need an explicit capacity scenario.

    An omitted clock uses 1 cycle per reporting ns, NOT a measured 1 GHz clock.
    ``words_per_bank`` must be a power of two (RTL contiguous address decode).
    """
    if type(diagonal) is not bool:
        raise ValueError("diagonal must be boolean")
    if type(words_per_bank) is not int or words_per_bank < 1 or words_per_bank & (words_per_bank - 1):
        raise ValueError("words_per_bank must be a positive power of two")
    known = clock_ghz is not None
    clock = 1.0 if clock_ghz is None else clock_ghz
    if type(clock) not in (int, float) or not math.isfinite(clock) or clock <= 0:
        raise ValueError("clock_ghz must be positive and finite")
    p = Param
    clock_source = ("[scenario] caller-supplied physical clock, not paper-verified" if known else
                    "[unknown] physical clock unpublished; 1 cycle/reporting-ns is a normalization only")
    pe = SpacePoint("pe", "cgra_pe", {"clock_ghz": p(clock, "GHz", clock_source),
                                       "precision": p("uint32", "1", f"[RTL config] data_bitwidth=32; {TEST}")}, repeat=16)
    mesh = SpacePoint("mesh", "mesh2d", {
        "rows": p(4, "count", f"[RTL config] {TEST}"),
        "cols": p(4, "count", f"[RTL config] {TEST}"),
        "diagonal": p(diagonal, "1", f"[RTL] MESH/KING_MESH; {RTL}/cgra/CgraRTL.py:61-68,192-268"),
        "link_bandwidth": p(4 * clock, "B/ns", f"[derived] one uint32 token/tick; {RTL}/tile/TileRTL.py:119"),
        "hop_latency": p(1 / clock, "ns", f"[RTL] input ChannelRTL latency=1; {RTL}/tile/TileRTL.py:119"),
    })
    capacity_source = (f"[RTL config] 32 words/bank; {TEST}" if words_per_bank == 32 else
                       "[scenario] caller-selected local SRAM capacity; not paper-verified")
    banks = SpacePoint("bank", "sram", {
        "capacity": p(4 * words_per_bank, "B", capacity_source),
        "bandwidth": p(4 * clock, "B/ns", f"[derived] one uint32 read AND write/tick/bank; {RTL}/mem/data/DataMemWrapperRTL.py:45-46"),
        "latency": p(0.0, "ns", f"[RTL config] combinational memory; {RTL}/cgra/test/CgraRTL_gemm_test_from_yaml.py:423"),
    }, repeat=2)
    return SpaceMatrix("vectorcgra", [SpaceMatrix("array", [pe], [mesh]), banks])


def vectorcgra_4x4_diag(**options):
    return vectorcgra_4x4(diagonal=True, **options)


PRESETS = {"vectorcgra_4x4": vectorcgra_4x4, "vectorcgra_4x4_diag": vectorcgra_4x4_diag}
