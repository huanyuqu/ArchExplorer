"""DRAM-PIM presets (NeuPIMs, HSNDI) and the hybrid-bonded DRAM helper.

Both presets build a host plus one DRAM and then run
:func:`arch_explorer.structure.transforms.pim_transform` on that DRAM, so the
PIM structure is exactly what the structure program produces: the DRAM split
into ``channel_i/dram_i/bank_j`` with one ``pim_unit`` and one ``bank_bus``
per bank. Map them with ``pim_offload``.

Every value carries a source and an evidence tag:

* ``[paper table]`` a table or text value of the device's own paper;
* ``[vendor]`` the maker's slides or documentation of the same device;
* ``[config]`` a number in the published simulator configuration of the
  paper (NeuPIMs at commit f299af3; numbers only, no code is used);
* ``[third-party]`` a value another team published for the device (paper or
  simulator configuration);
* ``[preprint, derived]`` computed from an unreviewed preprint's table;
* ``[derived]`` computed from sourced values by the stated formula;
* ``[ArchExplorer §V-B2]`` a value the ArchExplorer paper states for its
  HSNDI study, with the primary source it cites;
* ``[assumed]`` no published value; the stated choice is an assumption and an
  error source;
* ``[not modeled]`` no published value; the quantity is zero.

Values are chosen by source and measurement method only, never by agreement
with validation data.
"""
from __future__ import annotations

from arch_explorer.adf.model import Param, SpaceMatrix, SpacePoint

NEUPIMS = ("Heo et al., 'NeuPIMs: NPU-PIM Heterogeneous Acceleration for Batched LLM Inferencing', ASPLOS'24, "
           "doi:10.1145/3620666.3651380, arXiv:2403.00579")
NEUPIMS_T2 = f"{NEUPIMS}, Table 2 (hardware specification)"
NEUPIMS_REPO = "github.com/casys-kaist/NeuPIMs @ f299af3fc8f2"
NEWTONSIM_INI = f"{NEUPIMS_REPO} extern/NewtonSim/configs/HBM2_8Gb_x128_dualpim.ini"
AIM_HC34 = ("Kwon et al., 'System Architecture and Software Stack for GDDR6-AiM', Hot Chips 34 poster (2022), "
            "doi:10.1109/HCS55958.2022.9895629, PDF p.5 'GDDR6-AiM Feature Summary' and p.6-7")
AIM_SC23 = "SK hynix SC23 exhibitor forum slides (AiMX), PDF p.9 (per die)"
CENT = ("Gu, Khadem et al., 'PIM is All You Need' (CENT), ASPLOS'25, arXiv:2502.07578, Table 4 'GDDR6-PIM Timing "
        "Constraints' and §4.2")
AIM_SIM = "github.com/arkhadem/aim_simulator @ 0f28a07b src/dram/impl/GDDR6.cpp (GDDR6_AiM_timing, tCK 0.5 ns)"
TPU_V1 = ("Jouppi et al., 'In-Datacenter Performance Analysis of a Tensor Processing Unit', ISCA'17, arXiv:1704.04760, "
          "Sec. 2: 'one 64 KiB tile of weights plus one for double-buffering (to hide the 256 cycles it takes to shift "
          "a tile in)'")
ARCHEXPLORER = "ArchExplorer paper §V-B2 (HSNDI)"
IPU_JIA = ("Jia et al., 'Dissecting the Graphcore IPU Architecture via Microbenchmarking', arXiv:1912.03413 "
           "(device: Colossus Mk1 GC2 on a C2 board at 1.6 GHz)")
IPU_HC33 = "Knowles, 'Graphcore Colossus Mk2 IPU', Hot Chips 33 (2021), PDF p.5 (Mk1 line) and p.16"
LI_3D = ("Li et al., 'A Full-Stack Performance Evaluation Infrastructure for 3D-DRAM-based LLM Accelerators' "
         "(ATLAS), arXiv:2604.08044v1, Table 3")

MiB = 2**20
GiB = 2**30

# Hybrid-bonded DRAM densities (ArchExplorer [28] = Li et al.).
BONDED_CAPACITY_PER_MM2 = 25.6 * MiB          # per DRAM layer
BONDED_BANDWIDTH_PER_MM2 = 20.48              # B/ns (GB/s) per mm^2 of stack footprint
BONDED_MAX_LAYERS = 4


def _p(value, unit, source):
    return Param(value, unit, source)


def bonded_dram(area_mm2, layers=1, *, name="bonded") -> SpacePoint:
    """A hybrid-bonded DRAM shard of ``area_mm2`` footprint and ``layers`` DRAM dies (``technology="bonded"``).

    Capacity = 25.6 MiB/mm^2 per layer x area x layers; bandwidth = 20.48
    GB/s/mm^2 x area. Both are derived from Li et al. Table 3 (8192 banks of
    2.5 MiB per 800 mm^2 die; 16 cores x 16 channels x 64 GB/s over 800 mm^2);
    the bandwidth density is the 4-die stack's I/O over its footprint and is
    set by the pin budget, so it does not grow with ``layers``. ``layers`` is
    limited to the 4 dies of that stack. Access latency is not published
    (timing withheld by the authors) and is not modeled. Use with
    ``bond_memory(adf, core, bonded_dram(...))``: every core instance gets its
    own shard.
    """
    if isinstance(area_mm2, bool) or not isinstance(area_mm2, (int, float)) or not area_mm2 > 0:
        raise ValueError("area_mm2 must be positive")
    if type(layers) is not int or not 1 <= layers <= BONDED_MAX_LAYERS:
        raise ValueError(f"layers must be an integer in 1..{BONDED_MAX_LAYERS} (the stack height of {LI_3D})")
    capacity = int(BONDED_CAPACITY_PER_MM2 * area_mm2 * layers)
    return SpacePoint(name, "dram", {
        "technology": _p("bonded", "1", f"hybrid-bonded 3D DRAM; {LI_3D}"),
        "capacity": _p(capacity, "B", f"[preprint, derived] 25.6 MiB/mm^2 per layer (8192 x 2.5 MiB / 800 mm^2) "
                                      f"x {area_mm2} mm^2 x {layers} layer(s); {LI_3D}"),
        "bandwidth": _p(round(BONDED_BANDWIDTH_PER_MM2 * area_mm2, 3), "B/ns",
                        f"[preprint, derived] 20.48 GB/s/mm^2 (16 x 16 x 64 GB/s / 800 mm^2 footprint of the 4-die "
                        f"stack; set by pins, not layers) x {area_mm2} mm^2; {LI_3D}"),
        "latency": _p(0.0, "ns", f"[not modeled] DRAM timing withheld ('provided by our industry collaborators'); "
                                 f"{LI_3D} §3.4"),
    })


def pim_structure(dram, levels, counts, pim, network, registry=None) -> SpaceMatrix:
    """The matrix ``pim_transform(adf, <dram>, levels, counts, {compute: pim, network: network})`` puts in place
    of the DRAM point, built directly.

    ``pim_transform`` replays one checked basic edit per container, which
    takes about a minute for 32 x 32 banks; the result is the same tree (a
    test compares the two). The DRAM is split with its component's own
    ``split``, so extensive parameters divide exactly as in the transform.
    """
    from copy import deepcopy
    from arch_explorer.adf.registry import COMPONENTS
    registry = registry or COMPONENTS

    def partition(point, depth):
        groups = []
        for i, child in enumerate(registry.get(point.component).split(point, counts[depth], "even")):
            if depth + 1 < len(counts):
                groups.append(SpaceMatrix(f"{levels[depth]}_{i}", [partition(child, depth + 1)]))
            else:
                groups.append(SpaceMatrix(f"{levels[depth]}_{i}", [child, deepcopy(pim)], [deepcopy(network)]))
        return SpaceMatrix(point.name, groups, repeat=point.repeat)
    return partition(dram, 0)


def _pim(adf, memory, channels, banks, pim, bank_bus):
    levels, counts = (["channel", "bank"], [channels, banks]) if channels > 1 else (["bank"], [banks])
    elements = adf.elements
    index = next(i for i, e in enumerate(elements) if e.name == memory)
    elements[index] = pim_structure(elements[index], levels, counts, pim, bank_bus)
    return adf


# -- NeuPIMs ----------------------------------------------------------------------------
NEUPIMS_ARRAYS = {
    128: f"[paper table] 128 x 128; {NEUPIMS_T2}",
    64: "[paper figure condition] 64 x 64, the second array size of ArchExplorer Fig. 7(j); other NPU values as "
        f"{NEUPIMS_T2}",
}


def neupims(array=128, channels=32, banks=32) -> SpaceMatrix:
    """NeuPIMs (ASPLOS'24): an NPU with 8 systolic arrays and 8 vector units over 32 channels of dual-row-buffer HBM-PIM.

    ``array`` is the systolic array size (128 per Table 2, or 64);
    ``channels``/``banks`` default to Table 2 and exist for scaled tests.
    """
    if array not in NEUPIMS_ARRAYS:
        raise ValueError(f"array must be one of {sorted(NEUPIMS_ARRAYS)}")
    sa = SpacePoint("sa", "systolic_array", {
        "rows": _p(array, "count", NEUPIMS_ARRAYS[array]),
        "cols": _p(array, "count", NEUPIMS_ARRAYS[array]),
        "clock_ghz": _p(1.0, "GHz", f"[config] core_freq 1000 MHz; {NEUPIMS_REPO} configs/systolic_ws_128x128_dev.json"),
        "dataflow": _p("ws", "1", f"[config] core_type systolic_ws; {NEUPIMS_REPO} configs/systolic_ws_128x128_dev.json"),
        "precision": _p("fp16", "1", f"[assumed] config precision = 2 bytes ({NEUPIMS_REPO}); the float format is "
                                     "not stated, fp16 taken"),
        "stationary_buffering": _p("double", "1", f"[simulator] SystolicWS::issue_ex_inst overlaps each GEMM_PRELOAD "
                                                  "with the previous instruction and charges the full preload only "
                                                  f"on an empty pipeline ({NEUPIMS_REPO} src/SystolicWS.cc); weight "
                                                  f"double buffering as in {TPU_V1}"),
    }, repeat=8)
    vu = SpacePoint("vu", "vector_unit", {
        "matrix_kernels": _p(False, "1", f"[simulator] {NEUPIMS_REPO} src/NeuPIMSystolicWS.cc "
                                         "issue_ex_inst: GEMM/PRELOAD use compute pipeline, not vector pipelines"),
        "lanes": _p(128, "count", f"[paper table] vector unit 128 x 1, 8 per chip; {NEUPIMS_T2}"),
        "sfu_lanes": _p(128, "count", f"[config] exp_latency 1 cycle per lane; {NEUPIMS_REPO} "
                                      "configs/systolic_ws_128x128_dev.json"),
        "clock_ghz": _p(1.0, "GHz", f"[config] core_freq 1000 MHz; {NEUPIMS_REPO}"),
        "precision": _p("fp16", "1", "[assumed] as the systolic arrays"),
    }, repeat=8)
    channel_bandwidth = 128 / 8 * 2 * 1.0      # 128-bit channel, 2 transfers per 1 GHz clock -> 32 B/ns
    hbm = SpacePoint("hbm", "dram", {
        "technology": _p("hbm", "1", f"[paper table] HBM; {NEUPIMS_T2}"),
        "capacity": _p(channels * GiB, "B", f"[paper table] 1 GB per channel x {channels} channels; {NEUPIMS_T2}"),
        "bandwidth": _p(channels * channel_bandwidth, "B/ns",
                        f"[derived] {channels} channels x 128-bit x 2 transfers/clock x 1 GHz = {channel_bandwidth} GB/s "
                        f"each (device_width 128, {NEWTONSIM_INI}; frequency 1 GHz, {NEUPIMS_T2}) "
                        "[peak, not measured]"),
        "latency": _p(28.0, "ns", f"[derived] tRCD 14 + CL 14 cycles at 1 GHz ({NEUPIMS_T2}; CL from {NEWTONSIM_INI})"),
        "trcd": _p(14.0, "ns", f"[paper table] tRCD = 14 cycles at 1 GHz; {NEUPIMS_T2}"),
        "trp": _p(14.0, "ns", f"[paper table] tRP = 14; {NEUPIMS_T2}"),
        "tras": _p(34.0, "ns", f"[paper table] tRAS = 34; {NEUPIMS_T2}"),
        "trrd": _p(6.0, "ns", f"[paper table] tRRD_L = 6; {NEUPIMS_T2}"),
        "tfaw": _p(30.0, "ns", f"[paper table] tFAW = 30; {NEUPIMS_T2}"),
        "tccd": _p(2.0, "ns", f"[paper table] tCCD_L = 2 (all-bank commands span bank groups; the MAC rate gives "
                              f"the same 2 ns per column); {NEUPIMS_T2}"),
        "tcl": _p(14.0, "ns", f"[config] CL = 14 cycles, tCK = 1 ns; {NEWTONSIM_INI}"),
        "row_bytes": _p(1024, "B", f"[paper table] page size 1 KB; {NEUPIMS_T2}"),
        "column_bytes": _p(64, "B", f"[derived] device_width 128 bit x BL 4 / 8 ({NEWTONSIM_INI})"),
        "trfc": _p(260.0, "ns", f"[paper table] tRFC = 260 cycles at 1 GHz; {NEUPIMS_T2}"),
        "trefi": _p(3900.0, "ns", f"[paper table] tREFI = 3900; {NEUPIMS_T2}"),
    })
    fabric = SpacePoint("npu_fabric", "bus", {
        "bandwidth": _p(channels * channel_bandwidth, "B/ns",
                        "[derived] equal to the HBM bandwidth: the paper gives no NPU interconnect bandwidth, so the "
                        "fabric delivers what the channels supply"),
        "latency": _p(1.0, "ns", f"[config] icnt_latency 1 cycle at 1 GHz; {NEUPIMS_REPO} "
                                 "configs/systolic_ws_128x128_dev.json"),
    })
    pim = SpacePoint("pim", "pim_unit", {
        "ops_per_cycle": _p(32, "ops/cycle", f"[config][derived] pim_comp_coverage 16 multipliers per bank x 2 "
                                             f"({NEUPIMS_REPO} configs/memory_configs/neupims.json)"),
        "clock_ghz": _p(1.0, "GHz", f"[config] dram_freq 1000 MHz; {NEUPIMS_REPO} configs/memory_configs/neupims.json"),
        "precision": _p("fp16", "1", "[assumed] as the NPU"),
        "act_banks": _p(4, "count", f"[paper table] PIM_ACTIVATION opens 'usually 4 banks at a time due to the power "
                                    f"constraints (i.e., tFAW)'; {NEUPIMS} §5.2"),
        "row_buffer": _p("dual", "1", f"[paper table] MEM and PIM row buffer per bank; {NEUPIMS} §5.1"),
        "command_elements": _p(16, "count", f"[config] pim_comp_coverage 16 elements per COMP command "
                                            f"({NEUPIMS_REPO} configs/memory_configs/neupims.json; COMP commands "
                                            "tCCD apart in extern/NewtonSim src/timing.cc)"),
    })
    bank_bus = SpacePoint("bank_bus", "bus", {
        "bandwidth": _p(32.0, "B/ns", "[derived] one 64 B column per tCCD_L = 2 ns from the bank to the channel I/O"),
        "latency": _p(0.0, "ns", "[derived] access latency is carried by the DRAM latency"),
    })
    adf = SpaceMatrix("neupims", [sa, vu, hbm], [fabric])
    for key, value, unit, explanation in (
        ("ranks", 2 if banks % 8 == 0 else 1, "count", "extern/NewtonSim/src/configuration.cc: 2 ranks x 16 banks; scaled layouts use complete groups"),
        ("refresh_mode", "rank_staggered", "1", "extern/NewtonSim/src/refresh.cc: RANK_LEVEL_STAGGERED; absolute clock, first rank at tREFI/ranks"),
        ("completion_mode", "commands", "1", "extern/NewtonSim/src/neupims_controller.cc::IssueCommand; channel_state.cc::UpdateChannelTiming"),
        ("attention_pipeline", "heads", "1", "src/operations/NeuPIMSLogitSoftmax.cc and NeuPIMSAttend.cc: per-head consumers"),
        ("write_response", "early", "1", "extern/NewtonSim/src/neupims_controller.cc::AddTransaction: WRITE response accept+1 tick"),
        ("tccd_s", 1., "ns", "extern/NewtonSim/configs/HBM2_8Gb_x128_dualpim.ini: tCCD_S=1; GWRITE delay=(tCCD_S+tCCD_L)*32"),
        ("burst_ns", 2., "ns", "extern/NewtonSim/src/configuration.cc: BL=4 / 2 at tCK=1; read_delay=CL+burst_cycle"),
        ("trtp", 5., "ns", "extern/NewtonSim/src/configuration.cc: tRTP defaults to 5 (not tRTP_L/S), AL=0"),
    ):
        pim.params[key] = _p(value, unit, f"[simulator] {NEUPIMS_REPO} {explanation}")
    return _pim(adf, "hbm", channels, banks, pim, bank_bus)


# -- HSNDI ------------------------------------------------------------------------------
HSNDI_CORE_RATES = {
    # The §V-B2 statement: one AMP unit at 16 FP32 operations/cycle per core (Mk1 value of Jia et al.).
    "paper": (8, f"[ArchExplorer §V-B2] one AMP unit at 16 FP32 ops/cycle = 8 FMA lanes ({ARCHEXPLORER}, citing "
                 f"Graphcore; the value is the Mk1 FP32 rate of {IPU_JIA}); applied to 16-bit operands"),
    # Mk1 16-bit AMP rate.
    "mk1_fp16": (32, f"[third-party] AMP 64 mixed-precision ops/cycle = 32 FMA lanes; {IPU_JIA} §1.3"),
}


def hsndi(cores=196, channels=8, core_rate="paper", precision="bf16") -> SpaceMatrix:
    """HSNDI (ArchExplorer §V-B2): SRAM-PNM cores (Graphcore IPU tiles) beside GDDR6-AiM DRAM-PIM channels.

    ``cores`` SRAM-PNM cores and ``channels`` DRAM-PIM channels (the §V-B2
    study varies 0-270 cores and 16-5 channels; Fig. 12 uses 196 and 8).
    ``core_rate`` selects one of :data:`HSNDI_CORE_RATES`. ``precision`` is the
    operand format of the workload: AiM computes BF16.
    """
    if type(cores) is not int or cores < 1:
        raise ValueError("cores must be a positive integer (the cores host every non-attention operator)")
    if type(channels) is not int or channels < 1:
        raise ValueError("channels must be a positive integer")
    if core_rate not in HSNDI_CORE_RATES:
        raise ValueError(f"core_rate must be one of {sorted(HSNDI_CORE_RATES)}")
    clock = 1.6
    lanes, rate_source = HSNDI_CORE_RATES[core_rate]
    amp = SpacePoint("amp", "vector_unit", {
        "lanes": _p(lanes, "count", rate_source),
        "sfu_lanes": _p(lanes, "count", "[assumed] no published transcendental rate for the IPU tile; equal to the "
                                        "AMP lanes"),
        "clock_ghz": _p(clock, "GHz", f"[third-party] measured test system at 1.6 GHz; {IPU_JIA}"),
        "precision": _p(precision, "1", "workload operand format (AiM computes BF16)"),
    })
    sram = SpacePoint("sram", "sram", {
        "capacity": _p(256 * 1024, "B", f"[ArchExplorer §V-B2] 256 KB per core ({ARCHEXPLORER}); Mk1 tile 256 KiB, "
                                       f"{IPU_HC33}"),
        "bandwidth": _p(round(31.1e3 / 1216, 2), "B/ns",
                        f"[third-party][derived] 31.1 TB/s aggregate peak read / 1216 tiles; {IPU_JIA}"),
        "latency": _p(round(6 / clock, 3), "ns", f"[third-party] 6 cycles at 1.6 GHz; {IPU_JIA} §1.4"),
    })
    core_bus = SpacePoint("core_bus", "bus", {
        "bandwidth": _p(round(31.1e3 / 1216, 2), "B/ns", "[derived] equal to the tile SRAM bandwidth"),
        "latency": _p(0.0, "ns", "[derived] carried by the SRAM latency"),
    })
    core = SpaceMatrix("core", [amp, sram], [core_bus], repeat=cores)
    dram = SpacePoint("dram", "dram", {
        "technology": _p("gddr", "1", f"[vendor] GDDR6-AiM; {AIM_HC34}"),
        "capacity": _p(channels * 512 * MiB, "B", f"[vendor] 4 Gb (512 MiB) per die = one channel of 16 banks x "
                                                  f"32 MB; {AIM_HC34}; {AIM_SC23}"),
        "bandwidth": _p(channels * 32.0, "B/ns", f"[vendor] 32 GB/s external per channel (16 pins x 16 Gb/s) "
                                                 f"[peak, not measured]; {AIM_SC23}"),
        "latency": _p(43.0, "ns", f"[third-party][derived] tRCDRD 18 + tCL 25 ns; {CENT}"),
        "trcd": _p(18.0, "ns", f"[third-party] tRCDRD = 18 ns; {CENT}"),
        "trp": _p(16.0, "ns", f"[third-party] tRP = 16 ns; {CENT}"),
        "tras": _p(27.0, "ns", f"[third-party] tRAS = 27 ns; {CENT}"),
        "trrd": _p(5.5, "ns", f"[third-party] nRRDL = 11 cycles of 0.5 ns; {AIM_SIM}"),
        "tfaw": _p(21.0, "ns", f"[third-party] nFAW = 42 cycles of 0.5 ns, marked not used: ACT16 opens all 16 banks "
                               f"at once (reservoir capacitors, {CENT} §4.2); {AIM_SIM}"),
        "tccd": _p(1.0, "ns", f"[third-party] tCCDS = 1 ns (one all-bank MAC per 1 GHz PU cycle); {CENT}"),
        "tcl": _p(25.0, "ns", f"[third-party] tCL = 25 ns; {CENT}"),
        "row_bytes": _p(2048, "B", f"[vendor] 2 KB row (1024 BF16); {AIM_HC34} and HC35 PDF p.15"),
        "column_bytes": _p(32, "B", f"[vendor] one column access = 32 B = 16 BF16; {AIM_HC34} PDF p.6"),
        "trfc": _p(105.0, "ns", f"[third-party] nRFC = 210 cycles of 0.5 ns (the same cycle counts as the simulator's "
                                f"generic GDDR6 presets); {AIM_SIM}. The AiM controller schedules refresh against AiM "
                                f"commands (Kwon et al., Hot Chips 34, PDF p.12 'Refresh Engine' and p.25)"),
        "trefi": _p(1666.5, "ns", f"[third-party] nREFI = 3333 cycles of 0.5 ns; {AIM_SIM}"),
    })
    pim = SpacePoint("pim", "pim_unit", {
        "ops_per_cycle": _p(32, "ops/cycle", f"[vendor] 16 BF16 MACs per bank PU = 32 ops/cycle; {AIM_HC34}"),
        "clock_ghz": _p(1.0, "GHz", f"[vendor] PU operating speed 1 GHz; {AIM_HC34}"),
        "precision": _p("bf16", "1", f"[vendor] BF16; {AIM_HC34}"),
        "act_banks": _p(16, "count", f"[vendor] ACT16 activates sixteen banks in parallel; {AIM_HC34} PDF p.7"),
        "row_buffer": _p("shared", "1", "[vendor] one row buffer per bank: PIM work blocks normal access to the bank "
                                        f"(AiM single row buffer; NeuPIMs §5.1 'blocked mode')"),
    })
    bank_bus = SpacePoint("bank_bus", "bus", {
        "bandwidth": _p(32.0, "B/ns", f"[derived] one 32 B column per tCCDS = 1 ns ({AIM_HC34}; {CENT})"),
        "latency": _p(0.0, "ns", "[derived] access latency is carried by the DRAM latency"),
    })
    exchange = SpacePoint("exchange", "bus", {
        "bandwidth": _p(round(cores * 4 * clock, 2), "B/ns",
                        f"[vendor][derived] {cores} tiles x 32 bit/cycle send x 1.6 GHz ({IPU_HC33} p.16 'Tile 32b/cycle "
                        "send and receive'); one shared bus stands for the exchange, DRAM traffic is bounded by the "
                        "DRAM ports"),
        "latency": _p(0.0, "ns", "[not modeled] exchange latency not published per transfer"),
    })
    adf = SpaceMatrix("hsndi", [core, dram], [exchange])
    return _pim(adf, "dram", channels, 16, pim, bank_bus)


def hsndi_detail(cores=196, channels=8, *, refresh_service_ns=None, refresh_source=None):
    """Explicit source-initialized AiM configuration, separate from aggregate hsndi.

    Refresh service has no valid source default. Callers must supply a finite
    positive value in ns and its source/assumption. See docs/hsndi-detail.md.
    Host parameters are inherited for structure/cost only, not host timing.
    """
    from arch_explorer.components.aim_profile import AimProfile, SOURCE
    profile = AimProfile(refresh_service_ns=refresh_service_ns, refresh_source=refresh_source)
    adf = hsndi(cores=cores, channels=channels)
    adf.name = 'hsndi_detail'
    source = '[third-party][defined_effective_initialization] ' + SOURCE + '/src/dram/impl/GDDR6.cpp:580-716'
    def update(node):
        if isinstance(node, SpaceMatrix):
            for child in (*node.elements, *node.networks):
                update(child)
        elif node.component == 'dram':
            for key, value in profile.bank_values().items():
                prior = node.params[key]
                evidence = source
                if key == 'trfc':
                    evidence = '[caller_configuration] ' + refresh_source
                elif key == 'bandwidth':
                    evidence += '; 32B / (nBL2 * 1ns) / 16 banks'
                elif key == 'tfaw':
                    evidence += '; nFAW initialized to28 but source four-ACT constraint disabled'
                elif key == 'trrd':
                    evidence += '; same-group nRRDL6; cross-group nRRDS5 is in detailed profile'
                node.params[key] = _p(value, prior.unit, evidence)
        elif node.component == 'pim_unit':
            node.params['clock_ghz'] = _p(0.5, 'GHz', source + '; issue rate: one 16-BF16 MAC/bank per nCCDL2 * 1ns; not vendor PU clock')
        elif node.name == 'bank_bus':
            node.params['bandwidth'] = _p(16., 'B/ns', source + '; 32B column / (nCCDL2 * 1ns)')
    update(adf)
    return adf


PRESETS = {"neupims": neupims, "hsndi": hsndi, "hsndi_detail": hsndi_detail}

__all__ = ["BONDED_MAX_LAYERS", "HSNDI_CORE_RATES", "PRESETS", "bonded_dram", "hsndi", "hsndi_detail", "neupims"]
