"""Public model parameters, not measurements or current market quotations.

Equations are implemented independently; no external simulator is vendored.
"""
from arch_explorer.adf import Param

LLM = "https://github.com/PrincetonUniversity/LLMCompass/blob/2e015fd2ee750e6cad8d7152df52551e5b41ef20/cost_model/cost_model.py"
GA100 = "https://github.com/PrincetonUniversity/LLMCompass/blob/2e015fd2ee750e6cad8d7152df52551e5b41ef20/configs/GA100.json"
SUPPLY = "https://github.com/PrincetonUniversity/ttm-cas/blob/d98ac0586faaf63f8f4eac94aa7cd50f37421c1f/supply_chain_model.py (Khan/Mann AI Chips parameter table, historical model)"
ACTUARY = "https://github.com/Yinxiao-Feng/chiplet-actuary/blob/42dd5deeec2b2f2daf13b1f3ac6042df2d4e3b62/parameter.ini"
PAPER = "ArchExplorer HPCA draft section V-A2; paper-reported assumption, primary normalization audit pending"


def p(value, unit, source):
    return Param(value, unit, source)


CONSTANTS = {
    "process_nm": p(7, "nm", LLM + "; reference-node assumption unless overridden"),
    "density_7": p(96.3, "Mtransistor/mm2", SUPPLY),
    "density_5": p(138.2, "Mtransistor/mm2", SUPPLY),
    "density_90": p(1.6, "Mtransistor/mm2", SUPPLY),
    "sram_cell_7": p(1.70e-7, "mm2/bit", LLM + "; CACTI-derived 7nm model"),
    "sram_cell_5": p(1.25e-7, "mm2/bit", LLM + "; CACTI-derived 5nm model"),
    "cache_min_bytes": p(4096, "B", LLM),
    "cache_unit_bytes": p(2**19, "B", LLM),
    "fpu64_transistors": p(685300, "transistor", LLM + "; Ariane/OpenPiton estimate"),
    "int32_transistors": p(177690, "transistor", LLM + "; Ariane/OpenPiton estimate"),
    "control_per_lane": p(996200, "transistor", LLM + "; per_sublane_control_dict actually uses this value"),
    "network_per_endpoint": p(44300000, "transistor", LLM + "; per_core_comm_dict actually uses this value; endpoint proxy, not routed layout"),
    "hbm_phy": p(10.45, "mm2", LLM + "; per 1024-bit HBM2e interface"),
    "hbm_controller": p(552743000, "transistor", LLM),
    "ddr_phy": p(1.45, "mm2", LLM + "; per 32-bit DDR5 link unit"),
    "ddr_controller": p(90446400, "transistor", LLM),
    "nvlink_phy": p(1.888, "mm2", LLM + "; NVLink3 per link"),
    "nvlink_controller": p(74632000, "transistor", LLM),
    "pcie4_phy": p(0.48, "mm2", LLM + "; per PCIe4 lane"),
    "pcie4_controller": p(3962500, "transistor", LLM + "; per PCIe4 lane"),
    "hbm_usd_gib": p(7, "USD/GiB", LLM + "; Oct 2023 model GB interpreted as marketed binary capacity (80 -> $560)"),
    "ddr_usd_gib": p(2.4, "USD/GiB", LLM + "; Oct 2023 DDR5 assumption; not GDDR6 pricing"),
    "wafer_diameter": p(300, "mm", ACTUARY),
    "wafer_usd_7": p(9346, "USD", SUPPLY),
    "wafer_usd_5": p(16988, "USD", SUPPLY),
    "defects_7": p(0.0009, "1/mm2", ACTUARY + "; 0.09 defects/cm2"),
    "defects_5": p(0.0011, "1/mm2", ACTUARY + "; 0.11 defects/cm2"),
    "critical_levels": p(10, "1", ACTUARY),
    "package_area_scale": p(4, "1", ACTUARY + "; organic substrate scenario"),
    "substrate_usd_mm2": p(0.005, "USD/mm2", ACTUARY + "; organic substrate scenario"),
    "bump_usd_mm2": p(0.005, "USD/mm2", ACTUARY),
    "package_bond_yield": p(0.99, "1", ACTUARY + "; organic substrate scenario, not CoWoS"),
    "bonded_density": p(25.6, "MiB/mm2/layer", PAPER + "; ref 28, https://arxiv.org/abs/2604.08044"),
    "bonded_max_layers": p(4, "count", PAPER + "; ref 28"),
    "bond_yield": p(0.95, "1", PAPER + "; per layer, ref 13"),
    "pim_overhead": p(0.20, "1", PAPER + "; ref 19, incremental DRAM area and proportional price assumption"),
    "aim_die_bytes": p(512 * 2**20, "B", "https://hc34.hotchips.org/assets/program/posters/hc34.SKhynix.YongkeeKwon.v03.pdf p5; 4Gb/die = 16 banks x 32MiB"),
    "aim_units": p(16, "count", "https://hc34.hotchips.org/assets/program/posters/hc34.SKhynix.YongkeeKwon.v03.pdf p5; processing units per die"),
    "aim_ops": p(32, "ops/cycle", "https://hc34.hotchips.org/assets/program/posters/hc34.SKhynix.YongkeeKwon.v03.pdf; 16 MAC/PU/cycle, FMA=2 ops"),
    "aim_clock": p(1, "GHz", "https://hc34.hotchips.org/assets/program/posters/hc34.SKhynix.YongkeeKwon.v03.pdf p5"),
}
for index, value in enumerate((0.076, 0.142, 0.247, 0.393, 0.559, 0.704, 0.526, 0.602, 0.561)):
    CONSTANTS[f"cache_eff_{index}"] = p(value, "1", LLM + "; CACTI lookup for 2**(12+index) bytes")
for precision, bits in (("fp16", 10), ("bf16", 7), ("fp32", 23), ("fp64", 52)):
    CONSTANTS[f"mantissa_{precision}"] = p(bits, "bit", LLM + "; quadratic mantissa scaling; bf16 extension uses its 7 fraction bits")
for key, value in {"bias":32900, "D":-1090, "W":-883, "P":-5550, "DW":53.5,
                   "DD":0.015, "WW":0.0108, "PP":0.586, "DP":142, "WP":368}.items():
    CONSTANTS["rf_" + key] = p(value, "um2 polynomial coefficient", LLM + "; EMPIRE 90nm register-file polynomial")

# Explicit reference microarchitecture assumptions, not inferred device facts.
DEFAULTS = {
    "reference_simt_ops": p(128, "ops/cycle", GA100 + "; 4 sublanes x 16 FP32 FMAs x 2"),
    "fp32_units": p(64, "count", GA100 + "; 4 sublanes x 16"),
    "int32_units": p(64, "count", GA100 + "; 4 sublanes x 16"),
    "fp64_units": p(32, "count", GA100 + "; 4 sublanes x 8"),
    "sublanes": p(4, "count", GA100),
    "vector_width": p(32, "count", GA100),
    "rf_registers": p(16384, "count", GA100),
    "rf_bits": p(32, "bit", GA100),
    "rf_ports": p(4, "count", GA100),
    "local_sram_bytes": p(192 * 1024, "B", GA100),
    "hbm_stack_bytes": p(16 * 2**30, "B", "NVIDIA A100 80GB SXM4: five 16GiB stacks; reference assumption for other HBM"),
    "ddr_channel_bytes": p(2**30, "B", "model scenario: one 32-bit DDR5 interface per GiB; override for real controllers"),
    "bonded_layers": p(4, "count", PAPER + "; default max-layer scenario, override actual stack"),
    "die_count": p(1, "count", "model scenario: one logic die unless an explicit equal-area die count is supplied"),
}
