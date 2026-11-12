"""Analytical equations with unit-explicit, sourced coefficients."""
import math

from .constants import CONSTANTS as C


def sram_area(capacity, node=7):
    if capacity < 0:
        raise ValueError("SRAM capacity must be nonnegative")
    if capacity == 0:
        return 0.0
    unit = min(capacity, C["cache_unit_bytes"].value)
    count = math.ceil(capacity / unit)
    unit = max(unit, C["cache_min_bytes"].value)
    index = math.ceil(math.log2(unit)) - 12
    return count * unit * 8 * C[f"sram_cell_{node}"].value / C[f"cache_eff_{index}"].value


def fpu_area(units, precision, node=7):
    if precision == "int32":
        transistors = C["int32_transistors"].value
    else:
        transistors = C["fpu64_transistors"].value * (
            C[f"mantissa_{precision}"].value / C["mantissa_fp64"].value) ** 2
    return units * transistors / (1e6 * C[f"density_{node}"].value)


def register_area(files, registers, bits, ports, node=7):
    terms = {"bias":1, "D":registers, "W":bits, "P":ports, "DW":registers*bits,
             "DD":registers**2, "WW":bits**2, "PP":ports**2, "DP":registers*ports, "WP":bits*ports}
    area = sum(C["rf_" + k].value * value for k, value in terms.items())
    if area < 0:
        raise ValueError("Register-file polynomial outside its nonnegative range")
    return files * area / 1e6 * C["density_90"].value / C[f"density_{node}"].value


def raw_die_cost(area, node=7):
    if area == 0:
        return 0.0
    if area < 0:
        raise ValueError("Die area must be nonnegative")
    diameter = C["wafer_diameter"].value
    dies = math.pi * (diameter / 2)**2 / area - math.pi * diameter / math.sqrt(2 * area)
    if dies <= 0:
        raise ValueError("Logic die too large for the gross dies-per-wafer model; specify die_count")
    return C[f"wafer_usd_{node}"].value / dies


def manufacturing(area, memory_usd, node=7, dies=1):
    """Chiplet Actuary organic-substrate RE scenario; excludes NRE and CoWoS.

    Equal-area known-good logic dies, purchased memory already yielded; package
    chip count covers logic dies only (memory packaging is in its purchase price).
    """
    if type(dies) is not int or dies < 1:
        raise ValueError("die_count must be a positive integer")
    per_die = area / dies
    raw = dies * raw_die_cost(per_die, node)
    yield_die = (1 + per_die * C[f"defects_{node}"].value / C["critical_levels"].value) ** -C["critical_levels"].value
    defects = raw * (1 / yield_die - 1)
    bumps = area * C["bump_usd_mm2"].value
    package_area = area * C["package_area_scale"].value
    # Chiplet Actuary package.py organic-substrate RE layer-count rule.
    factor = 1 if dies == 1 else 2 if package_area > 900 else 1.75 if package_area > 289 else 1.5
    substrate = package_area * C["substrate_usd_mm2"].value * factor
    package_yield = C["package_bond_yield"].value ** dies
    package_defects = substrate * (1 / package_yield - 1)
    wasted_kgd = (raw + defects + bumps + memory_usd) * (1 / package_yield - 1)
    parts = {"raw_die_usd":raw, "die_defects_usd":defects, "memory_usd":memory_usd,
             "bumps_usd":bumps, "substrate_usd":substrate, "package_defects_usd":package_defects,
             "wasted_kgd_usd":wasted_kgd}
    return {**parts, "total_usd":sum(parts.values()), "die_yield":yield_die, "package_yield":package_yield,
            "scenario":"organic substrate, equal-area logic dies; not a GA100 CoWoS quote"}
