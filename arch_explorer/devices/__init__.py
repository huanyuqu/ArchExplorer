"""Device presets: ADFs whose every parameter carries a public source.

``get(name, **options)`` returns a fresh non-GPU ADF (``neupims``, ``hsndi``,
CGRA and manycore presets). ``bonded_dram`` builds a
hybrid-bonded DRAM shard for ``bond_memory``. Importing this package also
registers the component library, the mapping templates and the ``events``
backend, so ``arch_explorer.adf.evaluate`` works on the presets directly.
"""
import arch_explorer.simulator.events  # noqa: F401  (components, templates, events backend)
from arch_explorer.devices.pim import PRESETS as PIM_PRESETS, bonded_dram, hsndi, neupims
from arch_explorer.devices.cgra import PRESETS as CGRA_PRESETS, vectorcgra_4x4, vectorcgra_4x4_diag
from arch_explorer.devices.manycore import PRESETS as MANYCORE_PRESETS, ipu_gc200, tianjicx

PRESETS = {**PIM_PRESETS, **CGRA_PRESETS, **MANYCORE_PRESETS}


def get(name, **options):
    """A fresh preset ADF; ``options`` select documented alternative parameter values (see the preset)."""
    if name not in PRESETS:
        raise KeyError(f"Unknown device preset {name!r}; available: {sorted(PRESETS)}")
    return PRESETS[name](**options)


def names():
    return sorted(PRESETS)


__all__ = ["PRESETS", "bonded_dram", "get", "hsndi", "names", "neupims",
           "vectorcgra_4x4", "vectorcgra_4x4_diag", "ipu_gc200", "tianjicx"]
