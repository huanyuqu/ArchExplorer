"""Mapping templates (section III-C) and mapping selection tools.

Importing this package registers ``gsm_tiled``, ``dmc_partition``,
``tensor_parallel`` and ``pim_offload`` in ``arch_explorer.adf.MAPPING_TEMPLATES``.
"""
from arch_explorer.mapping.context import MappingContext
from arch_explorer.mapping.cgra import CgraSpatial
from arch_explorer.mapping.dmc import DmcPartition
from arch_explorer.mapping.gsm import GsmTiled
from arch_explorer.mapping.pim import PimOffload
from arch_explorer.mapping.search import MappingSearchResult, best_mapping, mapping_space
from arch_explorer.mapping.tp import TensorParallel
from arch_explorer.mapping.manycore import IpuBsp, TianjicSpatial
from arch_explorer.mapping.multi_device import MultiDevice
from arch_explorer.mapping.execution_graph import ExecutionGraph
from arch_explorer.mapping.roofline_graph import RooflineGraph
from arch_explorer.mapping.compiled import AdaptedEvents
from arch_explorer.mapping.routed import AdaptedRoutedEvents
from arch_explorer.mapping.compilation import CompileGraph

__all__ = ["CgraSpatial", "DmcPartition", "GsmTiled", "MappingContext", "MappingSearchResult", "PimOffload", "TensorParallel",
           "best_mapping", "mapping_space", "ExecutionGraph"]
