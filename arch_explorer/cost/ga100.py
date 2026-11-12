"""Sourced physical GA100 cost options, separate from the executing A100 SKU.

The whitepaper does not state full-die L2 capacity. The default retains its
published enabled 40 MiB as a known partial contribution, explicitly flagged.
Pass an independently sourced ``l2_capacity`` to compare another configuration
(e.g. LLMCompass's 48 MiB); do not attribute that number to NVIDIA's whitepaper.
"""
from arch_explorer.adf import Param

WHITEPAPER = "https://images.nvidia.com/aem-dam/en-zz/Solutions/data-center/nvidia-ampere-architecture-whitepaper.pdf"

# PDF page numbers match the printed Arabic pages in this V1.0 file.
GA100_PHYSICAL_FACTS = {
    "sm_count": Param(128, "count", WHITEPAPER + " p.19; Fig.6 p.20"),
    "hbm_stack_interfaces": Param(6, "count", WHITEPAPER + " p.19: six HBM2 stacks; not purchased 80GB memory capacity"),
    "memory_controllers": Param(12, "count", WHITEPAPER + " p.19"),
    "controller_width": Param(512, "bit", WHITEPAPER + " p.19"),
    "nvlink_links": Param(12, "count", WHITEPAPER + " p.52; Fig.6 p.20"),
    "pcie4_lanes": Param(16, "count", WHITEPAPER + " p.53: x16 PCIe Gen4 host interface; Fig.6 p.20"),
    "process_nm": Param(7, "nm", WHITEPAPER + " p.19: TSMC N7"),
    "local_sram_bytes": Param(192 * 1024, "B", WHITEPAPER + " pp.21,33: 192 KB/SM, binary capacity convention"),
}
GA100_UNKNOWNS = (
    "Full physical L2 capacity is not specified in the NVIDIA V1.0 whitepaper; p.35 gives enabled A100 40 MB only.",
    "RF port count, spare SRAM/ECC layout and detailed controller/gating/physical wiring areas are not specified.",
    "Separate SFU, instruction-cache, GPC/TPC/GigaThread/MIG, ROP, NVDEC/NVJPG and optical-flow areas are not characterized.",
    "Memory purchase remains the executing 80 GiB SKU; six physical stack interfaces do not imply six purchased 16 GiB stacks.",
)


def ga100_full_die_options(root="a100", *, l2_capacity=None):
    """Return JSON-serializable ``cost_model`` options for one A100 preset.

    Uses known full-die SM/interfaces, with no changes to active performance.
    The default's full-L2 contribution remains incomplete; supplied L2 evidence
    must be a sourced Param (or its dict). Network area stays a per-SM proxy.
    """
    facts = GA100_PHYSICAL_FACTS
    l2 = l2_capacity if isinstance(l2_capacity, Param) else Param.from_dict(l2_capacity) if l2_capacity else None
    notes = list(GA100_UNKNOWNS)
    if l2 is None:
        l2 = Param(40 * 2**20, "B", WHITEPAPER + " p.35: enabled A100 L2 only; full physical capacity UNKNOWN; partial contribution")
    else:
        if not l2.source.strip() or l2.unit != "B" or type(l2.value) is not int or l2.value < 40 * 2**20:
            raise ValueError("l2_capacity needs sourced integer bytes >= enabled 40 MiB")
        notes[0] += " This scenario uses the separately sourced l2_capacity override, not a whitepaper measurement."
    return {
        "basis":"raw", "process_nm":facts["process_nm"].to_dict(), "scope_notes":notes,
        "overrides":{
            f"{root}/sm":{"physical_instances":facts["sm_count"].to_dict(),
                          "local_sram_bytes":facts["local_sram_bytes"].to_dict()},
            f"{root}/l2":{"capacity":l2.to_dict()},
            f"{root}/dram":{"interface_count":Param(6, "count", WHITEPAPER +
                " p.19: 12 x 512-bit controllers = six 1024-bit model interface units; six HBM stacks").to_dict()},
            f"{root}/l2_fabric":{
                "endpoint_count":Param(128, "count", WHITEPAPER +
                    " p.19 full SM count; model uses one communication proxy per SM, not measured routing endpoints").to_dict(),
                "nvlink_links":facts["nvlink_links"].to_dict(),
                "pcie4_lanes":facts["pcie4_lanes"].to_dict(),
            },
        },
    }
