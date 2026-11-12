"""External registrations only; no simulation or device probing at import."""


def register():
    from arch_explorer.components.api import register_component
    from arch_explorer.adf.registry import register_template
    from arch_explorer.workload.api import register_operator, register_workload
    from .semantics import AFFINE
    from .graphs import build_workload
    from .hardware import TutorialArray
    from .mapping import RowTiles
    register_operator(AFFINE)
    register_workload("tutorial_branches", build_workload)
    register_component(TutorialArray)
    register_template(RowTiles)
