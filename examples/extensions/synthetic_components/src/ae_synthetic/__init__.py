"""Installable extension; importing it performs no discovery or execution."""


def register():
    """Explicit registration and the installed entry point share one function."""
    from arch_explorer.components.api import register_component, register_provider
    from arch_explorer.adf.registry import register_template
    from .component import TutorialMac
    from .provider import PROVIDER, TutorialSerial
    register_component(TutorialMac)
    register_template(TutorialSerial)
    register_provider(PROVIDER)
