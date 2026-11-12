"""Generic scheduling; importing public types never loads concrete components."""
from importlib import import_module

_EXPORTS = {
    **{name: '.engine' for name in ('Resource', 'Task', 'TaskGraph', 'Trace', 'simulate')},
    **{name: '.builder' for name in ('Simulator', 'build_simulator')},
    **{name: '.backend' for name in ('CapacityError', 'evaluate_events', 'label_latency', 'run_mapping')},
}
__all__ = list(_EXPORTS)


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(name)
    value = getattr(import_module(_EXPORTS[name], __name__), name)
    globals()[name] = value
    return value
