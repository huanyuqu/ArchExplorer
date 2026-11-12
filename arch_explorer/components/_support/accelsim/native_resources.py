"""Unique, installed native factory resources, grouped by hardware ownership.

Builds may flatten these files into their private native include tree. The
package itself retains one authority per basename; discovery never builds or
loads a native image.
"""
from pathlib import Path


_COMPONENTS = Path(__file__).resolve().parents[2]
_GROUPS = (
    ('sm/implementations/accelsim/native',
     ('shader.cc', 'shader_api.hpp', 'shader_services.hpp')),
    ('l2/implementations/accelsim/native', ('l2.cc',)),
    ('dram/implementations/accelsim/native', ('dram.cc',)),
    ('icnt/implementations/accelsim/native', ('icnt.cc', 'icnt_api.hpp')),
    ('assemblies/gpu/native',
     ('gpu_assembly.cc', 'gpu_assembly.hpp', 'gpu_entry.cc',
      'gpu_transport.cc', 'gpu_transport.hpp')),
    ('_support/accelsim/native',
     ('configuration_api.hpp', 'execution.cc', 'execution.hpp',
      'instruction_config.hpp', 'memory_api.hpp', 'memory_services.hpp',
      'memory_statistics.cc', 'memory_links.cc', 'memory_links.hpp',
      'memory_system.cc', 'memory_system.hpp', 'partition_random.hpp',
      'services.hpp', 'transactions.cc', 'transactions.hpp')),
)
_SUFFIXES = {'.cc', '.cpp', '.hpp', '.h'}


def native_files() -> dict[str, Path]:
    """Return the exact flattened resource set; fail on an incomplete install."""
    result = {}
    for relative, names in _GROUPS:
        directory = _COMPONENTS / relative
        if not directory.is_dir():
            raise ValueError('native resource directory missing: ' + relative)
        actual = {path.name for path in directory.iterdir()
                  if path.suffix in _SUFFIXES}
        if actual != set(names):
            raise ValueError('native resource member set differs: ' + relative)
        for name in names:
            path = directory / name
            if name in result:
                raise ValueError('duplicate native resource basename: ' + name)
            if not path.is_file() or path.is_symlink():
                raise ValueError('native resource must be a regular package file: ' + str(path))
            result[name] = path
    return dict(sorted(result.items()))


def native_include_directories() -> tuple[Path, ...]:
    """Directories supplying the same public, flat factory header names."""
    return tuple(dict.fromkeys(path.parent for path in native_files().values()))
