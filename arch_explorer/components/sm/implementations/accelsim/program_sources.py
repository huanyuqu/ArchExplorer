"""Strict program source closure for actual workload/target callables."""
from arch_explorer.components._support.paths import package_root
from functools import partial
import inspect
from pathlib import Path
from arch_explorer.components.accelsim.lowering_limits import sha

def sources(subjects=()):
    root=package_root()
    files={p.relative_to(root).as_posix():p for folder in ('workload','components/accelsim','components/sm/implementations/accelsim',
           'components/assemblies/gpu','components/_support/accelsim','components/l2','components/dram/implementations/accelsim',
           'components/icnt','components/l1_shared')
           for p in sorted((root/folder).rglob('*.py'))}
    for name in ('components/implementations.py', 'components/native_bindings.py', 'components/sm/definition.py', 'adf/model.py'):
        files[name]=root/name
    for name in ('mapping/accelsim_program.py','simulator/api.py'):
        files[name]=root/name
    for subject in subjects:
        while isinstance(subject,partial):
            subject=subject.func
        if not (inspect.isclass(subject) or inspect.isroutine(subject)):
            subject=type(subject)
        path=inspect.getsourcefile(subject)
        if not path:
            raise ValueError('program semantic/target source unavailable')
        path=Path(path).resolve()
        if path.is_relative_to(root):
            continue
        package=path.parent
        if (package/'__init__.py').is_file():
            while (package.parent/'__init__.py').is_file():
                package=package.parent
            for item in sorted(package.rglob('*.py')):
                key='extension:'+package.name+'/'+item.relative_to(package).as_posix()
                if key in files and files[key]!=item:
                    raise ValueError('ambiguous program extension source '+key)
                files[key]=item
        else:
            key='extension:'+subject.__module__+'/'+path.name
            if key in files and files[key]!=path:
                raise ValueError('ambiguous program extension source '+key)
            files[key]=path
    return {name:sha(path.read_bytes()) for name,path in sorted(files.items())}

