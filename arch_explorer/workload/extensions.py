"""Workload-owned semantic registration; installed discovery stays lazy.

Registrations are identities, not replacement hooks. A registrar may repeat
the same object, but cannot override an existing builtin or another package.
The component discovery transaction also snapshots these dictionaries.
"""
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
import inspect
import hashlib
from pathlib import Path
import sys


@dataclass(frozen=True)
class OperatorDefinition:
    name: str
    validate: object
    lower: object = None
    version: str = '1'
    reference: object = None


_OPERATORS = {}
_OPERATOR_STATES = {}
_WORKLOADS = {}
_SOURCE_TRACE = ContextVar("workload_source_trace", default=None)


def _name(value):
    if not isinstance(value, str) or not value or value.strip() != value:
        raise ValueError("workload/operator name must be a nonempty, unpadded string")
    return value


def _signature(function, *args, **kwargs):
    if not callable(function):
        raise TypeError("workload semantic hook must be callable")
    inspect.signature(function).bind(*args, **kwargs)


def register_operator(definition):
    """Register semantics and an optional legacy events lower; never replace."""
    if not isinstance(definition, OperatorDefinition):
        raise TypeError("register_operator requires an OperatorDefinition")
    name = _name(definition.name)
    _name(definition.version)
    _signature(definition.validate, None, {})
    if definition.lower is not None:
        _signature(definition.lower, None, "fp32", rows=None, tile=None)
    if definition.reference is not None:
        _signature(definition.reference, None, [])
    if name in _OPERATORS and _OPERATORS[name] is not definition:
        raise ValueError(f"operator {name!r} already registered")
    if name in _OPERATORS and _OPERATOR_STATES[name] != _operator_state(definition):
        raise ValueError(f"operator {name!r} registered definition was mutated")
    _OPERATORS[name] = definition
    _OPERATOR_STATES[name] = _operator_state(definition)
    return definition


def register_workload(kind, builder):
    """Register builder(params, *, tp=1); tp must be an explicit parameter."""
    kind = _name(kind)
    _signature(builder, {}, tp=1)
    param = inspect.signature(builder).parameters.get("tp")
    if param is None or param.kind not in (param.KEYWORD_ONLY, param.POSITIONAL_OR_KEYWORD):
        raise TypeError("workload builder must explicitly accept tp as a keyword")
    if kind in _WORKLOADS and _WORKLOADS[kind] is not builder:
        raise ValueError(f"workload kind {kind!r} already registered")
    _WORKLOADS[kind] = builder
    return builder


def _discover():
    from arch_explorer.components.extensions import discover_components
    discover_components()


def get_operator(name):
    _discover()
    if name not in _OPERATORS:
        raise ValueError(f"unknown operator {name!r}; registered: {sorted(_OPERATORS)}")
    definition = _OPERATORS[name]
    if _OPERATOR_STATES.get(name) != _operator_state(definition):
        raise ValueError(f"operator {name!r} registered definition was mutated")
    return definition


def _operator_state(definition):
    return (definition.name, definition.version, id(definition.validate),
            id(definition.lower), id(definition.reference))


def callable_identity(function):
    """Content identity for an installed Python hook, without calling it.

    The module and qualified name distinguish hooks sharing a file; the file
    digest follows the installed source, independent of its absolute location.
    Evidence consumers also receive actual callable objects via source_objects.
    """
    from functools import partial
    from .graph import finite_json
    if isinstance(function, partial):
        return dict(partial=callable_identity(function.func),
                    args=finite_json(function.args, tuples=True),
                    keywords=finite_json(function.keywords or {}, tuples=True))
    subject = function if inspect.isroutine(function) or inspect.isclass(function) else type(function)
    used = _SOURCE_TRACE.get()
    if used is not None:
        return used.identity(subject)
    return _callable_identity(subject)


def _callable_identity(subject):
    try:
        path = inspect.getsourcefile(subject)
    except TypeError:
        path = None
    if not path or not Path(path).is_file():
        raise ValueError(f"implementation source unavailable: {subject!r}")
    code = getattr(subject, '__code__', None)
    return {'module': subject.__module__, 'qualname': subject.__qualname__,
            'line': None if code is None else code.co_firstlineno,
            'sha256': hashlib.sha256(Path(path).read_bytes()).hexdigest()}


def operator_identity(operator):
    """Stable semantics/reference identity; no builder or semantic execution."""
    from .operators import Operator
    from .full_primitives import FullOperator, reference
    from .builtins import validate_full
    if type(operator) is FullOperator:
        return dict(kind='semantic', operator=operator.op, type='full', version='1',
                    validate=callable_identity(validate_full),
                    lower=callable_identity(FullOperator.kernel),
                    reference=callable_identity(reference))
    if type(operator) is not Operator:
        raise TypeError('operator_identity requires Operator or FullOperator')
    definition = get_operator(operator.op)
    return dict(kind='semantic', operator=definition.name, type='operator', version=definition.version,
                validate=callable_identity(definition.validate),
                lower=None if definition.lower is None else callable_identity(definition.lower),
                reference=None if definition.reference is None else callable_identity(definition.reference))


def has_workload(kind):
    _discover()
    return kind in _WORKLOADS


def _record_source(function):
    used = _SOURCE_TRACE.get()
    if function is not None and used is not None and not any(item is function for item in used):
        used.append(function)
        used.remember(function)


class SourceTrace(list):
    """Actual callable list with separately serializable stable identity facts."""
    def __init__(self):
        super().__init__()
        self._identities = []
        self._callables = {}
        self._files = {}
        self._packages = {}

    @staticmethod
    def _state(subject):
        module=getattr(subject,'__module__',None)
        code=getattr(subject,'__code__',None)
        return (module,getattr(subject,'__qualname__',None),id(code),code,
                getattr(sys.modules.get(module),'__file__',None))

    def _capture(self,path):
        if path in self._files:
            return
        package=path.parent
        if (package/'__init__.py').is_file():
            while (package.parent/'__init__.py').is_file():
                package=package.parent
            paths=tuple(sorted(package.rglob('*.py')))
            self._packages[package]=paths
        else:
            paths=(path,)
        for member in paths:
            self._files[member]=hashlib.sha256(member.read_bytes()).hexdigest()

    def identity(self,subject):
        key=id(subject);state=self._state(subject)
        if key in self._callables:
            previous,old_state,identity=self._callables[key]
            if previous is not subject or state!=old_state:
                raise ValueError('implementation callable changed during source trace')
            return dict(identity)
        try:
            source=inspect.getsourcefile(subject)
        except TypeError:
            source=None
        if not source or not Path(source).is_file():
            raise ValueError(f'implementation source unavailable: {subject!r}')
        path=Path(source).resolve()
        self._capture(path)
        code=getattr(subject,'__code__',None)
        identity=dict(module=subject.__module__,qualname=subject.__qualname__,
                      line=None if code is None else code.co_firstlineno,sha256=self._files[path])
        self._callables[key]=(subject,state,identity)
        return dict(identity)

    def remember(self,function):
        # Preserve deferred rejection of uninspectable sources. Public identity
        # requests still reject immediately, as before this scoped cache.
        subject=function if inspect.isroutine(function) or inspect.isclass(function) else type(function)
        try:
            self.identity(subject)
        except ValueError as error:
            if not str(error).startswith('implementation source unavailable:'):
                raise

    def verify(self):
        for subject,state,_ in self._callables.values():
            if self._state(subject)!=state:
                raise ValueError('implementation callable changed during source trace')
        for package,paths in self._packages.items():
            if tuple(sorted(package.rglob('*.py')))!=paths:
                raise ValueError('implementation source inventory changed during source trace')
        for path,digest in self._files.items():
            if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest()!=digest:
                raise ValueError('implementation source bytes changed during source trace')

    @property
    def identities(self):
        from .graph import finite_json
        return finite_json(self._identities)


def _record_identity(value):
    used = _SOURCE_TRACE.get()
    if used is not None and value not in used._identities:
        used._identities.append(value)


def _record_operator(operator):
    # Do not require source materialization when no provenance context is active.
    if _SOURCE_TRACE.get() is not None:
        _record_identity(operator_identity(operator))


@contextmanager
def source_trace():
    """Yield actual builder/validator/lower callables used in this context.

    Nested contexts have independent lists. After the inner context exits,
    recording resumes in the outer list; callers may explicitly merge lists.
    No graph is rebuilt and no hook is invoked to collect this evidence.
    """
    used = SourceTrace()
    token = _SOURCE_TRACE.set(used)
    try:
        yield used
    finally:
        try:
            used.verify()
        finally:
            _SOURCE_TRACE.reset(token)


def validate_operator(operator, tensors):
    definition = get_operator(operator.op)
    _record_operator(operator)
    _record_source(definition.validate)
    result = definition.validate(operator, tensors)
    if result is False:
        raise ValueError(f"{operator.name}: registered operator validation rejected the inputs")
    if result is not None and result is not True:
        raise TypeError("operator validate must return None/True or raise on invalid inputs")


def source_objects(workload_spec, ir, implementations=()):
    """Return declared sources without calling a builder/validator/lower."""
    from .full_primitives import FullOperator, reference
    from .builtins import validate_full
    result = []
    builder = _WORKLOADS.get(workload_spec.kind)
    if builder is not None:
        result.append(builder)
    for operator in ir.operators:
        if isinstance(operator, FullOperator):
            objects = (validate_full, FullOperator.kernel, reference)
        else:
            definition = get_operator(operator.op)
            objects = (definition.validate, definition.lower, definition.reference)
        for function in objects:
            if function is not None and not any(existing is function for existing in result):
                result.append(function)
    from .execution import implementation_source_objects
    for implementation in implementations:
        for function in implementation_source_objects(implementation):
            if not any(existing is function for existing in result):
                result.append(function)
    return tuple(result)


# Populate descriptors only. The builders import model implementations when
# called; importing this registry must not start discovery or model execution.
from .builtins import install as _install
_install()
