"""Exact, finite read-only source-era recognition; producer facts stay intact."""
import json
import hashlib
from pathlib import Path

_SOURCE_TABLE_SHA256 = 'ae03853cf0f74cc3a35dd2fe0658365baee663fcddf5f060e8952463e480ecf4'  # set from the exact Git-baseline capture below
_OLD_LOWER = {'line':48, 'module':'arch_explorer.components.accelsim.builtin_emitters',
              'qualname':'emit_builtin', 'sha256':'4f530ef87f738c5b194cb46ecd8794cde2cc2f0989a3d001bd4976fe97b6cef1'}
_NEW_LOWER = {'line':48, 'module':'arch_explorer.components.sm.implementations.accelsim.builtin_emitters',
              'qualname':'emit_builtin', 'sha256':'a12142423ac16ee1d632ed28433424f250fc709c611a4f354f39c751eaa6ff74'}
_CALLABLE_LINES = {'supported': (16, 16), 'install': (28, 28), '_broadcast': (40, 40), 'emit_builtin': (48, 48)}


def frozen():
    raw=Path(__file__).with_name('layout_v40.json').read_bytes()
    if hashlib.sha256(raw).hexdigest()!=_SOURCE_TABLE_SHA256:
        raise ValueError('frozen layout source table identity differs')
    return json.loads(raw)


def callable_view(value):
    """Finite comparison-only relocation of one byte-bound unchanged lowering."""
    if isinstance(value,dict):
        location=_CALLABLE_LINES.get(value['qualname']) if set(value)==set(_OLD_LOWER) and isinstance(value.get('qualname'),str) else None
        if (set(value)==set(_OLD_LOWER) and value['module']==_OLD_LOWER['module']
                and value['sha256']==_OLD_LOWER['sha256'] and location is not None and value['line']==location[0]):
            from importlib import import_module
            module=import_module(_NEW_LOWER['module'])
            if hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()!=_NEW_LOWER['sha256']:
                raise ValueError('relocated lowering changed beyond the frozen move')
            return dict(value, module=_NEW_LOWER['module'], sha256=_NEW_LOWER['sha256'], line=location[1])
        return {k:callable_view(v) for k,v in value.items()}
    if isinstance(value,list):return [callable_view(v) for v in value]
    return value


def source_era(producer, consumer, kind, *, read_only=False):
    if not isinstance(producer, dict) or not isinstance(consumer, dict):
        raise ValueError('complete component source vectors required')
    if producer == consumer:
        return 'current'
    if not read_only:
        raise ValueError('new execution requires the current component source vector')
    expected = frozen()[kind]
    producer_core = {k:v for k,v in producer.items() if not k.startswith('extension:')}
    extension = {k:v for k,v in producer.items() if k.startswith('extension:')}
    if producer_core != expected or extension != {k:v for k,v in consumer.items() if k.startswith('extension:')}:
        raise ValueError('no exact registered pre-layout source era')
    # Workload semantic implementations did not change in this responsibility move.
    if any(consumer.get(k) != value for k,value in expected.items() if k.startswith(('workload/','model_specs/'))):
        raise ValueError('historical semantic source identity differs')
    return 'sim40-component-layout-v1-read-only'
