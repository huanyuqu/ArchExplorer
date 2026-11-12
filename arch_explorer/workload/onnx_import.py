"""Optional, bounded ONNX 1.16.2/opset-17 frontend with portable source bytes.

The source package contains the original ModelProto and complete external data
files. Rebuilding validates their digests and reparses them with the official
parser/checker; it never reopens the original paths or fetches remote data.
"""
import base64
import copy
import hashlib
import math
from pathlib import Path, PurePosixPath

from .graph import finite_json
from .resource_budget import budget_scope,limit,normalize_budget,check_shape

SOURCE_SCHEMA = 'arch-explorer.onnx-source.v1'
MAX_BYTES = 16 * 1024 * 1024
MAX_NODES = 256
MAX_ELEMENTS = 1_000_000


def _onnx():
    try:
        import onnx
    except ImportError as error:
        raise ImportError('ONNX import requires the optional onnx extra: onnx==1.16.2 (with protobuf>=3.20.2)') from error
    if onnx.__version__ != '1.16.2':
        raise ValueError('ONNX frontend requires onnx==1.16.2; no implicit version conversion')
    return onnx


def _pack(data):
    if len(data) > limit('max_source_bytes',MAX_BYTES):
        raise ValueError('ONNX source byte bound exceeded')
    return dict(sha256=hashlib.sha256(data).hexdigest(), base64=base64.b64encode(data).decode('ascii'))


def _unpack(value):
    if type(value) is not dict or set(value) != {'sha256', 'base64'}:
        raise ValueError('ONNX source requires exactly sha256/base64')
    if type(value['base64']) is not str or len(value['base64']) > (limit('max_source_bytes',MAX_BYTES) + 2) // 3 * 4:
        raise ValueError('ONNX encoded source bound exceeded')
    try:
        data = base64.b64decode(value['base64'], validate=True)
    except (ValueError, TypeError) as error:
        raise ValueError('ONNX invalid source base64') from error
    if _pack(data) != value:
        raise ValueError('ONNX source digest/encoding mismatch')
    return data


def _location(value):
    if (type(value) is not str or not value or '\\' in value or ':' in value or '\x00' in value
            or PurePosixPath(value).is_absolute() or any(p in ('', '.', '..') for p in value.split('/'))):
        raise ValueError('ONNX external location must be a normalized relative path without escape')
    return value


def _tensors(model):
    yield from model.graph.initializer
    for node in model.graph.node:
        for attribute in node.attribute:
            if attribute.type == attribute.TENSOR:
                yield attribute.t


def _external(tensor):
    onnx = _onnx()
    if tensor.data_location != onnx.TensorProto.EXTERNAL:
        if tensor.external_data:
            raise ValueError('ONNX external metadata requires EXTERNAL data_location')
        return None
    fields = {item.key: item.value for item in tensor.external_data}
    if len(fields) != len(tensor.external_data) or set(fields) - {'location', 'offset', 'length', 'checksum'} or 'location' not in fields:
        raise ValueError('ONNX unsupported/duplicate external tensor metadata')
    _location(fields['location'])
    return fields


def _parse(data):
    onnx = _onnx()
    try:
        return onnx.load_model_from_string(data)
    except Exception as error:
        raise ValueError('ONNX ModelProto parse failed: ' + str(error)) from error


def _snapshot(path,external_root):
    """Return WorkloadSpec('onnx', ...), containing a portable, validated snapshot.

    inputs uses original graph.input names. Initializer/input overlaps are
    rejected. input_shapes maps root names to positive concrete logical shapes;
    rank-zero FP32 values use JSON numbers and one physical storage element.
    External locations are relative to the model directory; external_root is
    the explicitly supplied containment boundary for those local files.
    """
    p = Path(path)
    try:
        if p.stat().st_size > limit('max_source_bytes',MAX_BYTES):
            raise ValueError('ONNX model byte bound exceeded')
        data = p.read_bytes()
    except OSError as error:
        raise ValueError('ONNX source model unavailable: ' + p.name) from error
    model = _parse(data)
    external = {}
    for tensor in _tensors(model):
        fields = _external(tensor)
        if fields is None:
            continue
        if external_root is None:
            raise ValueError('ONNX external tensors require explicit external_root')
        try:
            root = Path(external_root).resolve(strict=True)
            target = (p.resolve().parent / fields['location']).resolve(strict=True)
        except OSError as error:
            raise ValueError('ONNX external data unavailable: ' + fields['location']) from error
        if not root.is_dir() or not target.is_relative_to(root) or not target.is_file():
            raise ValueError('ONNX external tensor escaped allowed root or is not a file')
        if target.stat().st_size > limit('max_source_bytes',MAX_BYTES):
            raise ValueError('ONNX external byte bound exceeded')
        if fields['location'] not in external:
            if len(data)+sum(len(_unpack(v)) for v in external.values())+target.stat().st_size>limit('max_source_bytes',MAX_BYTES):
                raise ValueError('ONNX total source byte bound exceeded')
            external[fields['location']] = _pack(target.read_bytes())
    if len(data) + sum(len(_unpack(v)) for v in external.values()) > limit('max_source_bytes',MAX_BYTES):
        raise ValueError('ONNX total source byte bound exceeded')
    return dict(schema=SOURCE_SCHEMA,model=_pack(data),external=external)


def _preflight(source,input_shapes,policy):
    from .onnx_resources import preflight_model
    sizes=dict(model_bytes=len(_unpack(source['model'])),external_bytes={k:len(_unpack(v)) for k,v in source['external'].items()})
    sizes['total_bytes']=sizes['model_bytes']+sum(sizes['external_bytes'].values())
    return preflight_model(_hydrate(source),input_shapes,policy,sizes)


def onnx_preflight(path,*,input_shapes=None,external_root=None,resource_budget=None):
    """Typed source/shape/resource admission without allocating numeric root arrays."""
    with budget_scope(resource_budget):
        return _preflight(_snapshot(path,external_root),{} if input_shapes is None else input_shapes,resource_budget)


def import_onnx(path, *, inputs, input_shapes=None, external_root=None, name='onnx',resource_budget=None):
    from arch_explorer.adf.evaluation import WorkloadSpec
    with budget_scope(resource_budget):
        source=_snapshot(path,external_root);shapes={} if input_shapes is None else input_shapes
        admitted=_preflight(source,shapes,resource_budget)
        if not admitted['admitted']:raise ValueError('ONNX resource budget exceeded: '+str(admitted['violations']))
        params=dict(name=name,source=source,inputs=inputs,input_shapes=shapes)
        if resource_budget is not None:params['resource_budget']=normalize_budget(resource_budget)
        params=finite_json(params);build_onnx(params)
        return WorkloadSpec('onnx',params)


def _hydrate(source):
    onnx = _onnx()
    if type(source) is not dict or set(source) != {'schema', 'model', 'external'} or source['schema'] != SOURCE_SCHEMA:
        raise ValueError('ONNX source package schema/fields differ')
    if type(source['external']) is not dict:
        raise ValueError('ONNX external source package must be an object')
    data = _unpack(source['model'])
    files = {_location(n): _unpack(v) for n, v in source['external'].items()}
    if len(data) + sum(map(len, files.values())) > limit('max_source_bytes',MAX_BYTES):
        raise ValueError('ONNX total source byte bound exceeded')
    model = _parse(data)
    if (model.ir_version != 8 or len(model.opset_import) != 1 or model.opset_import[0].domain != ''
            or model.opset_import[0].version != 17):
        raise ValueError('ONNX supports exactly IR 8, standard empty domain, opset 17')
    if model.functions or model.training_info or model.graph.sparse_initializer:
        raise ValueError('ONNX functions/training/sparse initializers are unsupported')
    if not 1 <= len(model.graph.node) <= limit('max_nodes',MAX_NODES):
        raise ValueError('ONNX requires 1..256 inference DAG nodes')
    used = set()
    for tensor in _tensors(model):
        if tensor.data_type not in (onnx.TensorProto.FLOAT, onnx.TensorProto.FLOAT16,onnx.TensorProto.BFLOAT16,onnx.TensorProto.INT32,onnx.TensorProto.INT64,onnx.TensorProto.BOOL):
            raise ValueError('ONNX tensor dtype must be FP32/FP16/BF16/int32/int64/bool')
        empty_shape = tensor.data_type == onnx.TensorProto.INT64 and list(tensor.dims) == [0]
        if not empty_shape:check_shape(list(tensor.dims))
        fields = _external(tensor)
        if fields is None:
            continue
        location = fields['location']
        if location not in files:
            raise ValueError('ONNX external data missing from source package: ' + location)
        raw = files[location]
        if 'checksum' in fields and fields['checksum'] != hashlib.sha1(raw).hexdigest():
            raise ValueError('ONNX external SHA1 checksum differs')
        from .onnx_resources import WIDTHS
        expected = math.prod(tensor.dims) * WIDTHS[tensor.data_type]
        def decimal(key, default):
            value = fields.get(key)
            if value is None:
                return default
            if not value.isascii() or not value.isdigit():
                raise ValueError('ONNX external ' + key + ' must be a nonnegative decimal integer')
            return int(value)
        offset = decimal('offset', 0)
        length = decimal('length', len(raw) - offset)
        if offset < 0 or length != expected or offset + length > len(raw):
            raise ValueError('ONNX external offset/length/dtype/shape byte range differs')
        if tensor.raw_data or tensor.float_data or tensor.int64_data or tensor.int32_data:
            raise ValueError('ONNX external tensor also contains inline data')
        tensor.raw_data = raw[offset:offset + length]
        tensor.ClearField('external_data')
        tensor.data_location = onnx.TensorProto.DEFAULT
        used.add(location)
    if used != set(files):
        raise ValueError('ONNX source package contains unreferenced external data')
    return model


def _dependencies():
    """Deterministic parser/checker Python and binary distribution identities."""
    onnx = _onnx()
    import google.protobuf
    from importlib.metadata import distribution
    rows = {}
    for package in ('onnx', 'protobuf'):
        dist = distribution(package)
        for entry in sorted(dist.files or (), key=str):
            if entry.suffix in ('.py', '.so', '.pyd'):
                p = Path(dist.locate_file(entry))
                rows[package + ':' + entry.as_posix()] = hashlib.sha256(p.read_bytes()).hexdigest()
    return dict(onnx=onnx.__version__, protobuf=google.protobuf.__version__, modules=rows)


def build_onnx(params, *, tp=1):
    if type(params) is not dict:raise ValueError('ONNX workload requires exactly name/source/inputs/input_shapes and optional resource_budget')
    with budget_scope(params.get('resource_budget')):
        return _build_onnx(params,tp=tp)


def _build_onnx(params, *, tp=1):
    from .extensions import _record_source
    from .onnx_lowering import lower_model
    if tp != 1:
        raise ValueError('ONNX frontend supports tp=1 only')
    if set(params)-{'name', 'source', 'inputs', 'input_shapes','resource_budget'} or not {'name','source','inputs','input_shapes'}<=set(params):
        raise ValueError('ONNX workload requires exactly name/source/inputs/input_shapes')
    p = finite_json(params)
    _record_source(_hydrate)
    _record_source(lower_model)
    model = _hydrate(p['source'])
    graph = lower_model(model, p['inputs'], p['input_shapes'], p['name'])
    if 'resource_budget' in p:graph.meta['resource_budget']=normalize_budget(p['resource_budget'])
    graph.meta['onnx']['source'] = copy.deepcopy(p['source'])
    graph.meta['onnx']['parser_dependencies'] = _dependencies()
    return graph


def describe_onnx(spec):
    """Return finite-JSON node/value/root/output correspondence, without answers."""
    from .api import workload_from_spec
    if spec.kind != 'onnx':
        raise ValueError('describe_onnx requires an onnx WorkloadSpec')
    return copy.deepcopy(workload_from_spec(spec).meta['onnx'])
