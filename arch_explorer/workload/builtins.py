"""Authoritative builtin operator contracts and workload constructors.

Matrix tensors may be shaped arbitrarily when their element count matches
the declared operation. Legacy packed attention buffers explicitly declare
input regions (element counts); no implicit oversized operand is accepted.
"""
import math

from arch_explorer.components.base import Kernel, dtype_bytes


def _positive(value, field):
    if type(value) is not int or value < 1:
        raise ValueError(f"{field} must be a positive integer")
    return value


def _count(tensor):
    return math.prod(tensor.shape)


def validate_basic(op, tensors):
    if op.op in ('add', 'scale', 'softmax') and 'shape' in op.attrs:
        from .tensor_ops import validate
        from .extensions import _record_source
        _record_source(validate)
        validate(op, tensors)
        return
    from .operators import MATRIX_OPS, VECTOR_COSTS
    a = op.attrs
    keys = ({"batch", "m", "k", "n", "shared_b", "kv_group", "dtype", "output_dtype", "input_elements"}
            if op.op == "matmul" else {"batch", "m", "k", "dtype"} if op.op == "mvm"
            else {"rows", "cols", "dtype"} if op.op in VECTOR_COSTS else {"bytes"})
    if set(a) - keys:
        raise ValueError(f"{op.name}: unknown {op.op} attributes {sorted(set(a)-keys)}")
    if len(op.outputs) != 1:
        raise ValueError(f"{op.name}: {op.op} requires one output")
    inputs = [tensors[n] for n in op.inputs]
    output = tensors[op.outputs[0]]
    dtype = a.get("dtype", inputs[0].dtype if inputs else output.dtype)
    dtype_bytes(dtype)
    out_dtype = a.get("output_dtype", dtype)
    dtype_bytes(out_dtype)
    if output.dtype != out_dtype or any(t.dtype != dtype for t in inputs):
        raise ValueError(f"{op.name}: operand/result dtype differs from declared precision")
    if op.op in MATRIX_OPS:
        if len(inputs) != 2:
            raise ValueError(f"{op.name}: {op.op} requires two inputs")
        b, m, k = (_positive(a.get(key, 1 if key == 'batch' else None), key)
                   for key in ("batch", "m", "k"))
        if op.op == "matmul":
            n = _positive(a.get("n"), "n")
            if type(a.get("shared_b", False)) is not bool:
                raise TypeError("shared_b must be bool")
            group = _positive(a.get("kv_group", 1), "kv_group")
            if b % group:
                raise ValueError("kv_group must divide batch")
            counts = [b*m*k, (1 if a.get("shared_b", False) else b//group)*k*n]
            out_count = b*m*n
            regions = a.get("input_elements")
            if regions is not None:
                if (not isinstance(regions, (tuple, list)) or len(regions) != 2
                        or any(type(x) is not int for x in regions) or list(regions) != counts):
                    raise ValueError("packed input_elements must exactly describe the consumed matrix operands")
                if any(_count(t) < count for t, count in zip(inputs, counts)):
                    raise ValueError("packed matrix input region exceeds its tensor")
            elif [_count(t) for t in inputs] != counts:
                raise ValueError(f"{op.name}: matrix input shape does not match declared dimensions")
        else:
            out_count = b*m
            if [_count(t) for t in inputs] != [m*k, b*k]:
                raise ValueError(f"{op.name}: MVM input shape does not match declared dimensions")
        if _count(output) != out_count:
            raise ValueError(f"{op.name}: matrix output shape does not match declared dimensions")
    elif op.op in VECTOR_COSTS:
        rows, cols = (_positive(a.get(key), key) for key in ("rows", "cols"))
        size = rows*cols
        expected = ([size, size] if op.op in ("add", "swiglu") else [size])
        if op.op == "swiglu" and len(inputs) == 1:
            expected = [2*size]  # packed gate/up projection
        if [_count(t) for t in inputs] != expected or _count(output) != size:
            raise ValueError(f"{op.name}: {op.op} tensor sizes do not match rows/cols")
    else:
        size = _positive(a.get("bytes"), "bytes")
        if len(inputs) != 1 or inputs[0].bytes != size or output.bytes != size:
            raise ValueError(f"{op.name}: allreduce buffer size differs from bytes")


def lower_basic(op, dtype, rows=None, tile=None):
    from .operators import VECTOR_COSTS
    a = op.attrs
    if op.op == "matmul":
        b, m, k, n = tile or (a.get("batch", 1), a["m"], a["k"], a["n"])
        return Kernel("gemm", (b, m, k, n), dtype)
    if op.op == "mvm":
        return Kernel("mvm", (a.get("batch", 1), rows or a["m"], a["k"]), dtype)
    if op.op in VECTOR_COSTS:
        return Kernel("vector", (rows or a["rows"], a["cols"]), dtype, **VECTOR_COSTS[op.op])
    raise NotImplementedError(f"{op.op} is not a compute kernel")


def validate_full(op, tensors):
    """Keep full primitives distinct from the legacy compact vector IR."""
    from .full_primitives import COSTS
    from .graph import shape
    a = op.attrs
    from .indexing import KINDS as INDEX_KINDS
    if op.op in INDEX_KINDS and 'rows' not in a and 'cols' not in a:
        from .indexing import validate
        from .extensions import _record_source
        _record_source(validate);validate(op,tensors)
        return
    if op.op in ('mul', 'scale', 'exp', 'transpose', 'reshape', 'cast', 'affine_norm', 'gelu_tanh',
                 'patchify', 'concat', 'bias', 'gelu_exact','quick_gelu','l2_norm') and 'rows' not in a and 'cols' not in a:
        from .tensor_ops import validate
        from .extensions import _record_source
        _record_source(validate)
        validate(op, tensors)
        return
    if op.op not in COSTS and op.op != "view":
        raise ValueError(f"unknown full primitive {op.op!r}")
    if len(op.outputs) != 1:
        raise ValueError("full primitive requires one output")
    output = tensors[op.outputs[0]]
    if shape(a.get("shape"), "primitive shape") != output.shape:
        raise ValueError(f"{op.name}: full primitive output shape differs")
    if op.op == "view":
        if len(op.inputs) != 1:
            raise ValueError("view requires one input")
        from .storage import view_tensor
        attrs=dict(shape=a['shape'])
        if a.get('axes') is not None:attrs['axes']=a['axes']
        expected=view_tensor(tensors,op.inputs[0],op.outputs[0],
            'transpose' if 'axes' in attrs else 'reshape',attrs)
        # Legacy contiguous views used None instead of explicit contiguous strides.
        if (output.alias_of,output.dtype,output.shape,output.offset)!=(expected.alias_of,expected.dtype,expected.shape,expected.offset):
            raise ValueError('legacy view storage descriptor differs from transform')
        if (output.strides or tuple(math.prod(output.shape[i+1:]) for i in range(len(output.shape))))!=expected.strides:
            raise ValueError('legacy view strides differ from transform')
        return
    rows, cols = (_positive(a.get(key), key) for key in ("rows", "cols"))
    if op.op == "argmax":
        if len(op.inputs) != 1 or tensors[op.inputs[0]].shape != (rows, cols) or _count(output) != rows:
            raise ValueError("argmax shape differs from its input")
    elif rows*cols != _count(output):
        raise ValueError(f"{op.name}: full primitive rows/cols differ from output")
    for key in ("read_bytes", "write_bytes", "scratch_bytes", "casts_per_element", "extra_row_simt"):
        value = a.get(key, 0)
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            raise ValueError(f"{op.name}.{key} must be finite and nonnegative")
    if a.get("write_bytes") != output.bytes:
        raise ValueError(f"{op.name}: primitive write size differs from output storage")
    if a.get("result_dtype", output.dtype) != output.dtype:
        raise ValueError(f"{op.name}: primitive result dtype differs")


def lower_full(op, dtype, rows=None, tile=None):
    from .full_primitives import FullOperator
    return FullOperator.kernel(op, dtype, rows=rows, tile=tile)


def _operator(params, *, tp=1):
    if tp != 1:
        raise ValueError("Operator workloads are not tensor-parallel; use tp=1")
    from .operators import operator_workload
    return operator_workload(params)


def _transformer(params, *, tp=1):
    from .llm import transformer_workload
    p = dict(params)
    unknown = set(p) - {"model", "phase", "batch", "seq_len", "layers", "dtype", "iterations", "resource_budget"}
    if unknown:
        raise ValueError(f"Unknown transformer parameters: {sorted(unknown)}")
    return transformer_workload(p.pop("model"), tp=tp, **p)


def _vit(params, *, tp=1):
    if tp != 1:
        raise ValueError("Full models support single-device gsm_tiled only")
    from .full_models import full_model_workload
    return full_model_workload("vit", params)


def _clip(params, *, tp=1):
    if tp != 1:
        raise ValueError("Full models support single-device gsm_tiled only")
    from .full_models import full_model_workload
    return full_model_workload("clip", params)


def _pi05(params, *, tp=1):
    if tp != 1:
        raise ValueError("pi05 supports single-device gsm_tiled only")
    from .pi05 import pi05_workload
    return pi05_workload(params)


def _graph(params, *, tp=1):
    if tp != 1:
        raise ValueError("graph workloads currently support tp=1 only")
    from .operators import Workload
    return Workload.from_dict(params)


def install():
    from .extensions import OperatorDefinition, register_operator, register_workload
    from .operators import OPERATORS
    from .full_primitives import COSTS, reference
    for name in OPERATORS:
        register_operator(OperatorDefinition(name, validate_basic, lower_basic, reference=reference))
    for name in sorted((set(COSTS) | {"view"}) - set(OPERATORS)):
        from .indexing import KINDS as INDEX_KINDS, reference as index_reference
        from .tensor_ops import reference as tensor_reference
        register_operator(OperatorDefinition(name, validate_full, lower_full,
            reference=index_reference if name in INDEX_KINDS else tensor_reference if name in ('l2_norm','cast') else reference))
    for kind, builder in (("operator", _operator), ("transformer", _transformer), ("vit", _vit),
                          ("clip", _clip), ("pi05", _pi05), ("graph", _graph)):
        register_workload(kind, builder)
    from .compiled_operators import install as install_compiled
    install_compiled()
    from .storage import install as install_storage
    install_storage()
    from .spatial_ops import install as install_spatial
    install_spatial()
    from .array_ops import install as install_array
    install_array()
    from .block_ops import install as install_block
    install_block()
    from .tensor_ops import NEW_KINDS, validate, reference as tensor_reference
    for kind in NEW_KINDS:
        register_operator(OperatorDefinition(kind, validate, reference=tensor_reference))
    from .indexing import KINDS as INDEX_KINDS,validate as index_validate,reference as index_reference
    for kind in set(INDEX_KINDS)-set(COSTS):
        register_operator(OperatorDefinition(kind,index_validate,reference=index_reference))
    from .transformer_block import build_block
    register_workload('transformer_block', build_block)
    from .transformer_block import build_structure
    from .composition import build_composition
    from .transformer_encoder import build_encoder
    register_workload('transformer_block_structure',build_structure)
    register_workload('composition',build_composition)
    register_workload('transformer_encoder',build_encoder)
    from .vit_classifier import build_classifier, build_structure, build_part
    register_workload('vit_classifier',build_classifier)
    register_workload('vit_classifier_structure',build_structure)
    register_workload('vit_part',build_part)
    from .clip_similarity import build_similarity,build_structure as clip_structure,build_part as clip_part
    register_workload('clip_similarity',build_similarity)
    register_workload('clip_structure',clip_structure)
    register_workload('clip_part',clip_part)
    from .mla_ops import install as install_mla
    install_mla()
    from .mla_decode import build_decode
    register_workload('mla_decode',build_decode)
    from .onnx_import import build_onnx
    register_workload('onnx', build_onnx)
