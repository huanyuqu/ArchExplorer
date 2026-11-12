"""Optional CPU framework export. ONNX parsing, types and lowering belong to workload.

No framework dependency is imported until export_torch is called. The caller
supplies an actual module (or a TorchScript file), every input and output name.
"""
from copy import deepcopy
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform


def _dependencies():
    try:
        import torch
        import onnx
    except ImportError as exc:
        raise ImportError('Framework export requires the optional torch and onnx dependencies') from exc
    return torch, onnx


def _write(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')


def _input_tensors(torch, inputs):
    if not isinstance(inputs, dict) or not inputs:
        raise ValueError('inputs must map ordered original argument names to dtype/shape/values')
    dtypes = {'fp32': torch.float32, 'fp16': torch.float16, 'bf16': torch.bfloat16,
              'int32': torch.int32, 'int64': torch.int64, 'bool': torch.bool}
    tensors = []
    for name, item in inputs.items():
        if not isinstance(name, str) or not name or not isinstance(item, dict) or set(item) != {'dtype', 'shape', 'values'}:
            raise ValueError('Each named input requires exactly dtype, shape and values')
        shape = item['shape']
        if (not isinstance(shape, list) or any(type(n) is not int or n <= 0 for n in shape)
                or item['dtype'] not in dtypes):
            raise ValueError('Input dtype and positive concrete shape must be explicit')
        # Validate original JSON values before tensor construction can truncate,
        # coerce bools, or raise a backend-specific integer-overflow exception.
        if item['dtype'] in ('int32', 'int64', 'bool'):
            def check(value):
                if isinstance(value, list):
                    return all(check(v) for v in value)
                if item['dtype'] == 'bool':
                    return type(value) is bool
                limit = 2**(31 if item['dtype'] == 'int32' else 63)
                return type(value) is int and -limit <= value < limit
            if not check(item['values']):
                raise ValueError('Integer/bool input values must have their exact declared type and range')
        tensor = torch.tensor(item['values'], dtype=dtypes[item['dtype']], device='cpu')
        if list(tensor.shape) != shape:
            raise ValueError('Input values differ from the declared shape')
        if tensor.is_floating_point() and not bool(torch.isfinite(tensor).all()):
            raise ValueError('Input values must remain finite in their declared storage dtype')
        tensors.append(tensor)
    return tuple(tensors)


def export_torch(module, *, inputs, output_dir, output_names, input_names=None,
                 dynamic_axes=None, external_data=False):
    """Export inference on CPU with explicit legacy exporter, opset17 and parameters.

    inputs maps names to {dtype, shape, values}. input_names declares positional
    argument order independently of JSON key order; if omitted, insertion order
    is used for Python callers. Supported
    root storage names are fp32/fp16/bf16/int32/int64/bool. output_dir must be new.
    A path loads a caller-supplied TorchScript file on CPU. A module object is
    deep-copied before eval(); the caller's training flags/weights are unchanged.
    dynamic_axes declares symbolic dimensions; INPUT-SHAPES always records the
    full actual root shapes, never symbolic-name substitutions or inferred roots.
    The returned manifest contains files, original node/attribute/type identities
    and export options, with no numerical intermediate or output answers.
    """
    torch, onnx = _dependencies()
    if not isinstance(inputs, dict) or not inputs:
        raise ValueError('inputs must map original argument names to dtype/shape/values')
    if input_names is not None:
        if (not isinstance(input_names, (list, tuple)) or len(input_names) != len(inputs)
                or any(not isinstance(n, str) for n in input_names) or set(input_names) != set(inputs)):
            raise ValueError('input_names must list every root exactly once in module argument order')
        inputs = {name: inputs[name] for name in input_names}
    if type(external_data) is not bool:
        raise ValueError('external_data must be bool')
    if (not isinstance(output_names, (list, tuple)) or not output_names
            or any(not isinstance(n, str) or not n for n in output_names)
            or len(set(output_names)) != len(output_names)):
        raise ValueError('output_names requires distinct explicit nonempty names')
    if dynamic_axes is not None:
        if not isinstance(dynamic_axes, dict) or set(dynamic_axes) - set(inputs) - set(output_names):
            raise ValueError('dynamic_axes must name declared inputs or outputs')
        for axes in dynamic_axes.values():
            if not isinstance(axes, dict) or any(type(a) is not int or a < 0 or not isinstance(s, str) or not s
                                                 for a, s in axes.items()):
                raise ValueError('dynamic_axes requires nonnegative axes mapped to symbol names')
    tensors = _input_tensors(torch, inputs)
    for name, axes in (dynamic_axes or {}).items():
        if name in inputs and any(axis >= len(inputs[name]['shape']) for axis in axes):
            raise ValueError('dynamic_axes axis exceeds the declared input rank')
    source = None
    if isinstance(module, (str, Path)):
        model_file = Path(module)
        source = {'sha256': hashlib.sha256(model_file.read_bytes()).hexdigest(), 'format': 'torchscript'}
        model = torch.jit.load(str(model_file), map_location='cpu')
    else:
        if not isinstance(module, torch.nn.Module):
            raise TypeError('module must be an actual torch.nn.Module or TorchScript file')
        model = deepcopy(module)
    if any(value.device.type != 'cpu' for value in (*model.parameters(), *model.buffers())):
        raise ValueError('Framework export accepts CPU modules only')
    model.eval()
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)
    options = {'dynamo': False, 'opset_version': 17, 'export_params': True,
               'do_constant_folding': True, 'keep_initializers_as_inputs': False,
               'training': 'EVAL', 'external_data': external_data,
               'dynamic_axes': dynamic_axes or {}, 'device': 'cpu'}
    manifest = {'schema': 'arch-explorer.framework-export.v1', 'status': 'started',
                'framework': {'torch': torch.__version__, 'onnx': onnx.__version__,
                              'python': platform.python_version()},
                'options': options, 'module_source': source, 'answers_included': False}
    try:
        with torch.no_grad():
            torch.onnx.export(model, tensors, str(output / 'model.onnx'), dynamo=False,
                export_params=True, opset_version=17, do_constant_folding=True,
                keep_initializers_as_inputs=False, training=torch.onnx.TrainingMode.EVAL,
                input_names=list(inputs), output_names=list(output_names), dynamic_axes=dynamic_axes)
        proto = onnx.load(str(output / 'model.onnx'), load_external_data=True)
        # Inspect actual exporter IR. Never relabel a protobuf version number.
        if proto.ir_version != 8 or [(x.domain, x.version) for x in proto.opset_import] != [('', 17)]:
            raise ValueError('Exporter must produce actual IR8 / standard opset17')
        if external_data:
            onnx.save_model(proto, str(output / 'model.onnx'), save_as_external_data=True,
                all_tensors_to_one_file=True, location='weights.bin', size_threshold=0,
                convert_attribute=False)
        onnx.checker.check_model(str(output / 'model.onnx'), full_check=True)
        root_names = [value.name for value in proto.graph.input]
        if root_names != list(inputs):
            raise ValueError('Exporter removed, renamed or reordered a declared root')
        if [value.name for value in proto.graph.output] != list(output_names):
            raise ValueError('Exporter output arity/names differ from the complete declared output_names')
        ports = {value.name: value for value in (*proto.graph.input, *proto.graph.output)}
        for name, axes in (dynamic_axes or {}).items():
            dimensions = ports[name].type.tensor_type.shape.dim
            for axis, symbol in axes.items():
                if axis >= len(dimensions):
                    raise ValueError('dynamic_axes axis exceeds the exported port rank')
                if dimensions[axis].dim_param != symbol:
                    raise ValueError('Exporter did not preserve a declared dynamic axis symbol')
        roots = {name: tensor.to(torch.float32).tolist() if tensor.is_floating_point() else tensor.tolist()
                 for name, tensor in zip(inputs, tensors)}
        _write(output / 'INPUTS.json', roots)
        _write(output / 'INPUT-SHAPES.json', {name: list(t.shape) for name, t in zip(inputs, tensors)})
        _write(output / 'TORCH-INPUTS.json', inputs)
        def value_info(value):
            typed = value.type.tensor_type
            return {'name': value.name, 'dtype': onnx.TensorProto.DataType.Name(typed.elem_type),
                    'shape': [d.dim_value if d.HasField('dim_value') else d.dim_param for d in typed.shape.dim]}
        nodes = [{'index': i, 'name': n.name, 'domain': n.domain, 'op_type': n.op_type,
                  'inputs': list(n.input), 'outputs': list(n.output),
                  'attributes': [onnx.helper.printable_attribute(a) for a in n.attribute]}
                 for i, n in enumerate(proto.graph.node)]
        _write(output / 'OPERATORS.json', {'ir_version': proto.ir_version,
            'opsets': [{'domain': x.domain, 'version': x.version} for x in proto.opset_import],
            'nodes': nodes, 'inputs': [value_info(x) for x in proto.graph.input],
            'outputs': [value_info(x) for x in proto.graph.output],
            'initializers': [{'name': x.name, 'dtype': onnx.TensorProto.DataType.Name(x.data_type),
                              'shape': list(x.dims)} for x in proto.graph.initializer]})
        manifest.update(status='exported-and-checked', ir_version=proto.ir_version,
            operators=sorted({n.op_type for n in proto.graph.node}), nodes=len(nodes),
            roots=root_names, output_names=list(output_names),
            files=[{'path': p.name, 'bytes': p.stat().st_size,
                    'sha256': hashlib.sha256(p.read_bytes()).hexdigest()}
                   for p in sorted(output.iterdir()) if p.is_file()])
        return manifest
    except BaseException as exc:
        manifest.update(status='failed', error={'type': type(exc).__name__, 'message': str(exc)})
        raise
    finally:
        _write(output / 'EXPORT.json', manifest)


def import_export(directory, *, name='onnx', **import_options):
    """Use the ordinary authoritative importer on a completed export directory."""
    from .onnx_compilation import onnx_workload_from_files
    root = Path(directory)
    return onnx_workload_from_files(root / 'model.onnx', inputs_path=root / 'INPUTS.json',
        input_shapes_path=root / 'INPUT-SHAPES.json',
        external_root=root, name=name, **import_options)
