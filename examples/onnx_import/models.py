"""Generate two complete standard ONNX files and explicit, answer-free roots.

This generator uses ONNX's official construction and serialization helpers.
It never imports ArchExplorer's importer or computes intermediate activations.
"""
import argparse
import hashlib
import json
from pathlib import Path


def _dependencies():
    try:
        import onnx
    except ImportError as exc:
        raise ValueError('ONNX examples require the optional dependency: install arch_explorer[onnx]') from exc
    import numpy as np
    return onnx, np


def _values(np, shape, *, offset, scale=0.0625):
    count = 1
    for size in shape:
        count *= size
    return (((np.arange(count, dtype=np.int64) + offset) * 7 % 29 - 14)
            .astype(np.float32).reshape(shape) * np.float32(scale))


def _mlp(onnx, np, variant, symbolic_batch):
    h, tensor = onnx.helper, onnx.TensorProto
    batch = 'batch' if symbolic_batch else 3
    roots = {'X': _values(np, (3, 5), offset=3 + variant),
             'R': _values(np, (3, 7), offset=11 + variant, scale=0.03125)}
    weights = [onnx.numpy_helper.from_array(_values(np, shape, offset=offset + variant), name)
               for name, shape, offset in [('W0', (7, 5), 17), ('B0', (7,), 7),
                                            ('B_other', (7,), 23), ('W1', (7, 4), 31),
                                            ('B1', (4,), 41)]]
    gain = onnx.numpy_helper.from_array(
        np.array([0.5, 1.0, 1.5, -0.5, 0.25, -1.0, 2.0], dtype=np.float32), 'gain_literal')
    nodes = [
        h.make_node('Gemm', ['X', 'W0', 'B0'], ['projection_a'], name='',
                    alpha=0.75, beta=0.5, transB=1),
        h.make_node('Gemm', ['X', 'W0', 'B_other'], ['projection_b'], name='projection',
                    alpha=0.5, beta=1.25, transB=1),
        h.make_node('Add', ['projection_a', 'projection_b'], ['shared_sum'], name='residual'),
        h.make_node('Add', ['shared_sum', 'R'], ['preactivation'], name='residual'),
        h.make_node('Relu', ['preactivation'], ['hidden']),
        h.make_node('Gemm', ['hidden', 'W1', 'B1'], ['Y'], alpha=1.0, beta=1.0),
        h.make_node('Constant', [], ['gain'], value=gain),
        h.make_node('Mul', ['hidden', 'gain'], ['H_scaled'])]
    graph = h.make_graph(nodes, 'synthetic_mlp',
        [h.make_tensor_value_info('X', tensor.FLOAT, [batch, 5]),
         h.make_tensor_value_info('R', tensor.FLOAT, [batch, 7])],
        [h.make_tensor_value_info('Y', tensor.FLOAT, [batch, 4]),
         h.make_tensor_value_info('H_scaled', tensor.FLOAT, [batch, 7])], initializer=weights)
    return graph, roots, {name: list(value.shape) for name, value in roots.items()} if symbolic_batch else {}


def _attention(onnx, np, variant, symbolic_batch):
    h, tensor = onnx.helper, onnx.TensorProto
    batch = 'batch' if symbolic_batch else 2
    roots = {'Q': _values(np, (2, 3, 4), offset=5 + variant),
             'K': _values(np, (2, 5, 4), offset=13 + variant),
             'V': _values(np, (2, 5, 3), offset=19 + variant),
             'mask': _values(np, (1, 3, 5), offset=2 + variant, scale=0.125)}
    nodes = [
        h.make_node('Transpose', ['K'], ['K_transposed'], perm=[0, 2, 1], name=''),
        h.make_node('MatMul', ['Q', 'K_transposed'], ['scores'], name='projection'),
        h.make_node('Constant', [], ['scale'],
                    value=onnx.numpy_helper.from_array(np.array([0.5], dtype=np.float32), 'scale_literal')),
        h.make_node('Mul', ['scores', 'scale'], ['scaled_scores']),
        h.make_node('Add', ['scaled_scores', 'mask'], ['masked_scores']),
        h.make_node('Softmax', ['masked_scores'], ['probabilities'], axis=-1),
        h.make_node('MatMul', ['probabilities', 'V'], ['context'], name='projection'),
        h.make_node('Constant', [], ['output_shape'],
                    value=onnx.numpy_helper.from_array(np.array([-1, 3], dtype=np.int64), 'shape_literal')),
        h.make_node('Reshape', ['context', 'output_shape'], ['flattened'], allowzero=0),
        h.make_node('Identity', ['flattened'], ['Y'])]
    graph = h.make_graph(nodes, 'synthetic_attention',
        [h.make_tensor_value_info('Q', tensor.FLOAT, [batch, 3, 4]),
         h.make_tensor_value_info('K', tensor.FLOAT, [batch, 5, 4]),
         h.make_tensor_value_info('V', tensor.FLOAT, [batch, 5, 3]),
         h.make_tensor_value_info('mask', tensor.FLOAT, [1, 3, 5])],
        [h.make_tensor_value_info('Y', tensor.FLOAT, [None if symbolic_batch else 6, 3]),
         h.make_tensor_value_info('probabilities', tensor.FLOAT, [batch, 3, 5])])
    return graph, roots, {name: list(value.shape) for name, value in roots.items()} if symbolic_batch else {}


def generate_models(output, *, external_data=False, symbolic_batch=False, variant=0):
    """Return the generated input manifest; all paths are relative to output.

    ``external_data`` moves MLP initializers into a real tensor file. Constant
    node literals remain embedded, including the int64 Reshape shape tensor.
    ``symbolic_batch`` records explicit concrete root shapes beside
    each model; the importer remains responsible for checking that declaration.
    """
    if type(external_data) is not bool or type(symbolic_batch) is not bool:
        raise ValueError('external_data and symbolic_batch must be bool')
    if type(variant) is not int or not 0 <= variant <= 1000:
        raise ValueError('variant must be an integer in 0..1000')
    onnx, np = _dependencies()
    root = Path(output).resolve()
    root.mkdir(parents=True, exist_ok=False)
    cases = []
    for name, builder in (('mlp', _mlp), ('attention', _attention)):
        directory = root / name
        directory.mkdir()
        graph, roots, dimensions = builder(onnx, np, variant, symbolic_batch)
        model = onnx.helper.make_model(graph, ir_version=8,
            opset_imports=[onnx.helper.make_opsetid('', 17)],
            producer_name='ArchExplorer synthetic input example')
        model_path = directory / 'model.onnx'
        onnx.save_model(model, model_path, save_as_external_data=external_data,
                        all_tensors_to_one_file=True, location='tensors.bin',
                        size_threshold=0, convert_attribute=False)
        onnx.checker.check_model(str(model_path), full_check=True)
        input_path = directory / 'INPUTS.json'
        input_path.write_text(json.dumps({k: v.tolist() for k, v in roots.items()}, indent=2,
                                         allow_nan=False) + '\n')
        specialization = directory / 'INPUT-SHAPES.json'
        specialization.write_text(json.dumps(dimensions, indent=2) + '\n')
        files = []
        for path in sorted(directory.iterdir()):
            data = path.read_bytes()
            files.append({'path': str(path.relative_to(root)), 'bytes': len(data),
                          'sha256': hashlib.sha256(data).hexdigest()})
        cases.append({'case': name, 'model': str(model_path.relative_to(root)),
                      'inputs': str(input_path.relative_to(root)),
                      'input_shapes': str(specialization.relative_to(root)),
                      'external_root': str(directory.relative_to(root)),
                      'root_shapes': {k: list(v.shape) for k, v in roots.items()},
                      'outputs': [v.name for v in graph.output], 'files': files})
    manifest = {'schema': 'archexplorer.onnx-example-inputs.v1', 'ir_version': 8,
                'opset': 17, 'dtype': 'fp32', 'variant': variant,
                'symbolic_batch': symbolic_batch, 'external_data': external_data,
                'answers_included': False, 'cases': cases}
    (root / 'INPUT-MANIFEST.json').write_text(json.dumps(manifest, indent=2) + '\n')
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    parser.add_argument('--external-data', action='store_true')
    parser.add_argument('--symbolic-batch', action='store_true')
    parser.add_argument('--variant', type=int, default=0)
    args = parser.parse_args()
    print(json.dumps(generate_models(args.output, external_data=args.external_data,
                                     symbolic_batch=args.symbolic_batch, variant=args.variant), indent=2))
