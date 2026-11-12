"""Real PyTorch modules with synthetic weights; no ArchExplorer graph construction."""
import argparse
import importlib.metadata
import json
from pathlib import Path
import sys


def models(torch):
    class MLP(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.input = torch.nn.Linear(5, 7)
            self.output = torch.nn.Linear(7, 4)

        def forward(self, X, R):
            hidden = torch.nn.functional.gelu(self.input(X) + R, approximate='none')
            return self.output(hidden), hidden

    class Decoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.norm1 = torch.nn.LayerNorm(4, eps=1e-5)
            self.norm2 = torch.nn.LayerNorm(4, eps=1e-5)
            self.q = torch.nn.Linear(4, 4)
            self.k = torch.nn.Linear(4, 4)
            self.v = torch.nn.Linear(4, 4)
            self.projection = torch.nn.Linear(4, 4)
            self.up = torch.nn.Linear(4, 7)
            self.down = torch.nn.Linear(7, 4)

        def norm(self, x, layer):
            return torch.nn.functional.layer_norm(x.float(), (4,), layer.weight.float(),
                                                  layer.bias.float(), layer.eps).to(x.dtype)

        def forward(self, X, mask):
            normalized = self.norm(X, self.norm1)
            q, k, v = self.q(normalized), self.k(normalized), self.v(normalized)
            scores = torch.matmul(q, k.transpose(-1, -2)) * 0.5 + mask
            probabilities = torch.softmax(scores.float(), dim=-1).to(X.dtype)
            residual = X + self.projection(torch.matmul(probabilities, v))
            hidden = torch.nn.functional.gelu(self.up(self.norm(residual, self.norm2)), approximate='none')
            return residual + self.down(hidden), probabilities
    return {'mlp': MLP, 'decoder': Decoder}


def generate(output, *, reference_output=None):
    import torch
    from arch_explorer.archs.framework_export import export_torch
    torch.set_num_threads(1)
    root = Path(output).resolve()
    root.mkdir(parents=True, exist_ok=False)
    refs = Path(reference_output).resolve() if reference_output else None
    if refs:
        refs.mkdir(parents=True, exist_ok=False)
    result = {'schema': 'arch-explorer.framework-inputs.v1', 'answers_included': False,
              'torch': torch.__version__, 'onnx': importlib.metadata.version('onnx'),
              'seed': 34, 'cases': [], 'failures': []}
    try:
        for family, cls in models(torch).items():
            for dtype, storage in [('fp32', torch.float32), ('fp16', torch.float16), ('bf16', torch.bfloat16)]:
                torch.manual_seed(34)
                module = cls().to(dtype=storage).eval()
                with torch.no_grad():
                    for i, (name, value) in enumerate(module.named_parameters()):
                        base = ((torch.arange(value.numel()).reshape(value.shape) * 7 + i * 3) % 23 - 11).float() / 64
                        if 'norm' in name and name.endswith('weight'):
                            base += 1
                        value.copy_(base.to(storage))
                shapes = {'X': [2, 5], 'R': [2, 7]} if family == 'mlp' else {'X': [2, 3, 4], 'mask': [1, 3, 3]}
                inputs, tensors = {}, []
                for i, (name, shape) in enumerate(shapes.items()):
                    count = 1
                    for size in shape:
                        count *= size
                    value = (((torch.arange(count).reshape(shape) * 5 + i * 7) % 19 - 9).float() / 32).to(storage)
                    if name == 'mask':
                        value = torch.triu(torch.full(shape, -64.0, dtype=storage), diagonal=1)
                    inputs[name] = {'dtype': dtype, 'shape': shape, 'values': value.float().tolist()}
                    tensors.append(value)
                names = ['Y', 'hidden' if family == 'mlp' else 'probabilities']
                axes = {n: {0: 'batch'} for n in ['X', *(['R'] if family == 'mlp' else []), *names]}
                if refs:
                    with torch.no_grad():
                        values = module(*tensors)
                    (refs / f'{family}-{dtype}.json').write_text(json.dumps({
                        'framework': torch.__version__, 'dtype': dtype, 'family': family,
                        'inputs': inputs, 'outputs': {n: v.float().tolist() for n, v in zip(names, values)}},
                        indent=2, allow_nan=False) + '\n')
                for external in (False, True):
                    case = f'{family}-{dtype}-' + ('external' if external else 'embedded')
                    try:
                        manifest = export_torch(module, inputs=inputs, output_dir=root / case,
                            output_names=names, dynamic_axes=axes, external_data=external)
                        result['cases'].append({'id': case, 'directory': case, 'family': family,
                            'dtype': dtype, 'external_data': external, 'export': manifest})
                    except Exception as exc:
                        result['failures'].append({'case': case, 'type': type(exc).__name__, 'message': str(exc)})
        if result['failures']:
            raise RuntimeError('Some actual framework exports failed; see INPUT-MANIFEST.json and EXPORT.json')
        return result
    finally:
        (root / 'INPUT-MANIFEST.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    parser.add_argument('--reference-output')
    args = parser.parse_args()
    result = generate(args.output, reference_output=args.reference_output)
    print(json.dumps({'cases': len(result['cases']), 'failures': result['failures']}))
