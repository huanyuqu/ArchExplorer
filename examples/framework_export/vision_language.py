"""Actual small CPU PyTorch vision/classifier and image/text modules.

These are synthetic end-to-end models, not ArchExplorer graphs or pretrained
weights. Shape, indexing, precision and arithmetic are exported by PyTorch.
"""
import argparse
import hashlib
import json
from pathlib import Path


def model_classes(torch):
    class Block(torch.nn.Module):
        def __init__(self, width=4, heads=2, intermediate=7):
            super().__init__()
            self.heads, self.head_dim = heads, width // heads
            self.norm1 = torch.nn.LayerNorm(width, eps=1e-5)
            self.norm2 = torch.nn.LayerNorm(width, eps=1e-5)
            self.q = torch.nn.Linear(width, width)
            self.k = torch.nn.Linear(width, width)
            self.v = torch.nn.Linear(width, width)
            self.output = torch.nn.Linear(width, width)
            self.up = torch.nn.Linear(width, intermediate)
            self.down = torch.nn.Linear(intermediate, width)

        def normalize(self, x, layer):
            return torch.nn.functional.layer_norm(x.float(), (4,), layer.weight.float(),
                layer.bias.float(), layer.eps).to(x.dtype)

        def forward(self, x, mask=None):
            batch, length, width = x.shape
            n = self.normalize(x, self.norm1)
            q = self.q(n).reshape(batch, length, self.heads, self.head_dim).transpose(1, 2)
            k = self.k(n).reshape(batch, length, self.heads, self.head_dim).transpose(1, 2)
            v = self.v(n).reshape(batch, length, self.heads, self.head_dim).transpose(1, 2)
            scores = torch.matmul(q, k.transpose(-1, -2)) * (self.head_dim ** -0.5)
            if mask is not None:
                scores = scores + mask
            probability = torch.softmax(scores.float(), dim=-1).to(x.dtype)
            attended = torch.matmul(probability, v).transpose(1, 2).reshape(batch, length, width)
            residual = x + self.output(attended)
            hidden = torch.nn.functional.gelu(self.up(self.normalize(residual, self.norm2)), approximate='none')
            return residual + self.down(hidden)

    class Vision(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.patch = torch.nn.Conv2d(3, 4, kernel_size=(2, 3), stride=(2, 3), bias=True)
            self.class_token = torch.nn.Parameter(torch.zeros(1, 1, 4))
            self.position = torch.nn.Parameter(torch.zeros(1, 5, 4))
            self.blocks = torch.nn.ModuleList([Block(), Block()])
            self.norm = torch.nn.LayerNorm(4, eps=1e-5)

        def forward(self, images):
            if images.dtype == torch.bfloat16:
                # Conv at opset17 has no BF16 type. Keep actual BF16 module
                # storage and make the accumulation/conversion explicit in
                # the framework itself, before its standard ONNX export.
                patch = torch.nn.functional.conv2d(images.float(), self.patch.weight.float(),
                    self.patch.bias.float(), stride=self.patch.stride,
                    padding=self.patch.padding, dilation=self.patch.dilation, groups=self.patch.groups).to(images.dtype)
            else:
                patch = self.patch(images)
            tokens = patch.flatten(2).transpose(1, 2)
            cls = self.class_token.expand(images.shape[0], -1, -1)
            x = torch.cat((cls, tokens), dim=1) + self.position
            for block in self.blocks:
                x = block(x)
            x = torch.nn.functional.layer_norm(x.float(), (4,), self.norm.weight.float(),
                self.norm.bias.float(), self.norm.eps).to(x.dtype)
            return x[:, 0, :]

    class Classifier(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.vision = Vision()
            self.head = torch.nn.Linear(4, 3)

        def forward(self, images):
            return self.head(self.vision(images))

    class Text(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = torch.nn.Embedding(16, 4)
            self.position = torch.nn.Parameter(torch.zeros(4, 4))
            self.blocks = torch.nn.ModuleList([Block(), Block()])
            self.norm = torch.nn.LayerNorm(4, eps=1e-5)
            self.register_buffer('causal', torch.triu(torch.full((4, 4), -64.0), diagonal=1))

        def forward(self, tokens):
            length = tokens.shape[1]
            x = self.embedding(tokens) + self.position[:length]
            for block in self.blocks:
                x = block(x, self.causal[:length, :length])
            x = torch.nn.functional.layer_norm(x.float(), (4,), self.norm.weight.float(),
                self.norm.bias.float(), self.norm.eps).to(x.dtype)
            first_maximum = tokens.argmax(dim=-1)
            return x[torch.arange(tokens.shape[0], device=tokens.device), first_maximum]

    class Dual(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.vision = Vision()
            self.text = Text()
            self.image_projection = torch.nn.Parameter(torch.zeros(4, 3))
            self.text_projection = torch.nn.Parameter(torch.zeros(4, 3))
            self.logit_scale = torch.nn.Parameter(torch.tensor(0.25, dtype=torch.float32))

        def forward(self, images, tokens):
            image = (self.vision(images) @ self.image_projection).float()
            text = (self.text(tokens) @ self.text_projection).float()
            image = image / image.norm(dim=-1, keepdim=True)
            text = text / text.norm(dim=-1, keepdim=True)
            scaled_image = self.logit_scale.float().exp() * image
            logits = scaled_image @ text.transpose(0, 1)
            return logits, logits.transpose(0, 1)
    return {'vision': Classifier, 'dual': Dual}


def make_model_and_inputs(family, dtype):
    import torch
    if family not in ('vision', 'dual') or dtype not in ('fp32', 'fp16', 'bf16'):
        raise ValueError('family requires vision/dual and dtype requires fp32/fp16/bf16')
    torch.manual_seed(35)
    storage = {'fp32': torch.float32, 'fp16': torch.float16, 'bf16': torch.bfloat16}[dtype]
    module = model_classes(torch)[family]().to(dtype=storage).eval()
    with torch.no_grad():
        for i, (name, value) in enumerate(module.named_parameters()):
            values = ((torch.arange(value.numel()).reshape(value.shape) * 7 + i * 3) % 29 - 14).float() / 64
            if 'norm' in name and name.endswith('weight'):
                values += 1
            value.copy_(values.to(storage))
    if family == 'dual':
        module.logit_scale = torch.nn.Parameter(torch.tensor(0.25, dtype=torch.float32))
    images = (((torch.arange(2 * 3 * 4 * 6).reshape(2, 3, 4, 6) * 5) % 23 - 11).float() / 32).to(storage)
    inputs = {'images': {'dtype': dtype, 'shape': list(images.shape), 'values': images.float().tolist()}}
    names, outputs = ['images'], ['logits']
    axes = {'images': {0: 'image_batch'}, 'logits': {0: 'image_batch'}}
    if family == 'dual':
        # Repeated maxima force first-argmax; image/text batch dimensions differ.
        tokens = torch.tensor([[1, 15, 15, 2], [3, 5, 7, 15], [15, 4, 15, 6]], dtype=torch.int64)
        inputs['tokens'] = {'dtype': 'int64', 'shape': list(tokens.shape), 'values': tokens.tolist()}
        names, outputs = ['images', 'tokens'], ['logits_per_image', 'logits_per_text']
        axes = {'images': {0: 'image_batch'}, 'tokens': {0: 'text_batch', 1: 'sequence'},
            'logits_per_image': {0: 'image_batch', 1: 'text_batch'},
            'logits_per_text': {0: 'text_batch', 1: 'image_batch'}}
    dimensions = {'image_batch': 2, 'image_channels': 3, 'image_height': 4, 'image_width': 6,
        'patch': [2, 3], 'width': 4, 'heads': 2, 'intermediate': 7, 'vision_layers': 2,
        'classes': 3} if family == 'vision' else {'image_batch': 2, 'text_batch': 3,
        'image_channels': 3, 'image_height': 4, 'image_width': 6, 'patch': [2, 3],
        'width': 4, 'heads': 2, 'intermediate': 7, 'vision_layers': 2, 'text_layers': 2,
        'vocabulary': 16, 'maximum_sequence': 4, 'sequence': 4, 'projection': 3}
    return module, inputs, names, outputs, axes, dimensions


def generate(output, *, family, dtype='fp32', external_data=False, reference_output=None):
    import torch
    from arch_explorer.archs.framework_export import export_torch
    torch.set_num_threads(1)
    model, inputs, names, outputs, axes, dimensions = make_model_and_inputs(family, dtype)
    before = {n: t.clone() for n, t in model.state_dict().items()}
    training = {n: m.training for n, m in model.named_modules()}
    exported = export_torch(model, inputs=inputs, input_names=names, output_dir=output,
        output_names=outputs, dynamic_axes=axes, external_data=external_data)
    assert all(torch.equal(before[n], t) for n, t in model.state_dict().items())
    assert training == {n: m.training for n, m in model.named_modules()}
    metadata = {'schema': 'arch-explorer.framework-model35.v1', 'family': family, 'dtype': dtype,
        'seed': 35, 'dimensions': dimensions, 'input_names': names, 'output_names': outputs,
        'external_data': external_data, 'caller_state_unchanged': True,
        'synthetic_weights': True, 'pretrained_weights': False, 'answers_included': False,
        'generator_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'scope': 'Small complete synthetic model; not paper-scale or pretrained model accuracy.'}
    Path(output, 'MODEL.json').write_text(json.dumps(metadata, indent=2, allow_nan=False) + '\n')
    if reference_output is not None:
        reference = Path(reference_output)
        reference.mkdir(parents=True, exist_ok=False)
        tensors = tuple(torch.tensor(inputs[n]['values'], dtype={'fp32': torch.float32, 'fp16': torch.float16,
            'bf16': torch.bfloat16, 'int64': torch.int64}[inputs[n]['dtype']]) for n in names)
        with torch.no_grad():
            values = model(*tensors)
        values = values if isinstance(values, tuple) else (values,)
        assert len(values) == len(outputs) and all(bool(torch.isfinite(v).all()) for v in values)
        (reference / 'TORCH.json').write_text(json.dumps({'framework': torch.__version__,
            'family': family, 'dtype': dtype, 'dimensions': dimensions,
            'outputs': {n: v.float().tolist() for n, v in zip(outputs, values)}}, indent=2, allow_nan=False) + '\n')
    return exported


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--family', choices=('vision', 'dual'), required=True)
    parser.add_argument('--dtype', choices=('fp32', 'fp16', 'bf16'), default='fp32')
    parser.add_argument('--external-data', action='store_true')
    parser.add_argument('--output', required=True)
    parser.add_argument('--reference-output')
    args = parser.parse_args()
    result = generate(args.output, family=args.family, dtype=args.dtype, external_data=args.external_data,
        reference_output=args.reference_output)
    print(json.dumps({'status': result['status'], 'family': args.family, 'dtype': args.dtype,
        'nodes': result['nodes'], 'operators': result['operators']}))
