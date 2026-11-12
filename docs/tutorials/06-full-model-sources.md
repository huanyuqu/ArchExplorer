# Complete model source verification

These hashes identify the file contents at verification time and do not call a mutable main a fixed commit. Upstream file snapshots are kept in development materials and are not exported with the package.

| Source | SHA256 |
| --- | --- |
| [clip-model.py](https://raw.githubusercontent.com/openai/CLIP/main/clip/model.py) | `9902cbe5ee90a1da2aa3e6f043e8a23dc1f8831193b963785c9af03d5c7bef2c` |
| [vit-config.json](https://huggingface.co/google/vit-base-patch16-224/resolve/main/config.json) | `abdf73cc4ce79495a83f32a8c9265daa7a1363f5a3efa4704c88a6e7f45ad19d` |
| [clip-config.json](https://huggingface.co/openai/clip-vit-base-patch32/resolve/main/config.json) | `b575ef3c36f2a057fa19e221650105052d61cc9c1a972ec15019c6261ec98770` |
| [modeling-vit.py](https://raw.githubusercontent.com/huggingface/transformers/v4.40.2/src/transformers/models/vit/modeling_vit.py) | `402ed74cc061b54cafd42b27d92ac23b107f908ec1b1a362166eb98a7e06ef9b` |
| [modeling_vit.py (implementation basis v4.13.0)](https://raw.githubusercontent.com/huggingface/transformers/v4.13.0/src/transformers/models/vit/modeling_vit.py) | `14818ee658fb1837468a6127ca761df26ab7bf966ab670eccdee4c0bfbd88ced` |
| [activations.py (v4.13.0)](https://raw.githubusercontent.com/huggingface/transformers/v4.13.0/src/transformers/activations.py) | `6065ad152b42cf790f120037a2be94765b8514a27c6dd4de581f162c308b3825` |

The ViT configuration comes from the Google classification checkpoint, whose version field is 4.13.0.dev0; the complete graph implementation uses the frozen v4.13.0 source code as the semantic basis, and v4.40.2 is a cross-check source during the entry development phase; the two are not the same version. The exact development commit or training environment of the original checkpoint is not locked. The original Google JAX code is only a cross-reference. CLIP execution semantics are based on the OpenAI implementation, and the openai/clip-vit-base-patch32 configuration is used for item-by-item dimension verification; this does not replace the Transformers implementation with the original OpenAI execution implementation. Training dropout is disabled during inference.

The full-graph ViT uses erf GELU and LayerNorm epsilon=1e-12; CLIP uses QuickGELU and epsilon=1e-5. Reduction/vector arithmetic and storage cast policies are explicitly recorded in the results; a uniform fp16/fp32 storage scheme is not equivalent to official mixed precision. Small-tensor reference verification validates the declared graph arithmetic, but cannot be used to claim named-weight prediction or hardware latency accuracy.
