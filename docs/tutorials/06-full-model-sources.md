# 完整模型来源核查

这些哈希标识核查时的文件内容，不把可变main称为固定commit。上游文件快照保存在开发资料中，不随包导出。

| 来源 | SHA256 |
| --- | --- |
| [clip-model.py](https://raw.githubusercontent.com/openai/CLIP/main/clip/model.py) | `9902cbe5ee90a1da2aa3e6f043e8a23dc1f8831193b963785c9af03d5c7bef2c` |
| [vit-config.json](https://huggingface.co/google/vit-base-patch16-224/resolve/main/config.json) | `abdf73cc4ce79495a83f32a8c9265daa7a1363f5a3efa4704c88a6e7f45ad19d` |
| [clip-config.json](https://huggingface.co/openai/clip-vit-base-patch32/resolve/main/config.json) | `b575ef3c36f2a057fa19e221650105052d61cc9c1a972ec15019c6261ec98770` |
| [modeling-vit.py](https://raw.githubusercontent.com/huggingface/transformers/v4.40.2/src/transformers/models/vit/modeling_vit.py) | `402ed74cc061b54cafd42b27d92ac23b107f908ec1b1a362166eb98a7e06ef9b` |
| [modeling_vit.py（实现依据 v4.13.0）](https://raw.githubusercontent.com/huggingface/transformers/v4.13.0/src/transformers/models/vit/modeling_vit.py) | `14818ee658fb1837468a6127ca761df26ab7bf966ab670eccdee4c0bfbd88ced` |
| [activations.py（v4.13.0）](https://raw.githubusercontent.com/huggingface/transformers/v4.13.0/src/transformers/activations.py) | `6065ad152b42cf790f120037a2be94765b8514a27c6dd4de581f162c308b3825` |

ViT配置来自google分类checkpoint，其版本字段为4.13.0.dev0；完整图实现采用冻结的v4.13.0源码作为语义依据，v4.40.2是入口开发阶段的交叉核查来源，两者不是同一版本。未锁定原checkpoint精确开发commit或训练环境。原Google JAX代码仅作交叉参考。CLIP执行语义以OpenAI实现为准，openai/clip-vit-base-patch32配置用于逐项核对维度；这不把Transformers实现替代为原OpenAI执行实现。训练dropout在推理关闭。

完整图的ViT采用erf GELU和LayerNorm epsilon=1e-12；CLIP采用QuickGELU和epsilon=1e-5。归约/向量算术与存储cast政策在结果中明确记录，均一fp16/fp32存储方案不等于官方混合精度。小张量参考验证了声明的图算术，不能据此声称具名权重预测或硬件时延准确率。
