# 可重现的框架输入

`python models.py --output models [--reference-output PRIVATE_DIR]` 用固定 seed34 和确定性合成权重，真实导出 MLP 与 decoder，各有 FP32/FP16/BF16 和 embedded/external 变体。符号 batch 配完整原根 shape，目录均须不存在。decoder 含投影、两次 LayerNorm、残差、精确 GELU 和 FP32 softmax 的显式存储转换。

输入包只有模型、权重、根值与 shape、实际节点清单和导出身份。可选 framework forward 输出写入单独的 PRIVATE_DIR，不进入无答案输入。例子调用公共 export_torch，不构造 ArchExplorer 图。

`vision_language.py` 真实导出完整小型视觉分类器和图文双塔。固定 seed35、合成权重，CPU batch 为2张图/3组文本；Conv patch、两层注意力块、类/位置标记与分类头完整。文本使用 int64 lookup、causal mask、首个最大token行选择；双塔投影与归一化后产生2×3和3×2 logits。FP16/BF16 是实际输入与模块存储；LayerNorm/softmax、最终 L2 和 logit_scale 的 FP32 边界显式保留。

```sh
python vision_language.py --family vision --dtype fp32 --output vision-fp32
python vision_language.py --family dual --dtype fp16 --external-data --output dual-fp16
python vision_language.py --family dual --dtype bf16 --external-data --output dual-bf16
```

BF16 patch 由模块显式转 FP32 Conv 再转回存储，因为 opset17 的标准 Conv 不含 BF16 类型。折叠后 initializer 的实际 dtype 和 Cast 以文件为准。原生 BF16 Conv 的官方 checker 拒绝与补充路径分开留证；不改 IR 或 dtype 标签。`--reference-output PRIVATE_DIR` 将真实 torch 答案单独保存，模型目录仍无答案。

`emit.py DIRECTORY --output NEW_DIR [--max-evaluations 12] [--candidate 0|1] [--layout materialized|views] [--resource-budget SMALL-BUDGET.json]` 读取实际导出目录，发出普通 SCENE/CONFIG/AGENT-CONFIG 等 JSON；随后用现有 compile/prepare/evaluate/search/agent/rerun 命令。默认探索两组放置和两布局；指定 candidate/layout 可限制复杂模型的一次有界消费。max_evaluations 是预算上限；失败仍记账。支持域由公共 importer 决定，导出成功不会绕过不支持节点的诊断。

两层双塔导出超过默认256节点限额。先按[框架入口](../../docs/framework-export.md)保存显式 SMALL-BUDGET.json，再运行 `python emit.py dual-fp16 --output inputs --candidate 1 --layout views --max-evaluations 3 --resource-budget SMALL-BUDGET.json`；该策略在根分配前预检，并随 WorkloadSpec 和 winner 保存。然后将 inputs/CONFIG.json 交通用 search、inputs/AGENT-CONFIG.json 交离线 scripted Agent。winner 保存全量规则和自包含模型/外部来源；evaluate BEST.json 是只读校验，rerun BEST.json --work-dir NEW 才是新执行。完整命令与阶段边界见下文链接。

用户模块和 TorchScript CLI、完整字段及重放流程见 [框架入口](../../docs/framework-export.md)。
