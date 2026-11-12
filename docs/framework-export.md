# 实际 PyTorch 模块导出与普通 ONNX 用户流程

框架导出是可选入口。已验证版本为 CPU PyTorch 2.5.1+cpu、ONNX1.16.2，使用 legacy exporter（dynamo=False）、opset17、推理模式、参数导出和常量折叠。实际生成 IR8；其他 IR 明确拒绝，不改 protobuf 版本数。没有 PyTorch/ONNX 时，核心库、CLI 与比较器仍可导入。建议独立环境按 [官方 CPU 安装表](https://pytorch.org/get-started/previous-versions/)安装精确版本，不下载预训练权重。

Python 入口 `arch_explorer.archs.framework_export.export_torch(module, *, inputs, output_dir, output_names, input_names=None, dynamic_axes=None, external_data=False)` 接受实际 CPU `torch.nn.Module` 或用户的 TorchScript 文件。对象先复制再 eval，原权重和 training 标志保持。只读取用户明确提供的所有根；不按模型名字替换图。

`inputs` 每个名字的值恰为 `{dtype, shape, values}`。dtype 为 fp32/fp16/bf16/int32/int64/bool，shape 是完整正整数列表，标量为 []；values 是完整有限值。输入在导出前实际转换到声明的存储精度，低精度根不是 FP32 元数据标签。整数根在 tensor 分配前拒绝小数、bool 和越界，布尔根要求 JSON bool；合法 int64 直接从整数构造并以整数保存，包含大于2^53的值。

`input_names` 是模块位置参数顺序，且必须恰覆盖全部根。Python 调用者可省略并采用自己的字典插入顺序；序列化 JSON 流程应显式传该列表。CLI 必须用 `--input-names`，不依赖 JSON 对象键顺序。`output_names` 必须给全部不同输出；生成后逐项核实际根名和输出名，不能静默删根或漏输出。

```sh
archexplorer export-torch module.pt --inputs TORCH-INPUTS.json \
  --input-names X R --output-names Y hidden --external-data \
  --dynamic-axes AXES.json --output-dir exported
```

AXES.json 形如 `{"X":{"0":"batch"},"R":{"0":"batch"},"Y":{"0":"batch"},"hidden":{"0":"batch"}}`。动态轴必须在实际端口秩内；输入轴先检查，导出后再核对所有声明端口的符号确已保留。INPUT-SHAPES.json 保存每个原根的完整实际 shape，导入时按原根名特化。符号不能解除模型的位置表、词表或空间尺寸约束。external_data 由官方 ONNX save_model 把真实初始化权重放入 weights.bin，location 相对 model.onnx；Constant 节点保留导出器原表示。

输出目录必须新建，包含 model.onnx、可选 weights.bin、INPUTS.json、INPUT-SHAPES.json、TORCH-INPUTS.json、OPERATORS.json 和 EXPORT.json。OPERATORS 列实际 node/domain/opset/属性/原名和 initializer dtype/shape。EXPORT 保存固定导出选项、依赖版本及文件摘要，导出失败后也在 finally 保存原类型和诊断。它记录输入，不含数值答案。

根类型、形状和输入顺序等调用边界在创建输出目录前检查；此时的拒绝由调用者保存原异常或CLI日志，尚无EXPORT文件。已经进入导出阶段的失败由EXPORT.json记录。

`import_export(directory, *, name='onnx', **import_options)` 先公共 onnx_preflight，再加载数值根，交唯一正式 ONNX importer；可用 `resource_budget` 传入完整策略。省略策略保持旧小包表示。具体支持域、预估资源、存储舍入和 lowering 由 workload 的接口决定。之后复用 `onnx_scene`、`onnx_search_config`、`onnx_agent_config`、`layout_candidates` 和普通 Env/CLI：

```sh
python emit.py exported --output inputs --max-evaluations 12
archexplorer compile inputs/SCENE.json --output COMPILED.json
archexplorer prepare COMPILED.json --output request
archexplorer evaluate request/COMPILATION-REQUEST.json --output RESULT.json
archexplorer search --config inputs/CONFIG.json --policy greedy --output BEST.json
archexplorer agent run --config inputs/AGENT-CONFIG.json --policy scripted --output AGENT.json
archexplorer evaluate BEST.json --output READONLY.json
archexplorer rerun BEST.json --work-dir new-run --output RERUN.json
```

两层视觉/文本双塔超过 importer 默认256节点限额。对本目录小型模型，先将以下显式策略保存为 SMALL-BUDGET.json；它声明预检和执行资源上限，随自包含来源保存，其他模型应依据自己的资源约束选择。

```json
{
  "schema": "arch-explorer.resource-budget.v1",
  "max_dimension": 8192,
  "max_host_bytes": 8589934592,
  "max_matrix_fmas": 100000000,
  "max_nodes": 4096,
  "max_rank": 4,
  "max_root_bytes": 33554432,
  "max_source_bytes": 16777216,
  "max_steps": 32768,
  "max_tensor_elements": 4000000,
  "max_tiles": 4096,
  "max_total_tensor_bytes": 268435456,
  "reference_mode": "execute"
}
```

运行 `python emit.py exported --output inputs --candidate 1 --layout views --max-evaluations 3 --resource-budget SMALL-BUDGET.json` 可限制一组放置与布局。`candidate=None, layout=None` 的默认值覆盖两组放置和两布局；限制条件只过滤公共配置，不改变 Graph、数值或成本。SCENE 与单次 Agent 使用所选组合，未指定时用0/materialized。max_evaluations 始终是上限，并受所选配置原预算限制；3单位允许一次基线、一次容量拒绝与一次带宽提案，实际是否执行及成功以账本为准。省略 resource_budget 保留原默认上限；拒绝原文必须留存，不能将超额调用记为成功。

预算是评价单位上限，不代表不同候选的数量。默认配置包含两组逐原算子放置/分块与两布局，并探索明确的容量和带宽；失败也按原 Env 合同记账。保存的 winner 携带全量有效规则、来源、输入、shape、预算、adapter/implementation 和编译身份。fresh 读取的纯解析/编译校验与产物 prepare、engine、solver、写入分别核查；显式 rerun 在新目录真实执行并另计费。

框架 forward、官方 ONNX evaluator、独立 CPU 参考、映射 CPU、组件计时、安装验收分别留证。BF16 官方 evaluator 的执行完成不等于数值通过；原始差异或不支持诊断应保留，不能将错误结果当 oracle。原始私有答案、论文测量和硬件标定不进入例子或公开导出。

`vision_language.py` 提供固定种子的实际视觉分类器与图文双塔。视觉输入为[2,3,4,6]，Conv patch为[2,3]，类标记/位置与两层完整块后接3类头；双塔另有 int64 tokens[3,4]、两层 causal 文本块、首个最大token的行选择、两投影/L2归一化和FP32 exp缩放，输出[2,3]与[3,2]。相同语义提供 FP32/FP16/BF16，精确维度与合成参数以 MODEL/原文件为准。小型完整结构不表示预训练准确率或论文规模。

BF16 的 patch 在真实框架模块里显式转 FP32 做 Conv，再转回 BF16；ONNX opset17 的 [Conv 类型约束](https://onnx.ai/onnx/operators/onnx__Conv.html#conv-11)不允许直接 BF16 Conv。导出常量折叠可能使 patch 权重成为 FP32 initializer，消费方按实际文件计费；不能重写 protobuf dtype 或去掉必要 Cast。原生 BF16 导出被官方 checker 拒绝的诊断应保留，与显式路径的成功分列。
