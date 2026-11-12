# 标准 ONNX 文件示例

`models.py` 用官方 ONNX helpers 生成小型 MLP 和 attention。生成器只提供模型、完整运行根、尺寸特化及文件摘要，既不导入 ArchExplorer 解析器，也不计算激活答案。模型为标准 domain、opset 17、IR 8、FP32；两图均有多个输出，MLP 共享初始权重，attention 使用批量矩阵乘、非默认 transpose、缩放、有限 mask、softmax、int64 reshape 常量和 identity。

安装可选依赖后，在仓外使用导出示例：

```sh
python models.py --output input-models --external-data
python emit.py input-models/mlp/model.onnx \
  --inputs input-models/mlp/INPUTS.json \
  --external-root input-models/mlp --output mlp-run-inputs
```

`--symbolic-batch` 为每个模型生成明确的 `INPUT-SHAPES.json`；消费时加 `--input-shapes input-models/mlp/INPUT-SHAPES.json`。attention 的模型和根位于 `input-models/attention/`。用户自有模型同样通过 model、完整 roots 和允许的 external-root 导入，消费入口不按模型名或文件名选择模板。

`emit.py` 导入一次，并把同一个普通 WorkloadSpec 写进 `WORKLOAD.json`、`SCENE.json`、`CONFIG.json` 和 `AGENT-CONFIG.json`。随后使用既有通用命令：

```sh
archexplorer compile mlp-run-inputs/SCENE.json --output compilation.json
archexplorer prepare compilation.json --output prepared-request
archexplorer evaluate prepared-request/COMPILATION-REQUEST.json --output result.json
archexplorer search --config mlp-run-inputs/CONFIG.json --policy greedy --output best.json
archexplorer agent run --config mlp-run-inputs/AGENT-CONFIG.json \
  --policy scripted --max-steps 3 --output agent-best.json
archexplorer evaluate best.json --output readonly-best.json
archexplorer rerun best.json --work-dir rerun --output rerun-best.json
```

搜索的两个放置/tile 与两布局共用 12 单位预算及明确的容量/带宽候选；离线 Agent 花费一个评价单位。示例硬件参数均为可见的合成配置，不表示测量时延精度。

纯读取和显式 rerun 分开：前者核来源并允许纯 compile/计划重建，后者在新目录实际执行并计费。原模型、external tensor、根值、尺寸特化和导入器身份由正式来源协议保存，详细行为见 `docs/onnx-import.md`。支持范围及拒绝域以对应候选的正式导入合同为准。
