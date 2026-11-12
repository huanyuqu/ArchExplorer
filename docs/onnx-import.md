# 标准 ONNX 文件导入与探索

用户可以把真实 ONNX ModelProto 文件、完整运行输入和明确硬件配置交给 ArchExplorer，得到普通 WorkloadSpec，继续使用公共 Graph、compiler、Env/search、离线 Agent 和保存恢复。文件中的节点机械转换为 Graph 算子；文件名、模型名与 display name 不选择手写网络模板。

可选依赖为 ONNX 1.16.2；不安装该 extra 时，既有负载和工具仍可导入。安装入口为 `pip install 'arch_explorer[onnx]'`。支持标准空 domain、opset17、实际 IR8、无控制流的推理 DAG，真实 FLOAT/FLOAT16/BFLOAT16 存储与 FP32 计算。支持有界 MatMul/Gemm/Add/Mul/Div/Erf/Cast/LayerNormalization/Relu/Softmax/Transpose/Reshape/Identity/Constant。符号根必须按原名称显式给完整 shape；完整 ONNX、训练、量化、自定义域、稀疏、动态控制和非 literal shape 运算仍有明确缺口。[实际框架导出](framework-export.md)与 dtype 参考、组件、安装验收分别记账。

## 从文件得到公共负载

```python
import json
from pathlib import Path
from arch_explorer.archs.onnx_compilation import (
    onnx_workload, onnx_scene, onnx_search_config, onnx_agent_config,
)
from arch_explorer.workload.onnx_import import describe_onnx

root = Path('input-models/mlp')
spec = onnx_workload(
    root / 'model.onnx',
    inputs=json.loads((root / 'INPUTS.json').read_text()),
    input_shapes=json.loads((root / 'INPUT-SHAPES.json').read_text()),
    external_root=root,
    name='user-example',
)
identities = describe_onnx(spec)
adf, same_spec, mapping = onnx_scene(spec, candidate=0, layout='materialized')
config = onnx_search_config(spec)
agent = onnx_agent_config(spec)
```

`onnx_workload(model, *, inputs, input_shapes=None, external_root=None, name='onnx', resource_budget=None)` 直接调用正式 `workload.onnx_import.import_onnx`。inputs 是原 graph.input 名称到完整有限 JSON 数值的字典；没有默认根或权重。input_shapes 是根名称到完整正整数 shape 的字典，用来特化符号维，不是符号名到整数。初始化器若同时列在 graph.input，导入器拒绝，不能隐式覆盖或忽略用户值。CORE 不在权威预检前 deepcopy 巨大数值根。

文件入口 `onnx_workload_from_files(model, *, inputs_path, input_shapes_path=None, external_root=None, name='onnx', resource_budget=None)` 先调用公共 onnx_preflight 检查源和 shape/预算，再按 admitted 结果加载根 JSON；根文件本身超过 max_host_bytes 也在读取前拒绝。CLI import-onnx 与框架 import_export 共用这个入口。`--resource-budget BUDGET.json` 可选；完整正规化预算进入 WorkloadSpec/Graph/编译来源。省略预算保持历史四字段小包表示，不修改全局上限。

有外部 tensor 时必须给 external_root 作为允许根边界；external location 相对模型所在目录解析，解析后的文件必须位于该允许根内。offset/length、实际字节和 shape/dtype 必须一致。缺文件、越界、目录逃逸、symlink 越界和不支持的属性均由导入器在 prepare 前拒绝，不发网络请求。合法 ONNX 仍可能不在本产品支持域；官方 checker 成功不代替产品可执行性。

`kind='onnx'` 是自动注册的内置负载，配置可显式使用 `registration_modules=[]`；省略该字段沿用同一空列表默认值。无需额外 registrar 或 provider，导入后的编译与执行继续使用已有组件绑定。

返回的 `WorkloadSpec(kind='onnx')` 默认参数为 name/source/inputs/input_shapes，显式预算增加正规化 resource_budget。source 保存模型及外部文件完整字节和摘要，默认源预算16MiB；可在公共 max_host_bytes 约束内显式调整 max_source_bytes 等限制，不能只删除保护。原绝对路径不成为部署前提。保存后删原文件仍可读取该快照；重新导入修改过的文件产生新来源。有效重封后修改模型/外部字节、摘要、根、shape 或预算，仍须通过原完整来源与产品校验。

`onnx_preflight(path, *, input_shapes=None, external_root=None, resource_budget=None)` 不接根数值，不构公共图/计划或执行组件。原始 shapes、dtype 与 root/weight/temp/host 字节估计、admitted/violations 保留在结果中；lowering 新增临时量仍须经后续编译预算。默认 reference_mode=execute 包含 CPU 参考；显式 inputs 仅验证全部根和存储舍入，执行零数值算子，不能计为完整数值通过。FP16/BF16 的真实存储舍入、Cast 流量和 LayerNorm stash 顺序由公共 workload/runtime 定义。

`describe_onnx(spec)` 提供原 node/value 到 Graph 的稳定映射。空或重复 display name 不影响身份，Gemm/非末轴 Softmax 可对应多个 Graph 算子。logical scalar [] 以 Graph [1] 存储，原 rank 保留。int64 shape literal 只供编译期 Reshape 使用；FP32 initializer/Constant 作为真实权重存储和读取。只有 literal Constant 折叠，其余动态节点包括死节点均保留费用。

## 两放置、两布局与有界搜索

`onnx_scene(workload, *, candidate=0, layout='materialized', hardware=None)` 返回 ADF、原 spec 和一份 MappingSpec。candidate 为0/1；layout 为 materialized/views。可传已有硬件 factory，遵循 `(adf, workload, mapping)` 三元组约定；mapping 给出一个 compute 与 memory 的正式路径。默认合成 Events board 暴露两个 compute 和共享 SRAM，所有硬件参数可在序列化 ADF 中检查。

`onnx_search_config(workload, *, candidate=None, layout=None, hardware=None)` 默认选择两个放置/tile × 两布局。Graph 的公开 operator IDs 与注册规则决定 mapping，不从原 display name 猜身份。tile 候选为 (m2,k3,n4) 和 (m3,k2,n3)，layout_candidates 负责别名是否合法；不满足条件的 reshape 按正式规则物化回退。packing、广播、transpose 及真实存储读取分别计费，CORE 不复制其公式。

配置沿用通用 Env，seed为7、上界12评价单位。容量候选为1B/当前容量，带宽为当前值/两倍；搜索评估提案，容量编译失败同样计费，账本给出 phase/valid/execution_attempted/error。固定 candidate 或 layout 会缩小候选数，预算变为三倍候选数。`onnx_agent_config(workload, *, candidate=0, layout='materialized', hardware=None)` 返回离线 operator_mappings/evaluate/submit 脚本，一个评价单位、至多三个步骤和三个工具调用。

## console 与导出示例

`examples/onnx_import/models.py` 生成真实合成 MLP 与 attention、完整根值、INPUT-SHAPES.json 和摘要，无预计算中间量。`--external-data` 写出真实 MLP tensor 文件；`--symbolic-batch` 为符号根写出完整特化；`--variant 0..1000` 只改变显式合成根/初始值，不选择模型实现。

```sh
python models.py --output input-models --external-data --symbolic-batch
archexplorer import-onnx input-models/mlp/model.onnx \
  --inputs input-models/mlp/INPUTS.json \
  --input-shapes input-models/mlp/INPUT-SHAPES.json \
  --external-root input-models/mlp --output WORKLOAD.json
python emit.py input-models/mlp/model.onnx \
  --inputs input-models/mlp/INPUTS.json \
  --input-shapes input-models/mlp/INPUT-SHAPES.json \
  --external-root input-models/mlp --output run-inputs
archexplorer compile run-inputs/SCENE.json --output COMPILED.json
archexplorer prepare COMPILED.json --output prepared
archexplorer evaluate prepared/COMPILATION-REQUEST.json --output RESULT.json
archexplorer search --config run-inputs/CONFIG.json --policy greedy --output BEST.json
archexplorer agent run --config run-inputs/AGENT-CONFIG.json --policy scripted --max-steps 3 --output AGENT.json
archexplorer evaluate BEST.json --output READONLY.json
archexplorer rerun BEST.json --work-dir rerun --output RERUN.json
```

import-onnx 的 model、--inputs 和 --output 必需；--input-shapes/--external-root 可省，--name 默认 onnx。emit.py 参数相同，一次导入后输出 WORKLOAD.json、SCENE.json（adf/workload/mapping）、CONFIG.json、AGENT-CONFIG.json、SCOPE.json。attention 使用其自己的 model.onnx 和 INPUTS.json，其他用户文件使用同一流程。

## 保存与恢复

探索保存采用 `archexplorer.compilation-search-result.v1`，胜出行保存实际有效 mapping、compiled_result、evaluation 和 diagnostics；原请求另留 evaluation_attempts[].requested_mapping。编译来源中的完整模型/外部数据、根和 input_shapes、完整实例规则、adapter/component 绑定及 options 一同参与保存身份。

CLI evaluate BEST 与 `exploration.replay(saved)` 严格复核来源和内容，允许纯 compile/计划重建，禁止实际 prepare/engine/solver/写入或网络。CLI 按 --output 写最终 JSON 是用户显式输出，单独记录。Env 的保存结果只读工具为 `evaluate_request({'payload': saved})`，不消耗评价单位。显式 rerun 使用新目录并实际计费执行。

本功能提供有界软件前端，合成 Events 服务没有硬件时延标定，IPU required-incomplete 与论文规模缺口保持。官方版本规范及序列化入口见 [ONNX 1.16.2 源码](https://github.com/onnx/onnx/tree/v1.16.2)、[外部数据规范](https://onnx.ai/onnx/repo-docs/ExternalData.html)。
