# 外部算子：语义、目标实现与图映射

先读用户的 ADF、图和输入，再选择目标能力及执行实现。新算子的数学规则在用户包中，
执行实现负责将它转为某个目标支持的产物；mapping 选择实现、放置、搬运和存储，组件与
simulator 负责真实机制和调度。新增算子不需要给 GPU provider、调度器或 CLI 增加名字分支。

本文的[独立包](../examples/extensions/operator_implementations/README.md)定义主库未知的
`example_shifted_square`：FP32 `y=(x+shift)*(x+shift)`。两条分支的 shift 分别为 1.25 和 -0.5，
add 汇合后再以 shift=0.25 计算最终输出。图名是普通用户名字，不选择某个内置模型。
四分之一整数的合成输入允许用独立有理数运算生成精确预期，不依赖设备的时间模型。

## 安装与最小事件流程

需要同时提供公开 `ImplementationDefinition`、`ExecutionProgram` 和 `execution_graph` 的主包。
旧的单 lower 接口本身不提供此能力。先安装匹配主包，再单独安装外包：

```sh
python -m pip install /ABS/FRAMEWORK/arch_explorer-0.2.0-py3-none-any.whl
python -m pip install ./examples/extensions/operator_implementations
cd /tmp
python -m ae_operator_implementations generate --output square-inputs
python -m ae_operator_implementations check-reference --graph square-inputs/graph.json --output numerical.json
archexplorer demo prepare --specs square-inputs/events-scene.json --output square-events
archexplorer demo run square-events
archexplorer evaluate square-events/mapping-000/result.json --output square-replay.json
```

框架先构建，外包后安装；主库、provider 和调度器无需因该算子变化。入口点为
`arch_explorer.components` 的 `ae_operator_implementations:register`，只注册，不执行。
全程不设置 PYTHONPATH、不插入源码目录、不做 monkeypatch。生成的输入是普通 JSON，可编辑后重新准备。

`adf.json` 使用已有 vector_unit、SRAM、DRAM、bus。所有参数带单位及合成来源，没有自建 provider。
`graph.json` 是有分支/汇合的严格拓扑图，`inputs.json` 与 `expected.json` 是小数组及独立数值预期。
`mappings.json` 含两个显式实现选择；`events-scene.json` 是给通用 demo 的完整场景。
改变 ADF 后先重新读取，再准备请求；事件性能结果本身不计算张量值。

## 语义与实现分别注册

```python
from arch_explorer.workload.api import register_operator
from ae_operator_implementations import SEMANTIC

register_operator(SEMANTIC)  # 和自动发现复用同一个稳定对象，身份幂等
```

外包的模块级对象通过 `OperatorDefinition(OPERATOR, validate, version='1', reference=reference)` 构造。
语义检查输入/输出形状、FP32 dtype 和有限 shift。reference 接收按输入顺序排列的只读 ndarray，
返回输出 ndarray；它不调度、不选择组件。数值参考正确、时序/资源正确、native 软件一致性是
分别证明的事情，不能互相替代。

执行实现拥有稳定 ID/version、目标、dtype、要求及 supports。此例的两个 events 实现：

- `example.shifted_square.single`：一个 vector Kernel，每元素两次 SIMD 运算；不是将两次运算伪装成一次 FMA。
- `example.shifted_square.steps`：shift 与 square 两个有依赖的步骤，声明一个内部临时 Tensor。

二者都要求现有 `kernel:vector` 能力，支持 FP32 materialized 布局。真实组件仍须接受实际 Kernel，
能力 token 不代替形状/机制校验。内部临时名字由父映射按节点隔离，分支不会共用同一缓冲；
临时容量、最后消费者和释放走正常 storage 计划，不在用户包复制设备状态机或补一个固定延迟。

旧 `OperatorDefinition(name, validate, lower)` 仍可兼容为 events 的 `legacy:<op>` 实现。
新包把语义和目标实现分开；相同语义可有多个实现，注册顺序不能决定结果。

## 公开策略选择与错误诊断

```python
from arch_explorer.adf import MappingSpec

mapping = MappingSpec('execution_graph', {
    'implementations': {
        'left_square': 'example.shifted_square.steps',
        'right_square': 'example.shifted_square.steps',
        'final_square': 'example.shifted_square.steps',
    },
    'placement': 'round_robin',
    'serialize': False,
    'memory': 'operator_board/shared',
})
```

`implementations` 按真实图节点选择，不是算子名白名单。两个兼容实现存在时，省略选择应拒绝
歧义；显式 ID 不存在、目标不符、缺 capability 或 dtype/shape/layout 不支持也应拒绝。
诊断应保留节点、实现及目标；修正输入或安装正确实现后，在新目录重新准备。

`placement` 可选 first/round_robin；`serialize=False` 保留 DAG 并行，True 是显式串行策略。
`memory` 可选，默认该设备 DRAM。本例两种实现使用相同 round_robin/非串行/SRAM，
因此差异来自实际步骤、临时量和存储/传输，而不是改报告标签。具体时间与峰值以实际计划为准。
特殊策略仍可扩展，但普通单设备图不用重写 run_device。

## GPU、来源和边界

GPU 同样选择 target=`accelsim` 的公开目标实现，组件 provider 消费可执行程序和来源。
外部 lower 从 `context.options['emitter']` 取得本次求值的 ProgramEmitter；它不进入请求 JSON。
本包先 load 输入、constant shift，再执行 FADD/FMUL 和 finish。多步版本通过
`temporary('shifted', shape)`、两个 kernel 以及真实 store/load 连接；不是伪 kernel 或估算时间。
reference 仍为同一普通语义回调。示例用 R8/R9，避开发射器的索引和地址 scratch 寄存器。

`graph.meta['inputs']` 明确提供所有外部输入/权重的有限数值数组，没有隐式随机填充。
公开 `workload.reference_api.execute_reference` 执行完整图；check-reference 与独立有理数
预期核对两条分支、汇合和输出。改写图或输入后需提供匹配参考，不能继续套用该样例 oracle。

```sh
python -m ae_operator_implementations gpu-case --graph square-inputs/graph.json --engine /ABS/ENGINE/engine.json --output square-gpu.json
python /ABS/SOURCE/examples/gpu_instantiation/workflow.py run square-gpu.json --output square-native
python /ABS/SOURCE/examples/gpu_instantiation/workflow.py replay square-native --output square-native-replay.json
```

这里 `/ABS/SOURCE` 是匹配的公开 source-export/sdist，workflow 复用原有 CORE 组件 GPU 比较入口。
两个 mapping 分别是单核/32线程与多核临时量/64线程，两个因素都改变，不做单因素性能归因。
输入图可以任意命名；新增语义与两个目标的实现都留在外包，不改 kind 假装 ViT/CLIP，
不写第二个 provider 绕过。已有 ViT/CLIP 通过同一公开机制使用自己的图/参考适配。

请求与结果应绑定实际语义 validate/reference、执行 lower/support/version、mapping 和输入字节。
安装包源码改变后，旧结果只读重放必须拒绝，不能默默选新的插件。重复注册和失败插件须遵循
discovery 的冲突/回滚规则。准备失败不应留下 provider 运行输出；缺失来源不能补零或忽略。

native 组件/oracle 对照需要独立进程和输出，前台逐 kernel 与物理 final/尾部分别检查。
新 component-engine 主延迟是物理完成；保留 foreground_final/drain_ticks/drain_core_cycles，
不把尾部当额外 kernel。CPU native 预算含 finish，单 run 至多 600 秒/8 GiB；不采集 GPU。
合成参考生成 trace 不是框架 CUDA 实采、预训练权重或硬件精度验证。
