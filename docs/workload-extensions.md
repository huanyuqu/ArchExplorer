# 从 ADF 文件运行自己的算子图

需要同一语义的多个目标实现、显式选择或多步/临时张量时，先读
[外部执行实现教程](operator-implementations.md)与
[独立实现包](../examples/extensions/operator_implementations/README.md)。它使用公共 execution_graph
处理普通图的放置和存储；下面保留单 lower 与自定义 row mapping 的兼容示例。

先准备一个 ADF 文件，再把它读成内存描述，最后由框架生成可执行模型。
下面假定文件已经存在；首次使用时先完成下一节的安装与 generate，再执行这段读取代码：

```python
import json
from pathlib import Path
from arch_explorer.adf import from_dict

adf_data = json.loads(Path("branch-inputs/adf.json").read_text())
adf = from_dict(adf_data)
print(adf.to_dict())
```

这里 `adf.json` 是硬件描述文件，`adf` 是包含组件类型、参数和来源的内存对象；二者都不是
运行中的模拟器。注册组件的 build 将 ADF 实例化为资源，mapping 将算子图变成任务，
`events` 执行器才调度这些任务。改变文件后须重新读取并准备请求，不能认为已准备的模型会自动更新。

## 安装真实外部扩展

本教程的[独立包](../examples/extensions/workload_components/README.md)包含新算子、图构造器、
合成阵列和外部 mapping。它不实现 provider，不修改主包的算子/模型名单、CLI 或 scheduler。
先安装含 `arch_explorer.workload.api` 的主包；从源码分发或 checkout 根目录执行：

```sh
python -m pip install .
python -m pip install ./examples/extensions/workload_components
cd /tmp
python -m ae_workload_example generate --output branch-inputs
```

此后所有命令可在仓库外运行。没有修改搜索路径，也不需要 PyTorch、ONNX、权重或 GPU。
自动发现使用 `arch_explorer.components` 入口点，目标 `ae_workload_example:register`。
需要嵌入 Python 时也可显式调用同一个 register；它只注册，不执行仿真。

生成的 `adf.json` 描述两个 `tutorial_array`、一个共享 SRAM 和 bus；所有参数带单位及合成来源。
`affine-graph.json`、`builtin-graph.json` 是两个独立图；`mappings.json` 保存两个完整映射选择。
这些是可独立编辑的普通输入文件，不是绑定某个设备系列的 preset。

## 先接一个只复用已有算子的新图

`builtin-graph.json` 有两条 scale 分支和一个 add 汇合。新图的入口是
`WorkloadSpec('graph', graph.to_dict())`，无需给主项目增加模型 kind。
`workloads` 列表里的独立场景加权不等于这个图的依赖关系。

```python
from arch_explorer.adf import WorkloadSpec
from arch_explorer.workload.api import Workload

graph = Workload.from_dict(json.loads(Path("branch-inputs/builtin-graph.json").read_text()))
spec = WorkloadSpec("graph", graph.to_dict())
```

图是有限 JSON DAG，明确张量、算子、输入/输出及参数；不是可序列化的任意 Python 对象。
循环、悬空张量、多个生产者、损坏 shape/dtype 和非有限数应在执行前拒绝。
输入从所选内存初始驻留，不代表模拟了输入数据的外部采集。
图严格要求拓扑顺序，不自动排序；前向引用和循环会拒绝。本例始终按拓扑顺序写图。

## 注册真正的新算子

公开作者入口为 `arch_explorer.workload.api`，保留同一份 Tensor、Operator 和 Workload 类型。
`OperatorDefinition(name, validate, lower)` 是单 lower 兼容形式，适配为 events 的 `legacy:<op>`。
新接口可分别注册语义 validate/reference/version 与 `ImplementationDefinition` 的目标 lower/support。
`register_operator(definition)` 注册同一个定义对象，竞争名称必须报错；不在组件里分派模型名称。

示例的新操作是：

```text
y = scale * x + bias
activation = identity 或 relu
```

`scale=1.25`、`bias=-0.5` 和字符串 activation 都是合法属性。验证器检查输入/输出同为 FP32
`[rows, cols]`，并验证有限系数、属性集合和形状。它没有放宽其他内置算子的语义。
lower 返回一个现有的 vector Kernel：每元素独立乘法和加法各一次，ReLU 再加一次 SIMD 指令。
这是当前合成实现选择，不是把任意新算子伪装成 GEMM；不支持的 dtype/tile 明确拒绝。

外部定义只需两个函数和一个稳定注册对象：

```python
from arch_explorer.workload.api import OperatorDefinition, register_operator
from ae_workload_example.semantics import validate_affine, lower_affine

# 实际包复用其模块级 AFFINE 对象，避免每次创建竞争定义。
definition = OperatorDefinition("my_affine", validate_affine, lower_affine)
register_operator(definition)
```

普通多步执行可返回公开 `ExecutionProgram` 与内部 Tensor，经公共图 mapping 处理依赖和存活；
特殊放置策略或新硬件机制再扩展 mapping/组件。目标产物表达不了的语义必须拒绝，
不能默认零时延。这里是形状和资源需求仿真，不计算张量数值，也不证明模型数值正确性。
可选 `register_workload(kind, builder)` 仅便利构造；builder 的签名是
`builder(params, *, tp=1)`。示例 `tutorial_branches` 明确拒绝 tp>1，不静默忽略它。

## 自定义 mapping 和硬件

`tutorial_rows` 通过公开 MappingTemplate 和 MappingContext 构建任务，不调用私有调度器。
它按行切分每个算子，在两个阵列间轮流放置；两个候选的 tile_rows 为 4 和 16。
每份输出有独立存储分配，消费者依赖实际生产任务；两条分支没有被冒充独立加权场景。
输入/输出流量使用 SRAM/bus 资源，容量与存活由现有执行器检查；mapping 的公开 plan
强制容量检查，终端输出保持至 return。

合成阵列每 tile 的周期数为 `launch_cycles + ceil(SIMD instructions / lanes)`，再按
clock_ghz 转换到 ns。组件只理解 kernel、精度和资源，不认识 affine 图或用户模型名。
改变 lanes 或 clock 应影响其资源需求；tile 粒度会改变真实任务数和启动开销。

```sh
python -m ae_workload_example hardware --lanes 8 --clock-ghz 2 --output faster-adf.json
```

该 mapping 只支持两个指定阵列、共享 SRAM/bus、等形状二维 FP32 激活、维度 1–4096、
一次迭代及无 SFU 的 SIMD lowering；不支持别名、矩阵 kernel、其他 dtype、多设备或 TP。
更换硬件时先检查 mapping 适用性。源码中的合成公式不是物理校准；未知成本仍为 null。

## 从文件准备、执行、比较和重放

```sh
python -m ae_workload_example scene --adf branch-inputs/adf.json \
  --graph branch-inputs/affine-graph.json --mappings branch-inputs/mappings.json \
  --output branch-scene.json
archexplorer demo prepare --specs branch-scene.json --output branch-run
archexplorer demo run branch-run
archexplorer evaluate branch-run/mapping-000/result.json --output branch-replay.json
```

使用新输出目录。scene 命令读取并校验传入文件，选 `provider='events'`，不增加新的 CLI 模型分支。
将 graph 路径换成 builtin-graph.json，可用同一硬件与 mapping 跑已有算子的新 DAG。
将 ADF 路径换成 faster-adf.json，可独立比较硬件变化；不要同时改图后再归因到硬件。

同一条 API 路径也接受上述内存描述：

```python
from arch_explorer.adf import MappingSpec
from arch_explorer.simulator.api import prepare_request, assess_request, evaluate_request

mapping = MappingSpec("tutorial_rows", {"tile_rows": 4})
request = prepare_request(adf, spec, mapping, provider="events", output_dir="api-run",
                          options={"trace": True})
assessment = assess_request(request)
if not assessment["supported"]:
    raise ValueError(assessment["gaps"])
result = evaluate_request(request)
replayed = evaluate_request(result)  # 只读消费原执行收据，不再模拟
assert replayed["evaluation"] == result["evaluation"]
```

prepare 只生成资源与任务计划；只有 evaluate/run 成功返回 Evaluation 才是本次执行结果。
比较 `details.tutorial_rows.placements`、`details.stats.tasks`、`memory_peak`、breakdown、
latency_ns 和参数 provenance。墙钟、预检通过或文件存在均不替代模拟结果。
保留请求引用的原路径、原始执行收据和确切安装包；源代码或输入改变后需新 prepare。
源绑定失败不可通过更新哈希绕过，更不能重新模拟来冒充只读重放。

需要让 agent 操作此流程时使用[用户 skills](agent-skills.md)；硬件组件作者的基础说明见
[组件扩展教程](component-extensions.md)。
