# 从一个组件到可安装扩展

`simulator` 是通用仿真生成框架：根据 ADF 的实例与连接、负载和 mapping 生成执行计划，
处理资源、依赖、调度和生命周期。具体硬件的参数、机制、状态机和外部引擎适配由
`components` 拥有。Accel-Sim 保留原版 oracle 与组件实例化两种路径；新路径由通用 Runtime
连接 SM/cluster、ICNT、L2、DRAM。不能把各组件各自计时后相加。框架通过注册能力选择执行器。

如果新增算子只需要已有硬件能力，使用[外部目标实现](operator-implementations.md)，分别注册
workload 语义/reference 与目标 lower/support；不用改组件或新增 provider。
只有确实需要新硬件机制时才进入下面的组件作者流程。

普通组件可以只实现 `ComponentType.build` 和 `Demand`，无需实现外部引擎协议。
`arch_explorer.components.api` 提供组件作者需要的类型，重导出保留原类型身份：
`ComponentType`、`ParamSpec`、`Conventions`、`BuildContext`、`PointModel`、`ComputeModel`、
`MemoryModel`、`NetworkModel`、`Demand`、`Kernel`、`Provenance`、资源/任务类型，及
`ExecutionModel`、`ExecutionRequest`、`ExecutionEvent`、`DemandAdapter`。
扩展不需要导入 `simulator.events` 内部文件。

## 先声明参数与能力

[独立扩展包](../examples/extensions/synthetic_components/README.md) 中的 `TutorialMac`
声明 `macs_per_ns`，单位 `MAC/ns`，以及只接受 `fp32` 的 `precision`。
它的构建函数返回 `ComputeModel`；`kernel_time` 只接受 FP32 GEMM/MVM，按 MAC 数除以
合成速率返回 ns。错误单位、布尔速率、越界值和不支持的 kernel 都应拒绝。
参数 `source` 说明这是合成假设。面积和成本未建模，保持未知。

```python
from arch_explorer.components.api import register_component
from ae_synthetic.component import TutorialMac
register_component(TutorialMac)
```

组件声明其能力，mapping 声明放置、切分和适用条件。能构建一个组件不表示任意 mapping
都能执行它。示例将这个 MAC 替换进带显式 SRAM/DRAM/网络的单核 DMC 场景，再使用已有
`dmc_partition` 映射运行一个 FP32 matmul。示例包的 `scenes.event_objects()` 返回标准
ADF、WorkloadSpec 和 MappingSpec，可直接传入 `adf.evaluate`。

## 实际安装与发现

在主项目的源码分发或 checkout 根目录执行：

```sh
python -m pip install .
python -m pip install ./examples/extensions/synthetic_components
cd /tmp
python -m ae_synthetic events --output installed-component.json
```

扩展有自己的 `pyproject.toml` 和 `src/ae_synthetic`，不加入主项目包发现名单。
它的入口点配置是：

```toml
[project.entry-points."arch_explorer.components"]
tutorial = "ae_synthetic:register"
```

无参数 `register()` 只注册描述和工厂，不执行仿真、探测设备或加载外部二进制。
同一个函数也可在 Python 中显式调用。框架在执行前发现已安装扩展：

```python
from arch_explorer.components.api import discover_components, providers
report = discover_components()
print(report["loaded"], report["failures"])
print(providers())
```

每个失败扩展单独报告，不让无关组件失效；重复名称、provider id 或不兼容版本必须明确
报错。给自己的组件和 provider 使用唯一名称。排错时保留 discovery 的失败原因。

## 需要独立引擎时注册 provider

外部引擎或必须整体执行的组件组使用 `register_provider(provider)`。对象声明稳定的
`id`、精确 `version`、`component_types`、`execution_domain`（`single` 或 `joint`）、
`capabilities`、精确的 `request_schemas` / `result_schemas`，并实现：

```python
prepare_request(adf, workload, mapping, *, output_dir, options=None)  # 不执行仿真
assess_request(payload)  # supported: bool, gaps: list
evaluate_request(payload)  # 完整 request 与 Evaluation.to_dict()
```

prepare 的 payload 回显标准 `adf`、`workload`、`mapping`、`options` 字典，其他字段归
provider 管理。结果保存完整组件请求和执行产物绑定，重放时检查来源和结果，不改写原件。
外部引擎路径及限制放在 options；硬件参数仍来自 ADF，语义来自 WorkloadSpec。

示例的 `tutorial.serial_mac` 是独立合成执行域：只接受一个 `tutorial_mac`、维度
1–64 的 FP32 matmul、`tutorial_serial` 的 `tile_m=1|2`，以及 `startup_ns`。
该 mapping 只供此 provider 使用，不可套用事件引擎。混合组件域、BF16、其他负载和
未知参数会拒绝。它不模拟内存、数值输出、功耗或真实硬件。

```sh
python -m ae_synthetic scene --output /tmp/mac-scene.json
archexplorer demo prepare --specs /tmp/mac-scene.json --output /tmp/mac-run
archexplorer demo run /tmp/mac-run
archexplorer evaluate /tmp/mac-run/mapping-000/result.json --output /tmp/mac-replay.json
```

这些步骤通过安装入口点发现 provider，没有修改主项目源码或搜索路径。
示例 128 MAC / 8 MAC/ns，加上每 tile 5 ns，两个候选分别为 36 ns 和 26 ns；
这是公开的合成机制示例。改变 mapping 会改变真实的 tile 列表，未建模的指标仍为 null。

## 通用生成和重放入口

```python
from pathlib import Path
from ae_synthetic.scenes import provider_scene
from arch_explorer.adf import MappingSpec, WorkloadSpec, from_dict
from arch_explorer.simulator.api import prepare_request, assess_request, evaluate_request

scene = provider_scene()
request = prepare_request(from_dict(scene["adf"]), WorkloadSpec(**scene["workload"]),
                          MappingSpec(**scene["mappings"][0]), output_dir="/tmp/mac-python",
                          provider=scene["provider"], options=scene["options"])
assessment = assess_request(request)
if not assessment["supported"]:
    raise ValueError(assessment["gaps"])
result = evaluate_request(request)
replayed = evaluate_request(result)  # 检查已保存执行，不重跑 provider 引擎
assert replayed["evaluation"] == result["evaluation"]
```

请求外层为 `arch-explorer.simulator-request.v1`，包含 provider 的 id/version、三个标准
spec、options、provider payload 和完整绑定。结果外层为 `arch-explorer.simulator-result.v1`，
包含完整 request、同一 provider、原有 Evaluation 字典、provider 原始结果和绑定。
细节原样保留；`latency_ns` 仍是 ns，不用 Python 墙钟替换。保留请求引用的文件与原路径。
新结果和已注册的历史精确 schema 都交同一 resolver，CLI 不识别具体引擎名字。

demo 的候选集合在任何 evaluate 前完整检查：标准对象、候选标签、路径、全部请求身份和
准备记录必须一致；先复核所有能力，再执行。重复、替换、缺失或错配的请求不能形成成功
比较。更改任何规格或 options 后应 prepare 到新目录。

## 有状态组件和执行域

有状态机制使用公开的 `submit / next_event_time / advance`，保持反压拒绝时无副作用，
分别报告 `consumer_ready`、`physical_complete` 和 `resource_ready`。不要把消费者可读误作
物理写回完成，也不要绕过存活容量或资源占用。具体示例见[联合执行合同](unified-execution.md)。

执行域不能任意拼接。Accel-Sim provider 的子组件需要同一次原生执行；不支持与其他
独立引擎耦合时应在执行前拒绝。其组件实现位于 `components.accelsim`，通用 simulator
不根据模板或 provider 名字选择专用机制。用户场景构造、CLI 模型名字和权重选择也不属于
provider。完整 GPU 示例见[ViT/CLIP 教程](tutorials/05-full-models.md)。
