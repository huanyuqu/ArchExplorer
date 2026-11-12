# 5. 新负载 → GPU 组件 → mapping → 原生程序仿真

公共编译、Events/Roofline、布局探索和保存恢复的完整小型模型入口见
[ViT 公共编译教程](../vit-compilation.md) 与 [CLIP 公共双塔教程](../clip-compilation.md)。
本页继续使用原生程序 provider 和外部引擎；已有原生结果保留原身份，
不能把新的公共模型配置或 CPU/组件结果改称旧原生程序的新验证。

这个 demo 用明确缩小的完整 ViT 分类前向和完整 CLIP 图文双塔前向，展示如何接入负载、
选择 mapping、生成原生程序并比较模拟结果。ViT 的输出是分类 logits；CLIP 包含视觉塔、
文本塔、投影、归一化和相似度输出。只执行一个 attention 或一个整数 kernel 不构成这两个 demo。

快速配置的尺寸、输入、数值参考及生成方式由 `program_demo_specs()` 的 `scope` 和组件产物报告。
小配置不代表 ViT-B/16 或 CLIP ViT-B/32；标准模型语义保存在
`examples/workloads/vit.json` 和 `clip.json`，其原生执行能力需单独检查。
真实采集、公开 trace 与参考执行生成的合成 trace 必须按各自来源解释。
合成参考执行不证明官方框架编译一致性或物理 GPU 时延精度。

## 准备安装和外部引擎

安装 ArchExplorer 后，先按 [引擎说明](../accelsim.md) 准备固定版本的 Accel-Sim/GPGPU-Sim
二进制、仿真库及引擎 manifest。manifest 引用外部文件，不能用 CUDA runtime 替代仿真库。
原生模拟在 CPU 上运行；准备和执行不会自动采集真实 GPU，也不下载模型权重。

本流程使用 `arch_explorer.simulator.api` 和显式选择的 `accelsim` provider。
组件包拥有原生机制；场景由 `archs.accelsim.program_demo_specs` 构造。只带单 kernel
能力的构建会明确报缺失；没有引擎或程序能力时不会切换到预测公式。
后端对算子、ISA、输入、完整性或资源报告 unsupported 时，保留报告并停止，不能把它计为完成。

## 从一个新目录运行

使用尚不存在的输出目录。下面的 `/absolute/path/engine.json` 要替换为实际 manifest：

```sh
archexplorer demo prepare --model vit --engine /absolute/path/engine.json --output runs/vit
cat runs/vit/specs.json
cat runs/vit/mapping-000/assessment.json
cat runs/vit/mapping-001/assessment.json
archexplorer demo run runs/vit
cat runs/vit/comparison.json

archexplorer demo prepare --model clip --engine /absolute/path/engine.json --output runs/clip
archexplorer demo run runs/clip
cat runs/clip/comparison.json
```

`prepare` 只生成请求和预检。`prepared` 表示可以尝试执行，尚无成功 Evaluation；
`unsupported` 表示逐项缺口。`run` 会重新预检所有请求，调用原生后端，只有每个候选都返回有效
Evaluation 才生成 `comparison.json`。失败写入 `run.json`，不生成成功的整体比较。
输出目录和原生文件路径均保存在本次请求中，重放时保留目录与外部引擎文件。

| 文件 | 用途 |
| --- | --- |
| `specs.json` | ADF、完整 WorkloadSpec、两个 MappingSpec 和范围/来源声明 |
| `preparation.json` | prepare 时的标准对象、候选声明与完整公开 request 的规范化身份关联 |
| `adf.json`、`workload.json`、`mappings.json` | 分别查看组件参数、负载、mapping |
| `mapping-NNN/request.json` | 公开 program request，绑定负载、组件和实际程序 |
| `mapping-NNN/assessment.json` | 完整程序能力与缺口检查，不含替代时延 |
| `mapping-NNN/program/` | 后端拥有的程序、参考、trace 等产物，以请求描述为准 |
| `mapping-NNN/result.json` | 原生结果与完整 request，可直接重放 |
| `comparison.json` | 每个候选的完整 Evaluation：ns、native cycles/计数、来源和范围在后端 details 中 |

比较时同时检查原生周期、时钟来源、实际 kernel/指令覆盖和参考输出。
两个 mapping 需要在参考上语义等价，并改变真实 program、切分或执行几何；标签不同不够。
run 会先核对整个候选集合、准备记录、公开 specs 和每份完整 request，再检查支持能力；
重复引用（含路径别名）、复制同一请求、缺失/额外/改序候选、标签改写或准备后替换请求都会失败，
不会先执行合法的那部分。比较中的 mapping 来自已核对的准备关联。
修改形状、ADF、mapping 或请求时应重新 prepare 到新目录；旧目录缺少准备记录也需重新生成。
本地关联记录和哈希用于核对该次 prepare，不认证整个目录被任意重写后的语义，也不替代后端程序绑定。
第二种 mapping 不保证更快。未知成本仍为 null，主机 Python 墙钟不作为 GPU 推理时延。
一个完整顺序程序由同一次原生执行处理跨 kernel/cache 状态，不能逐 kernel 冷启动后相加。

## 用 Python 查看对象、结构和组件映射

```python
from dataclasses import asdict
from arch_explorer.adf import WorkloadSpec, MappingSpec, from_dict
from arch_explorer.archs.accelsim import program_demo_specs
from arch_explorer.simulator.api import prepare_request, assess_request, evaluate_request
from arch_explorer.workload.full_models import full_model_workload

specs = program_demo_specs("vit")
print(specs["scope"])
workload = WorkloadSpec(**specs["workload"])
adf = from_dict(specs["adf"])
mappings = [MappingSpec(**value) for value in specs["mappings"]]
graph = full_model_workload(workload.kind, workload.params)
print(asdict(graph))  # 所有 tensors、operators、依赖及输出元数据
print(adf.to_dict())  # 公共组件、参数和来源；原生字段映射由组件 provider 检查

for index, mapping in enumerate(mappings):
    request = prepare_request(adf, workload, mapping,
                              output_dir=f"runs/python-vit-{index}",
                              provider=specs["provider"],
                              options={"engine": "/absolute/path/engine.json"})
    assessment = assess_request(request)
    print(mapping.to_dict(), assessment)
    if not assessment["supported"]:
        raise ValueError(assessment)
    result = evaluate_request(request)
    print(result["evaluation"])  # 统一 Evaluation.to_dict()，含原始后端详情
```

公开 demo runner 同样可从 Python 使用：

```python
from arch_explorer.demo import prepare_demo, run_demo
prepare_demo("runs/python-demo", model="clip", engine="/absolute/path/engine.json")
comparison = run_demo("runs/python-demo")
```

`python -m arch_explorer.demo prepare ...` 与 `archexplorer demo prepare ...` 相同，安装 wheel
后可以在任意工作目录调用，不依赖仓库内工具脚本或开发者 Workspace。

## 改一个形状，保持 GPU 组件不变

将同一小模型的 batch 改为 2，重新生成请求，而非编辑生成后的哈希或 geometry：

```python
import json
from arch_explorer.archs.accelsim import program_demo_specs
specs = program_demo_specs("vit")
specs["workload"]["params"]["batch"] = 2
with open("batch-two.json", "w") as file:
    json.dump(specs, file, indent=2)
```

```sh
archexplorer demo prepare --specs batch-two.json --engine /absolute/path/engine.json --output runs/vit-batch-two
archexplorer demo run runs/vit-batch-two
```

新的输入和数值参考由 workload/外部语义拥有，目标执行实现重新产生并绑定程序。
后端不支持新形状时明确拒绝；这条拒绝不算新形状仿真成功。
扩展同类负载只改外部语义/图及目标执行实现；GPU provider、CLI、runner 和物理组件不增加模型名字分支。
新增算子参照[外部执行实现教程](../operator-implementations.md)，不是给 provider 再增加一个模型 kind。
若需要新的算子或指令，分别在拥有该语义的模块声明支持，未支持的指令不能忽略或给零时延。

## 改组件参数，保持负载不变

例如将 SM core 时钟改为合法的 1500 MHz，并明确标记为对照场景：

```python
import json
from arch_explorer.archs.accelsim import program_demo_specs
from arch_explorer.adf import Param, from_dict
specs = program_demo_specs("vit")
adf = from_dict(specs["adf"])
adf.child("cluster").child("sm").params["clock_mhz"] = Param(
    1500, "MHz", "demo sensitivity scenario: 1500 MHz; not a measured device")
specs["adf"] = adf.to_dict()
print(adf.to_dict())
with open("changed-clock.json", "w") as file:
    json.dump(specs, file, indent=2)
```

```sh
archexplorer demo prepare --specs changed-clock.json --engine /absolute/path/engine.json --output runs/vit-clock
archexplorer demo run runs/vit-clock
```

原生配置编译器负责检查参数合法性和写入字段。查看新结果中的 component map、配置来源和
原生计数，不能在 Python 中缩放旧 cycles，也不承诺时延随一个参数单调变化。

## 独立重放和失败

```sh
archexplorer evaluate runs/vit/mapping-000/request.json --output runs/vit-first.json
archexplorer evaluate runs/vit/mapping-000/result.json --output runs/vit-replayed.json
```

新外层 schema 为 `arch-explorer.simulator-request.v1` / `arch-explorer.simulator-result.v1`，
绑定 provider id/version、标准规格和完整 payload。历史精确 schema 由同一通用 resolver
转给 provider 的注册 handler。[单 kernel 样例](../accelsim.md) 不作为完整模型证据。
engine 缺失、超时、非法来源、缺 kernel、错误绑定或未知指令必须失败；不得补零或复用历史预测。
这里只说明原生接入与结果复核，硬件精度、标准模型执行和真实权重预测分别需要证据。
