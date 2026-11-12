# Agent 架构探索环境

ArchExplorer 为 Portal Resident、Claude Code 和搜索策略提供同一套有状态工具。
模型由外部 Agent 宿主运行；本工具只提供 Python、JSON-lines 和 MCP 接口，以及离线
scripted 演示策略。无需配置 LLM API、模型名称或密钥。

## 配置与启动

`examples/adf-dmc.json` 是通用合成配置，可直接使用随包提供的
组件、mapping 模板与注册的 `events` 执行器运行。环境启动时通过
`arch_explorer.components.api.discover_components()` 加载内置目录和已安装扩展，供首次
评估前的种子校验、工具描述和 mapping 选择使用。`adf.evaluate` 也使用注册能力。
无需给示例配置额外注册模块。扩展作者见[独立组件包教程](component-extensions.md)。

`archexplorer components` 与 `describe_environment.component_catalog` 使用同一公共
组件目录，按硬件类型列出实现ID/版本、来源、输入、参数和限制。原 `components` 字段保留
结构搜索所需的参数域投影；不会在Agent内另建实现名单。ADF点的 `implementation`
携带精确绑定及独立参数，完整 `seed_adf` 和 `create` spec 都保留该字段。
普通 `set_parameters` 仍编辑共同硬件参数，不把实现参数混入共同命名空间。
同一SM的roofline工作量和Accel-Sim原生程序有不同契约，不能只换名称后直接复用输入。
见[实现选择与外包示例](component-implementations.md)。

公共编译先用 `task_adapters` 读取注册目录，再使用 `compile_workload`、`prepare_compiled` 和 `evaluate_request` 工具：
前两项不执行、不消耗评估预算，第三项对公共准备请求计一次尝试，失败同样计费；
对已保存结果只读校验则不消耗预算。按组件路径传bindings，适配器专参置于其options；
公共mapping传compute/memory/rules，具体options由注册规则验证。目录由公开task_adapters返回，不设Agent名单。
描述符与describe_environment.task_adapters一致。operator_mappings工具及describe_environment同名字段读取公共规则目录。
旧v1/v2编译分别通过显式migrate_compilation_v1/migrate_compilation_v2生成新v3，
不迁移或重标旧执行结果；缺兼容项、歧义、错域与专参错误由公共契约拒绝。
准备请求结果不会替换当前探索候选或进入best排名，原 `evaluate` / `submit` 保持原义。
参数与默认matmul/外包调用见[公共编译教程](compilation.md)。

配置 `compilation: {bindings?: object, implementations?: object}` 让既有 evaluate/search 对每个选中映射执行公共编译→准备→评估；
不同时指定旧backend。backend_options仍属于prepare。该模式的映射检查/编译/准备失败计一次候选尝试，阶段与原因保留；
无评估缓存。mapping_selectors可为请求对象或非空请求对象列表，后者逐项通过正式request_space。
search.proposals可保留原结构编辑，也可使用 `{tool:"set_parameters",args:{parameters:[...]}}`。
新编译探索结果由evaluate_request只读验证且不替换当前候选、不花新预算；原evaluate是明确的新评估。
CLI rerun/Python exploration.rerun另行创建新执行。预算与格式详见[attention与探索](attention.md)。

| 字段 | 语义；省略时的默认值 |
| --- | --- |
| seed_adf | 完整 ADF JSON，或 `{"preset":"dmc|vectorcgra_4x4|vectorcgra_4x4_diag","parameters":{...}}`；必填。Python API 也接受 ADF 对象。 |
| workloads | WorkloadSpec JSON 列表；kind、params、weight、name。weight 为有限正值；列表非空。 |
| backend / backend_options | 已注册后端名，默认 events；options 直接交给后端。 |
| registration_modules | 内置目录加载后额外导入的本地插件注册模块；默认空。仅用于扩展模型或显式测试替身，属于可信本地配置；提交和重放保留这些模块名。 |
| max_evaluations | backend 调用总上限，默认 100。 |
| cost_budget_usd / area_budget_mm2 | 正数或 null；null 不启用该约束。 |
| cost_penalty / area_penalty | 非负罚系数，默认各 1.0。 |
| parameter_samples / mapping_samples | 正整数，默认各 4。前者包括基线参数版本，后者是每 workload 的 mapping 上限。 |
| parameter_radius / growth_probability | 默认 0.25 / 0.75；规则见 structure.md。 |
| shrink_probability | 默认 0.75；已知成本或面积超预算时，优先缩减规模参数的概率。 |
| parameter_domains | 默认 []；按 path/param 列出 values 和 source 的显式数值搜索域。校验现有组件约束，支持为原本固定的坐标授权离散搜索。 |
| cost_model | 独立面积/成本模型配置：basis=raw 或 full、带来源的 process_nm / die_count、按 ADF 路径的 overrides；见 cost.md。 |
| random_seed | 整数，默认 0。相同操作序列与后端保证确定性时可重放。 |
| mapping_templates | 限定模板名列表；省略时使用所有已注册且适用的模板。 |
| trace_path | JSONL 文件路径，必须尚不存在；省略时自动建立临时目录。服务将路径输出至 stderr。 |
| search | max_steps、temperature、cooling、max_relative_degradation、可选 proposals。 |
| agent | script、max_steps、max_tool_calls、log_path，仅用于离线脚本。 |

示例把全部算法参数写明，面积/成本预算为 null。需要约束时设置预算和罚系数，并先检查
独立成本模型的口径与假设；不能把模型估计当作报价。
配置统一由 ADF 环境校验；未知字段和缺少必需字段返回普通配置错误。

```sh
archexplorer search --config examples/adf-dmc.json --policy sa --output results/sa.json
archexplorer search --config examples/adf-dmc.json --policy greedy --basic-only --output results/basic.json
archexplorer agent run --config examples/adf-dmc.json --policy scripted --output results/scripted.json
archexplorer evaluate results/scripted.json --output results/replayed.json
```

## 工具与状态

### CGRA 具名种子与通用搜索

`examples/adf-cgra.json` 与 `examples/adf-cgra-diag.json` 分别从普通/对角 mesh
起步，均使用能放入默认 SRAM 的合成 uint32 GEMM。种子参数直接传给 `devices.get`；
默认每 bank 32 字，`{"preset":"vectorcgra_4x4","parameters":{"words_per_bank":128}}`
是调用者显式选择的扩容场景，不能称为论文硬件参数。非法参数直接报错；容量不足不会自动扩容。

```sh
archexplorer search --config examples/adf-cgra.json --policy sa --output results/cgra-sa.json
archexplorer search --config examples/adf-cgra.json --policy greedy --output results/cgra-greedy.json
archexplorer search --config examples/adf-cgra.json --policy random --output results/cgra-random.json
archexplorer agent run --config examples/adf-cgra.json --policy scripted --output results/cgra-agent.json
archexplorer evaluate results/cgra-agent.json --output results/cgra-replayed.json
archexplorer env serve --config examples/adf-cgra.json
archexplorer mcp --config examples/adf-cgra.json
```

以上命令可把配置文件替换为 `examples/adf-cgra-diag.json`；所有 search/Agent 的结果
都能交给独立进程 `evaluate` 重放。两份示例固定 random_seed=17、总预算12、每次
mapping 候选4、最多6个结构步骤。默认结构提议与其他 ADF 共用组件目录和基本编辑，
有限步数内可能因物理不支持或重复而留下未用预算。scripted 脚本显式评估两种拓扑、
恢复和提交 best；这是一个声明的受限演示，不是默认可达集合。

`describe_environment.search_space` 报告 `component_graph`。任意合法 ADF 都可输入，
包括不同名字或分组的树；组件校验和映射/后端的物理有效域仍决定候选是否可执行。

外部 Agent 从真实工具返回值执行切换，例如在同一个 `env serve` 进程中：

```python
import json, subprocess
p = subprocess.Popen(["archexplorer", "env", "serve", "--config", "examples/adf-cgra.json"],
                     stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
def call(tool, args=None):
    p.stdin.write(json.dumps({"id": tool, "tool": tool, "args": args or {}}) + "\n")
    p.stdin.flush()
    response = json.loads(p.stdout.readline())
    if "error" in response:
        raise RuntimeError(response["error"])
    return response["result"]
description = call("describe_environment")
call("evaluate", {"optimize": "mapping"})
call("checkpoint", {"name": "seed"})
# Reuse the explicit, atomic topology edit from this example's scripted actions.
config = json.load(open("examples/adf-cgra.json"))
topology_edit = next(a["args"] for a in config["agent"]["script"] if a["tool"] == "apply_edits")
call("apply_edits", topology_edit)
call("evaluate", {"optimize": "mapping"})
call("restore", {"name": "seed"})
call("submit", {"selection": "best", "path": "results/cgra-tools.json"})
p.stdin.close()
assert p.wait() == 0
```

CGRA 使用通用组件目录候选与显式参数编辑，`parameter_samples` 按共同规则执行。
用户给出的参数选择必须满足组件域和后端物理关系；不支持的阵列、精度、容量或
时钟/带宽组合会在评价时报告错误。`cgra_spatial` 的 ii、placement、routing、memory_port
仍由现有 mapping 空间定义；ii 是标量最低发射间隔，不是编译器 modulo loop II。

示例不设置面积/美元预算；未知指标保留 null。启用这些预算时目标不可用。
未知物理clock用于周期归一化，不能把 latency_ns 宣称为测量纳秒性能。
原生 details.cycles、来源和警告在提交和重放中保留，见 [物理边界](cgra.md)。

### 通用工具

机器可读定义唯一来源为 `arch_explorer.agent_env.schemas.TOOL_SCHEMAS`；全部传输使用它。
Python 调用如 `env.evaluate(optimize="mapping")` 返回 `{"result": ...}` 或
`{"error":{"code":...,"message":...,"details":...}}`。也可用 `env.call(name, args)`。

| 工具 | 使用方式 |
| --- | --- |
| describe_environment | 读取目标、预算、组件角色/参数/单位/domain、编辑 schema、mapping 模板与全部工具 schema。 |
| get_state | 当前 ADF JSON、缩进树、规范指纹、组件面积/成本估计、最近评估和剩余额度。 |
| apply_edits | `{"edits":[...]}`；原子执行 1–3 个基本编辑或一个高层程序。返回展开的基本编辑。 |
| set_parameters | `{"parameters":[{"path":"chip/core/sm","param":"clock_ghz","value":1.2,"source":"synthetic choice"}]}`；限定 domain。 |
| evaluate | optimize 为 none、mapping 或 parameters_and_mapping。none 使用当前已选 mapping；首次自动取适用模板的一个合法候选。可显式提供每 workload 一个 mappings。 |
| checkpoint / restore | `{"name":"baseline"}`；保存/恢复 ADF、mapping、最近评估和编辑记录。checkpoint 可省略 name。 |
| undo | 撤销最近一次成功改变状态的操作，包括 evaluate 选出的参数/mapping。 |
| history | offset/limit 分页读取调用摘要与后端尝试；默认 0/100。 |
| best | 返回目标最小的完整评估候选及已用额度。 |
| submit | selection 为 best（默认）或 current；可选 path 写出可独立重放的 JSON 文件。 |

一次评估单位是“对一个 workload 的一个具体 mapping 尝试调用后端”。中/内层每个尝试
分别计数，失败也计数。开始参数版本前至少保留一个完整 workload suite 的额度；最后
不足一套的余额保留。结构编辑、参数设置、查看和恢复均不消耗额度。restore/undo 不退还
额度、不回滚历史、不重置随机数状态；best 跨恢复保留。

每个 workload 输出原始 Evaluation、按累计任务时间排序的算子、利用率最高的资源、
provenance 来源摘要。任务时间可能重叠，累计值不能当作端到端延迟。组件面积/成本构成
由 `arch_explorer.cost.estimate` 提供；目标使用它的 `area_mm2/cost_usd`，与后端延迟合成。
`get_state.costs` 与评估候选的 `cost_estimate` 包含类别/路径分解、来源、假设、略去和不支持项。
未知组件导致总面积/成本为 null；若影响启用的约束，目标也为 null，不能称为满足预算。
DRAM 硅面积独立列出，缺少它不等于逻辑 die 面积或采购价格未知。后端原有 Evaluation
数值保留为原始输出，可能仍为 null；不覆盖独立模型。详见 [成本模型口径](cost.md)。

结构编辑通过 ADF 语法与组件参数校验后，仍可能不满足后端构造或 mapping 的要求。
规则以[builder 与模板的 ADF 约束](components.md#adf-requirements-of-the-builder-and-the-templates)
为准。`evaluate` 失败时查看 `error.details.trials`：`cause` 保留结构化原因；若已尝试
后端，`cause.details.attempts` 包含计数索引、具体 mapping 与原始错误。例如 mesh 容量
不足时包含网络路径、端点数和 mesh 尺寸。模板全部不适用时返回 `no_mapping`，此时尚未
消耗后端额度；根据 `get_state` 中的当前树和上述约束修正结构，或恢复 checkpoint。
`history` 保留完整尝试记录，失败的后端调用仍消耗额度；失败不会成为最佳候选。

所有工具调用和返回写入 JSONL；首行为配置。相同模型版本和注册模块下可以重放：

```python
import json
from arch_explorer.agent_env import ExplorationEnv, replay_trace

env = ExplorationEnv(json.load(open("examples/adf-dmc.json")))
env.checkpoint(name="seed")
env.evaluate(optimize="parameters_and_mapping")
env.submit(path="results/best.json")
restored = replay_trace(json.loads(line) for line in env.trace_path.read_text().splitlines())
assert restored.used == env.used
```

trace 重放逐调用核对结果与计数，忽略 submit 的外部文件写入。checkpoint 只在当前服务
进程中存在；服务重启可用 trace 重放恢复。不同后端版本或随机后端可能产生 trace_mismatch。

## Claude Code：MCP stdio

安装可选依赖：在仓库根目录执行 `python -m pip install '.[mcp]'`。MCP 传输使用 SDK 1.x 的
low-level server；本版本验证过 1.30.0。未安装时命令给出明确安装说明。
项目 `.mcp.json` 的 mcpServers / command / args 格式见
[Claude Code 官方 MCP 文档](https://code.claude.com/docs/en/mcp)。使用绝对路径，以免受
Claude Code 启动目录影响：

```json
{
  "mcpServers": {
    "archexplorer": {
      "type": "stdio",
      "command": "/absolute/path/to/venv/bin/archexplorer",
      "args": ["mcp", "--config", "/absolute/path/to/adf-dmc.json"]
    }
  }
}
```

客户端接入后读取工具列表，再调用 describe_environment。stdout 只承载 MCP 协议，
日志走 stderr。同一个服务器进程是一条探索会话；不同 Agent 的公平比较需各启一个
新进程并采用同样配置。

MCP 的 JSON-RPC 帧由 SDK 校验。本版本验证的 SDK 1.30.0 会丢弃非有限 id 帧并继续
处理下一条合法请求；它不会为这种帧生成 JSON-lines 的 `invalid_request` 回复。
工具参数中的非有限数值返回 `invalid_arguments`，会话仍可继续使用。

## Portal Resident：JSON-lines stdio

Resident 可用自身已获授权的执行工具启动本地命令，持有 stdin/stdout 管道直到探索结束。
同一进程按顺序发送一行 JSON、读取一行回复；不要每次调用重新启动服务。
这条本地协议不需要 Portal 账号或密钥，也不会改变 Resident 的身份。

```python
import json
import subprocess

process = subprocess.Popen(
    ["/absolute/path/to/venv/bin/archexplorer", "env", "serve",
     "--config", "/absolute/path/to/adf-dmc.json"],
    stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1,
)

def call(number, tool, args=None):
    process.stdin.write(json.dumps({"id": number, "tool": tool, "args": args or {}}) + "\n")
    process.stdin.flush()
    reply = json.loads(process.stdout.readline())
    assert reply["id"] == number
    return reply

try:
    print(call(1, "describe_environment"))
    print(call(2, "checkpoint", {"name": "seed"}))
    print(call(3, "evaluate", {"optimize": "parameters_and_mapping"}))
    print(call(4, "apply_edits", {"edits": [{"op": "split", "point": "chip/dram", "num": 2}]}))
    print(call(5, "evaluate", {"optimize": "mapping"}))
    print(call(6, "best"))
    print(call(7, "submit", {"path": "/absolute/path/to/results/agent.json"}))
finally:
    process.stdin.close()
    process.wait(timeout=30)
```

每行必须包含 `id` 和字符串 `tool`，可选 `args`。id 接受字符串、整数或 null；不接受
布尔值、小数、对象或数组。解析拒绝 NaN、Infinity、-Infinity 及浮点溢出（如 1e309），
返回 `invalid_request` 后继续下一行；解析失败或 id 不合法时响应 id 为 null。
合法帧的 id 原样回传。响应无法编码成有限 JSON 时返回 `invalid_response` 并继续；此时
工具可能已经执行，应先查看 history，避免直接重试有状态操作。

未知工具、错误参数及非法编辑返回 error，服务继续接受下一行。`evaluate` 和 `submit`
的结果文件可由独立进程重放，使用同样的组件/backend 注册入口。

## 离线脚本与公平比较

`agent run --policy scripted` 不做网络请求。`agent.script` 是 `{tool,args}` 列表，逐条
执行；示例配置演示查看、checkpoint、基线评估、结构编辑、评估、restore、best、submit。
没有 script 时持续调用 parameters_and_mapping 直到评估预算或步数上限。
max_steps 默认 30，max_tool_calls 默认等于 max_steps；结束时导出已经找到的最佳候选。
最终导出不消耗评估额，单独列于 trace。脚本步数包含失败调用。

SA/greedy/random 与 Agent 比较时保持种子 ADF、workload/权重、后端及模型版本、预算与
罚系数、中/内层样本数、随机种子一致。用实际 backend 调用数绘制最佳目标轨迹，报告
invalid/duplicate 提议与提前停止；不能用工具调用次数替代评估次数。若要比较结构策略，
为 scripted 明确提供结构编辑序列；无 script 的默认策略只做固定结构的参数/mapping 调优。

## 给 Agent 的探索任务模板

```text
在当前 ArchExplorer 环境中寻找低延迟架构。先调用 describe_environment 和 get_state，
确认 workload、权重、评估余额、成本/面积预算与罚系数。不得修改这些实验条件。
检查 get_state.costs 的 basis、assumptions、omitted、unsupported；不得把参考模型当成完整物理设计或实际采购报价。

先保存 seed checkpoint，再以 parameters_and_mapping 评估基线。根据组件目录和瓶颈摘要，
提出 1–3 个基本编辑或一个高层变换；记录每次改动希望改善的瓶颈。编辑成功不代表后端
一定能构造或执行；评估失败时读取 error.details.trials 的 cause 和后端 attempts，结合
get_state 与 docs/components.md 的 “ADF requirements of the builder and the templates”
一节修正或恢复。不要把未评估结果当作性能证据。结构改变后使用相同中/内层设置评估，必要时 restore；
注意 restore 不退还额度。计数依据后端评估次数。

预算不足时停止评估。调用 best，检查 feasible、unavailable_constraints、每个 workload
延迟与 provenance。缺失指标须在结论中明确写出。以 submit(selection="best", path=指定路径)
交付结果，并用 archexplorer evaluate 独立重放。最终报告基线与最佳目标、实际评估次数、
主要结构/参数/mapping 变化、预算可用性和模型限制，附结果与 JSONL trace 路径。
不要发起额外模型 API 请求；直接使用当前宿主提供的探索工具。
```

## 条件 mapping 空间：聚合与显式指令

`describe_environment().result.mapping_spaces` 是按 workload 顺序排列的模板契约。
普通模板的 `contract.kind=flat` 仍为参数名到选项列表；条件模板的
`contract.kind=conditional` 列出互斥的 `branches`、直接 API 的 `direct_defaults`、
默认采样选择 `sampling_defaults` 和被硬件/负载排除的 `unavailable` 原因。
`sampling_spaces` 给出当前配置筛选后的活动参数乘积，可逐键选择其中一个值生成完整 mapping。
该发现步骤只做输入校验，不运行指令或事务时序；结构/负载变化后需重新发现。

`pim_offload.space(adf, workload)` 返回 `ConditionalMappingSpace`，不再把互斥参数压成一个字典。
Python 调用者可使用 `arch_explorer.adf.registry.mapping_spaces(space, selectors)` 取得活动乘积；
不要再用 `space.items()` 或把各分支的键取并集。普通模板的 `.space()` 返回值不变。

| 分支 | 活动参数 |
| --- | --- |
| aggregate / aggregate | attention、sub_batches、tp、inner、两个 execution 字段，以及**所选** inner 模板的 `inner.*` |
| instructions / aggregate | attention=pim、sub_batches=1、合法 tp、inner=gsm_tiled、两个 execution 字段；无 `inner.*` |
| instructions / transactions | 同上；另需满足源事务硬件和 workload 条件 |
| aggregate / transactions | 不支持，不出现在合法分支中 |

默认采样只包含 aggregate/aggregate。显式指令执行成本较高，应在小场景中主动选择，
例如配置中加入以下字段（ADF 为符合条件的 NeuPIMs，workload 为受支持的 FP16 decode）：

```json
{
  "mapping_templates": ["pim_offload"],
  "mapping_selectors": {
    "pim_offload": {"npu_execution": "instructions", "memory_execution": "transactions"}
  },
  "parameter_samples": 1,
  "mapping_samples": 1,
  "max_evaluations": 1,
  "search": {"max_steps": 1, "proposals": []}
}
```

这是附加配置片段，仍需 `seed_adf` 和 `workloads`。把 memory_execution 改为 aggregate 即选
指令/带宽消融路径。`mapping_selectors` 按模板固定选项；未给出的 execution 字段仍使用
sampling_defaults，不能只写 transactions 让系统暗中开启 instructions。
普通 flat 模板也可用 selectors 固定选项；无匹配分支、未知键或未活动键都报错。
采样按互斥乘积的总大小抽有限索引，不物化大笛卡尔积；count、seed 和失败预算口径不变。

完整环境/search mapping 必须包含所选分支的全部活动键，包括 execution 字段。
`submit` 原样保存完整 mapping，CLI `evaluate` 按同一条件契约验证后重放；不丢弃非活动键。
直接 `adf.evaluate` 保留简写兼容：`MappingSpec('pim_offload', {})` 是原聚合默认，
`{'npu_execution':'instructions'}` 默认 memory=aggregate、attention=pim、sub_batches=1、tp=1、
inner=gsm_tiled。聚合 inner 参数省略时沿用对应 inner 模板默认。简写不是完整 Agent mapping；
向环境传显式 mappings 时应从发现的活动空间补齐所有键。

旧结果若把 gsm_tiled 与 dmc_partition 的 inner 参数并在一起，或把 inner.* 带进显式指令，
现在明确拒绝：这些原先被忽略/互相冲突的配置不能通过偷偷删键冒充等价重放。
应保留原文件并从合法分支重新评价生成新结果。硬件容量/时钟、FP16/iteration、head/GQA、
返回 buffer 与 KV 地址范围等条件不满足时，unavailable 会记录原因；不是自动扩容或切换物理模型。
模板发现无法预知所有运行时容量/调度失败；这些仍按实际 backend 尝试记录并消耗预算。
