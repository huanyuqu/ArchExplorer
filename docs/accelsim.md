

# Accel-Sim 原生 GPU 后端

`backend="accelsim"` 将 ADF 组件编译为原生配置，由外部 Accel-Sim
统一推进周期，再返回 `Evaluation`。Python 不模拟第二份时序，也不将
SM、缓存、互连和 DRAM 的重叠周期相加。完整小尺寸 ViT/CLIP 现在由
[程序级入口](accelsim-program.md)执行：完整数值参考、实际 mapping、
多 kernel 单流和一次原生进程。该路径使用明确合成的参考程序与权重，
硬件时延精度及框架编译/kernel 选择等价性仍未验证。下文单 kernel
合同及历史标定接口保持各自版本边界。

固定来源为 [Accel-Sim v2.0.1](https://github.com/accel-sim/accel-sim-framework/tree/1c9cc831c5d948f89fa9dc8f21f6c87952f5f2ea)
和 [GPGPU-Sim v2.0.1](https://github.com/accel-sim/gpgpu-sim_distribution/tree/c24e8858eee552b3bc2919b147f14e2a88571775)。
后端不下载工具，不启动 CUDA/GPU kernel，不携带原生二进制或第三方源码。

## 组件与支持结构

`arch_explorer.archs.accelsim.sm90_adf()` 提供 SM90_H100 配置种子。
原生配置所有显式选项均来自 ADF 参数，未列出的原生选项使用上述固定版本的解析器默认值。
`simulator/accelsim/profile.py` 记录每个字段的上游文件与行号；大部分参数固定为单值。
合成探索可以修改公开列出的合法子集，不能据此推断修改后的真实设备精度。

| ADF 对象 | 原生责任 | 实例与约束 |
| --- | --- | --- |
| `gpu/cluster/sm`：`accelsim_sm` | `trace_shader_core_ctx`、调度器、执行和寄存器资源 | 每 cluster 一个 SM；寄存器块上限不超过 SM 容量 |
| `gpu/cluster/l1_shared`：`accelsim_l1_shared` | `ldst_unit`、L1D/shared | 每 SM 一份；缓存结构、shared 容量与银行配置固定 |
| `gpu/icnt`：`accelsim_icnt` | `InterconnectInterface`、BookSim 单级 fly 网络 | `network_mode=1`；n=1、subnets=2、单 VC；k=SM clusters+L2 subpartitions |
| `gpu/partition/l2`：`accelsim_l2` | `memory_sub_partition`、L2 | 每 DRAM channel 两份；缓存配置与 LRC 固定 |
| `gpu/partition/dram`：`accelsim_dram` | `memory_partition_unit`、`dram_t` | 20/40 channels、两个 chiplet partitions；保留上游 simple DRAM 配置 |

GPU 根 `repeat=1`；两组只含表中点，不含内层网络或递归矩阵。
SM cluster repeat 支持 2/4/8/16/32/64/128/132；DRAM partition repeat 支持 20/40。
除 L2 repeat=2，其余点 repeat=1。改变组 repeat 时，须同步更改互连 `booksim_k`。
不支持任意 mesh、多 GPU、异构 SM 或其他连接结构，非法结构在启动前拒绝。

合法可调子集为每 SM/每块寄存器 32768/65536、CTA 上限 16/32、L1 延迟
41/42/48、shared 延迟 29/30、L2 ROP 延迟 237/238、DRAM 延迟
194/195/200，以及 BookSim input/VC buffer 128/256/512 flits。
core 支持 990/1980/3960 MHz，icnt 和 L2 必须等于 core 的两倍，DRAM
支持 1553/3106 MHz。这些是首版配置能力约束。原生有更多选项，首版未开放它们。

`compile_adf(adf)` 返回配置内容与 `component_map`。输出为
`gpgpusim.config` 和 `config_hopper_islip.icnt`；后者同样逐字段从 ADF 生成。
每个映射项包含 ADF path、参数、值、单位、来源、原生配置键/对象、实例数和约束。
`config_sha256` 对这两个文件组成的、按键排序的 JSON 对象求 SHA-256。
运行限制另外写入 `limits.config`，其哈希在 run 记录中保留。

原生枚举将 `network_mode=1` 定义为 BookSim，`2` 才是 LocalInterconnect。
H100 配置中“local xbar”的注释不能替代代码枚举。本后端保持模式 1 与其完整
单级 fly 配置，拒绝静默切换模式。声明组件的 `build()` 在旧 events 后端明确拒绝；
它们不提供 Demand 模型。

## 可安装的真实执行入口

先按上游说明独立构建固定版本的 `accel-sim.out` 和 GPGPU-Sim 的
`libcudart.so`。这是 CPU 时序模拟器，构建需要 CUDA headers/tools、GCC、Make、
Bison、Flex、M4、makedepend、zlib、OpenGL 和 zstd；不需要运行 GPU。
上游 Make 路径还需要将 `util/tracer_nvbit/tracer_tool` 加入头文件搜索路径。
某些系统需 `--no-as-needed`，因为上游将库参数放在对象文件之前。
本地隔离安装和构建记录由部署者保留，避免 setup 自动更新移动的 dev/master。

引擎 manifest 使用 `arch-explorer.accelsim-engine.v1`：

```json
{
  "schema": "arch-explorer.accelsim-engine.v1",
  "sources": {
    "accelsim": "1c9cc831c5d948f89fa9dc8f21f6c87952f5f2ea",
    "gpgpusim": "c24e8858eee552b3bc2919b147f14e2a88571775"
  },
  "binary": {"path": "/absolute/path/accel-sim.out", "sha256": "实际文件哈希"},
  "libraries": [{"path": "/absolute/path/libcudart.so", "sha256": "实际文件哈希"}],
  "build": {"command": "可重放构建命令", "dependencies": "依赖版本、下载来源和哈希"}
}
```

这里的 `libcudart.so` 必须是 GPGPU-Sim 构建产物，不能填 CUDA runtime。
manifest 为本地可信构建的身份记录，哈希本身不构成供应链认证。
执行前核对每个二进制/库的哈希；输出还必须带两个固定提交的原生 build identity。
缺失引擎抛出可操作的 `ImportError`，不会回退其他模型。

安装 ArchExplorer 后，可在源码树外生成完全公开的合成样例：

```sh
python -m arch_explorer.components.accelsim.example /tmp/accelsim-example --engine /absolute/path/engine.json
archexplorer evaluate /tmp/accelsim-example/request.json --output /tmp/accelsim-result.json
archexplorer evaluate /tmp/accelsim-result.json --output /tmp/accelsim-replayed.json
```

CLI request 的 schema 是 `arch-explorer.accelsim-request.v1`，顶层恰为
`schema, adf, workload, mapping, options`。`options` 接收 `engine`、
`output_dir`、`timeout_s`、`max_cycles`、`memory_limit_mb`；默认分别为未设置、
临时目录、120 秒、1000000 core cycles、4096 MiB。
结果含完整 request，可直接由相同 CLI 重放。
路径在当前进程中解释，建议使用样例生成器输出的绝对路径。

公开 Python API 与 CLI 使用相同后端：

```python
import json
from arch_explorer.adf import WorkloadSpec, MappingSpec, evaluate, from_dict

r = json.load(open('/tmp/accelsim-example/request.json'))
result = evaluate(from_dict(r['adf']), WorkloadSpec(**r['workload']),
                  MappingSpec(**r['mapping']), backend='accelsim',
                  cost_model=False, **r['options'])
print(result.latency_ns, result.details['native'])
```

## Trace 与 mapping 契约

`WorkloadSpec.kind` 仅支持 `operator`，`MappingSpec.template` 必须为 `accelsim_trace`。
trace 适用性模块统一检查以下字段，公共 evaluate、CLI 和结果重放都在启动原生引擎前消费它。
这些字段全部必需，未知字段会被拒绝；首版不额外开放执行参数或自由 metadata。

| 对象 | 必需字段与精确类型 | 权威与意义 |
| --- | --- | --- |
| workload.params | `op`：字符串 `trace_kernel`；`kernel`：合法标识符字符串；`grid`、`block`：三维正整数数组；`accelsim`：文件引用对象 | kernel、grid、block 必须与实际解析的 trace 一致；只接受当前 trace_kernel 语义 |
| workload.params.accelsim | `manifest`：非空路径字符串；`sha256`：64 位小写十六进制字符串 | manifest 原始 bytes 的 SHA-256 |
| mapping.params | `trace_id`、`compiler`：非空字符串；`grid`、`block`：三维正整数数组 | 几何来自实际 trace；trace_id/compiler 是必须与 manifest 同文的调用方来源声明 |
| manifest.workload | `kind`、`params`，其中 params 为上面的 op/kernel/grid/block，不带 accelsim 引用 | 字段同样必需且严格校验，并分别与请求及实际 trace 核对 |
| manifest.mapping | 完整 `template`、`params` | 同一 mapping 契约，不能仅靠两份声明相等绕过实际几何 |
| manifest.kernels[0] | file、两个 SHA 字符串、name、grid、block、stream、thread_instructions、ctas | 所有嵌套项按类型检查，名称、几何、stream 和完成计数必须与实际 trace 一致 |

grid/block 必须是恰含三个正整数的 JSON array；bool、float、字符串、tuple、空/短/长数组均拒绝。
kernel.stream 为非负整数，ctas/thread_instructions 为正整数；Python 的 True==1 或 1.0==1
不能替代类型检查。v1 实际 trace 必须为一个 32-thread CTA、stream 0，manifest 顶层 scope
必须为 `single-kernel-device-cycles`。workload/mapping 不能额外声明 stream/scope 来扩大范围。

合法新 kernel 名称可以使用，须同步实际 trace、请求和 manifest；没有示例名字白名单。
任意 op、shape、batch、tile 等未认证参数会被拒绝，不会被当成另一任务的执行结果。
name/weight 仍是 WorkloadSpec 的展示名称和套件权重，不影响执行身份。

哈希说明文件 bytes 一致，不认证采集或编译历史。source、compiler 和 trace_id 属于调用方声明：
修改它们并同步重算 manifest 哈希只会变更来源声明，软件无法从相同 SASS bytes 认证真实编译器、
设备采集或历史。执行事实始终按实际 trace 核对，不能由来源声明覆写。
此后端首版不接入现有 mapping 搜索模板。

trace schema 是 `arch-explorer.accelsim-trace.v1`，完整例子由上述命令生成：

- `format="traceg"`、`tracer_version=3`、`isa="sm90"`，固定引擎和 profile 身份。
- `source` 说明 kind、GPU、toolchain 和 description；样例明确为手写合成数据，无 GPU 来源。
- `commands` 指定 `kernelslist.g` 及哈希，内容恰为一个 `kernel-*.traceg` 文件名。
- `kernels` 恰一项，包含原始/当前文件哈希、名称、grid/block、stream、CTA 和线程指令数。
  首版原始 trace 与消费 trace 相同，不接受未经验证的派生转换。
- `scope="single-kernel-device-cycles"`。

首版只接受一个 32-thread CTA、一个完整 warp、stream 0；支持 IADD3、LDG.E、
STG.E 和末尾 EXIT。指令、寄存器、完整 block、头字段、mask、指令数和哈希
在启动前核对。没有 shared/local 寻址、分支、tracez、异步指令、多 kernel、并发或 memcpy 支持。
上游能支持更多，本合同尚未为它们定义适用性验证。
将单 kernel trace 改称 ViT/CLIP 会被拒绝，不能产生完整模型时延。

## 动态 trace v2

`arch-explorer.accelsim-trace.v2` 保留上面的请求与 mapping 字段，通过实际输入事实扩展适用域。
支持 SM90 的普通 traceg/tracez v3/v6、完整多 CTA/warp、合法可打印内核名、动态 PC 重复、
零/部分 active mask 和逐 lane EXIT。所有 grid 中的 CTA 与 block 中的 warp 都必须出现，
每条 warp 流必须消费登记记录并 EXIT 全部有效 lane。CTA/warp 记录顺序不作为名字或尺寸白名单。
文本 header 必须由一行格式注释终止，这是固定原生 reader 的边界，不能用 `#BEGIN_TB` 代替。

v2 kernel 项新增必需的 `facts`：实际 header 资源、CTA/warp 流数、动态 warp/线程指令数、
opcode/mask 直方图、重复 PC 数和逐流摘要。它们与实际文件、请求及 manifest 逐项绑定，
更新哈希不能伪装不同几何或计数。源码中的 Hopper base opcode 表来自固定提交；
已知 opcode 还须满足普通操作数和后缀格式。未知 opcode/ISA、Tensor/TMA/异步专用格式、
跨 CTA cluster、动态资源变更均明确拒绝，不能消化成零时间。普通 LDS/STS 与 BAR.SYNC
有小型合成原生检查；这不认证所有命名/异步屏障。原生只为 R 寄存器建立 scoreboard，
UR/P/UP 依赖未被完整建模；可执行软件不能因此声称真实硬件时序正确。

tracez 校验按固定 `tracez_format.h` 的 footer/index 与 zstd 分块读取，核验偏移、计数、
帧和完整执行范围。原始 tracez 字节直接交原生引擎；不重写 ISA/指令或删除 predicated-off 记录。
默认边界为原文件 64 MiB、展开文本 256 MiB、单分块 8 MiB、4096 CTA、200 万动态记录。
内核真实寄存器/shared/block 资源在启动前核对 ADF；该检查不添加时延修正。

外部已采集 trace 可在安装包外创建请求：

```python
from arch_explorer.components.accelsim.trace_v2 import create_request

request = create_request('/data/kernel-77.tracez', '/tmp/native-request',
    source={'kind': 'hardware', 'gpu': '来源设备声明', 'toolchain': '采集工具版本',
            'description': '完整 kernel 的来源和子范围'},
    trace_id='external-resource-id', compiler='来源编译声明', engine='/data/engine.json')
```

随后运行既有 `archexplorer evaluate`。来源字段是声明，哈希不是采集真实性证书。
单个 H200 来源 kernel 在 H100 profile 下成功只证明软件消费；不代表任一设备精度。
真实 trace 与原生工具是外部资源，不包含在 wheel 中。

无引擎/无设备的能力预检：

```sh
python -m arch_explorer.components.accelsim.capabilities /tmp/native-request/request.json
python -m arch_explorer.components.accelsim.capabilities --cubin /data/existing.cubin
python -m arch_explorer.components.accelsim.capabilities /tmp/native-request/request.json --purpose hardware-accuracy --target /data/target-identity.json
```

返回 `passed`/`blocked` 和分类理由，blocked 退出码为 2。target identity 具有
`gpu, isa, evidence_sha256` 三项，引用既有设备记录，预检不查询 GPU。
静态 cubin 只能提供离线 ISA 见证，不能重建动态 trace。固定原生版本无 SM120 OpcodeMap，
Blackwell ISA 与目标物理参数缺口分别报告；不默认采用 H100 或零成本。

固定原生 `trace_driven.cc` 的 dispatch 包含 SM35、60、61、70、75、80、86、90。
这与本适配器已验证的普通 SM90 输入域分别声明：SM80/86 等返回 `adapter_isa`，
不能说原生没有对应 ISA；SM120 返回 `native_isa`。静态 cubin 始终另报
`dynamic_trace_missing`，有 dispatch 也不认证全部操作数、依赖或真实设备时序。
target、cubin、v1/v2 manifest 和实际 trace header 复用 `native_isa.py` 中同一事实表。

## 保留结果的只读协议审查

`arch_explorer.components.accelsim.protocol` 校验已有完整预测与实际消费对象的闭合，
复用生产 `load_trace`、`compile_adf`、`load_engine` 和 `parse_stats`。
它不启动引擎、查询设备或建立运行目录。软件通过仅表示事后审查闭合；
不能把旧预测宣称为新测量前的冻结，也不能据此开启硬件窗口。

```python
from arch_explorer.components.accelsim.protocol import prepare_review, preflight_protocol

spec = prepare_review('/data/request.json', '/data/result.json',
    '/data/source-origin.json', '/data/source.tar', '/data/history.json',
    '/tmp/new-review')
result = preflight_protocol(spec)
```

`prepare_review` 只在不存在的新目录生成审查元数据；保留的结果、源码归档及原生
run/stdout/stderr/暂存输入必须已存在。原生执行源码与当前审查器源码分别记录，
不会把新审查器版本追认成旧执行版本。命令行等价入口：

```sh
python -m arch_explorer.components.accelsim.protocol /tmp/new-review/protocol.json
```

协议为 `arch-explorer.accelsim-protocol.v1`，必需字段是 `schema, purpose,
review_kind, request, frozen, history, case_identity, target, compile_trace_binding`。
当前 `review_kind` 固定 `posthoc-software-review`，`compile_trace_binding` 固定 false。
`frozen` 是唯一、完整的 `{role,path,sha256}` 列表，绑定实际 request、manifest、
trace、commands、ADF、完整 config、engine manifest/binary/全部库、完整 prediction、
执行 source/来源/归档、history、native run/stdout/stderr 和每个实际暂存输入。
缺项、重复、额外角色、类型错误或重算哈希后的无关文件都拒绝。

预测必须包含完整 `arch-explorer.accelsim-result.v1` 封套，所有执行身份、运行状态、
config/ADF、guard、trace 范围及数值都与保留记录一致。只规范化输出位置和 workload
展示名/权重；timeout、cycle limit 和 memory limit 参与身份。顶层 latency、两份
JSON native 统计与哈希绑定的原始输出经生产解析器重新得到的周期、频率、计数和范围
闭合。复制一致的错误 JSON 数字不能作为独立原生证据。

外层封套保持 v1，供已有 CLI 的结果重放入口识别。新结果的 details 必须声明
`schema=arch-explorer.accelsim-result-details.v2` 和
`component_map_schema=arch-explorer.accelsim-component-map.v2`，并包含
`device_facts`。没有设备事实绑定时该字段必须为 null；有绑定时，它必须与当前
请求审计得到的整个报告严格一致，包括原件身份、known、unknown、合成假设、
launch facts、hardware accuracy 和 live lease。缺失、多余、混用版本或删去未知项
均拒绝；历史设备事实不授予当前执行租约，也不构成物理精度验证。

公有协议只读兼容两个有限的历史生产语义：SIM28 c2d4bdc 与 SIM29 94a515c。
选择依据是实际 run 与归档闭合后，生产 backend/config/component/profile（以及
适用的 device/constraints）的固定字节摘要，不能仅靠调用方版本标签选择。
SIM28 的旧有限参数域、拓扑来源说明和按组件归属的旧时钟域注释按当时格式完整
重建并比较；SIM29 保留当时的事实字段、参数域、来源分类与 CORE ready-cycle
注释。新版本还审计拓扑事实分类。这些投影只处理上述版本的元数据差异，
有效 native 配置字节/哈希、参数值/单位/来源、完整 provenance、拓扑实例、
request/engine/trace/guard、原生输入输出与所有数值仍逐项闭合。
旧注释不被升级为已验证物理事实。不识别的生产字节或未来结果版本保守拒绝，
需要显式新增兼容规则；既有冻结 request/prediction/source/run 不改写。

case identity 从实际 trace 字节与执行范围生成，不依赖 manifest 的描述字段；
已知同一动态 flow 的 tracez/text 另有 semantic identity，防止换编码规避已知重合。
旧 `arch-explorer.sim27-history.v1` 只读兼容；新 `arch-explorer.accelsim-history.v1`
以哈希绑定的 parents 引用原记录并累积 case/semantic/binary 登记，不清空旧集合。
未知跨二进制或采集关联仍未知，没有已知重合不等于证明未测。

执行源码归档须绑定来源记录、commit 标签及实际执行模块字节；这些是可核对的
关联，不是签名证书。独立测量/编译采集关联和新窗口尚未验证，`hardware-accuracy`
仍拒绝。旧私有 draft 协议必须显式迁移，不能追改历史冻结或来源证据。

## 时间、结果与失败

`latency_ns = final gpu_tot_sim_cycle × 1000 / core_clock_MHz`。
来源为固定 GPGPU-Sim `src/gpgpu-sim/gpu-sim.cc`：`init_clock_domains()` 将 MHz
换为 Hz，`gpu_print_stat()` 输出累计 core cycles。只取最后累计计数；
单 kernel 的 `gpu_sim_cycle` 必须与之相等，不重复相加。
范围包含配置声明的原生 kernel launch latency、设备执行和排空，不含宿主 CPU
运行时间、拷贝或整模型其他工作。其余时钟域保留在配置与映射中，由原生调度器处理。

仅映射可靠的完成周期和退休线程指令数；v2 还核对原生退休 warp 指令数。
`gpu_tot_issued_cta` 是发射数，没有直接打印的 completed-CTA counter；
完成性来自完整输入 EXIT、精确退休计数与无截断正常退出的联合核查，报告明确标注推断依据。
`breakdown`、`utilization` 为空，
面积和成本为 `None`。默认成本估算器也将这些新组件标记为未知。
结果 details 保留组件映射、能力、trace/config/engine/适配代码哈希、原生统计和证据目录。

每次执行建立独立目录，保存配置、暂存 trace、command list、stdout/stderr 和 `run.json`。
只传固定参数数组；不提供 shell 命令字段。上游 legacy reader 内部使用 popen，
因此暂存目录必须是 shell-safe 绝对路径，文件名固定；原始路径不会被直接拼进 shell。
原生进程使用受限环境、独立进程组、内存/core dump/CPU/wall timeout 限制。
超时仅终止此进程组；失败目录保留。

非零退出、deadlock、max-cycle 截断、缺失/重复 kernel、统计不完整、错误 ISA/
哈希/来源字段/mapping 都不能生成成功 Evaluation。原生退出码为零也必须通过
完整计数检查。未知字段和不支持的参数/拓扑在启动前拒绝。

## 来源与许可

本项目 Python 接入代码适用 Apache-2.0，保留项目 NOTICE。
上游配置参数来自上述两个固定提交，相关版权为 Timothy Rogers/Purdue University
及 GPGPU-Sim 作者；上游采用其各自 BSD-style 许可。
源码、二进制及 CUDA Toolkit 仍适用各自 LICENSE/版权声明，未被重新许可为 Apache-2.0。
本项目不分发它们。部署时保留下载包及引擎源码的 LICENSE/COPYRIGHT 文件。
参数来源的上游许可证原文随 `simulator/accelsim/profile.py` 的
`THIRD_PARTY_NOTICES` 保留在源码、wheel 和 sdist 中。

[官方路线](https://accel-sim.github.io/) 将 Blackwell 列为实验性后续工作。
本入口只接受上述 SM90 子集；H100 配置和合成 trace 不能证明 RTX PRO 5000
Blackwell 的 ISA、资源、调度、时钟或物理精度。后续需要分别补齐真实 trace 输入范围、
设备支持审计和独立冻结的硬件验证，再推进完整 ViT/CLIP 评价。


## SM120 typed mechanism variant

`arch-explorer.accelsim-engine.v2` identifies a separate patched engine.
The installable backend verifies the actual binary and simulator-library
SHA-256 against an approved build, both upstream revisions, minimal patch
identities and resulting source trees. The fixed original engine continues
to consume its original SM90 v1/v2 requests. A SM120 trace requires the new
engine and the explicit `sm120_mechanism_adf()` configuration.

This is a synthetic two-SM mechanism topology with forty memory channels
and four independent logical R/UR/P/UP bank spaces. It does not represent
a calibrated Blackwell device. Uniform register accesses count one logical
warp-wide unit; P/R accesses count observed lanes. All timings still come
from the one native resource simulation. There is no Python body time or
shape correction. Area, energy and hardware accuracy remain unknown.

The new trace manifest is `arch-explorer.accelsim-trace.v3`, with actual
binary version 120 and typed native trace version 7. Every instruction
preserves explicit predicate-control identity and the observed pre-predicate
active mask, including all-predicated-off records. Width and lane addresses
are explicit; ordinary LDG/STG widths 16/32/64/128 bits, descriptor UR pairs
and wide R pairs are preserved. Supported bases/modifiers are enumerated
in `typed_trace.py`; specialized/unknown instructions and ordering modes
refuse execution. The native operand collector and scoreboard consume the
typed identities, including multiple outputs and zero/true constants.

For hardware sources, `capture_binding` freezes original SIM28NV1 records,
complete NVBit static metadata, actual cubin, registration, launch and live
preflight. The public reader reconstructs the consumed trace from these
original bytes, checks every static instruction encoding against ELF .text,
full per-warp straight-line PC sequences and lane exits, resource scope and
actual allocation bounds. This capture dialect currently covers complete
straight-line ordinary kernels with predicated EXIT, not arbitrary divergent
control flow. The independent offline capture reader additionally checks
static decoded operands and controlled effective addresses/output values.
Hash binding establishes byte identity; it does not certify a dishonest
provenance declaration or replace independent observation.

`typed_trace.create_request()` has the same source/trace_id/compiler/engine
interface as the v2 creator and accepts `capture_binding` for hardware inputs.
`python -m arch_explorer.components.accelsim.capabilities --cubin FILE
--engine ENGINE.json` can distinguish actual approved SM120 native capability
from the original engine. A static cubin still lacks a dynamic trace. The
hardware-accuracy preflight stays blocked by missing validated physical
parameters even after complete software execution.

Patch/build details are in `tools/accelsim_native/README.md`, and the bounded
NVBit ABI is in `tools/accelsim_native/capture/SPEC.md`. No native binary,
NVBit library, private capture, hardware result or model weight is shipped.


### Device-fact constrained scenarios (SIM29)

The original H100 and SIM28 two-SM fixtures remain attributed software seeds.
The configuration compiler also accepts positive cluster/channel counts and
independent positive integral-MHz core/interconnect/L2/DRAM clocks. This is a
configuration capability, not a calibrated Blackwell profile. No Python latency,
shape, operator, or measured-residual adjustment is applied.

`archs.accelsim.device_scenario(assumptions, evidence)` reads an existing
`arch-explorer.sim28-live-preflight.v1` file without querying a GPU. Its original
SHA256, UUID and CC are bound through an optional `device_evidence` Param. That
Param contains a file reference only; all effective values have one authority in
ADF. The compiler rereads the bound original and rejects changed bytes, units,
source labels, resource values or SM count. A hash proves byte identity, not that
a caller's claimed device history is true. The supported historical importer
currently targets the existing CC12.0 ordinary mechanism; other CCs are refused.

```python
from arch_explorer.adf import Param
from arch_explorer.archs.accelsim import sm120_mechanism_adf, device_scenario
from arch_explorer.components.accelsim.config import compile_adf

# An explicit, complete set of synthetic assumptions is mandatory. The importer
# supplies no missing clock/cache/DRAM/NoC performance values.
a = sm120_mechanism_adf()
a.networks[0].params['booksim_k'] = Param(190, 'count',
    'synthetic assumption: 110 SM endpoints + 40 channels * 2 subpartitions')
a.child('cluster').child('l1_shared').params['gpgpu_adaptive_cache_config'] = Param(
    0, '1', 'synthetic assumption: static shared allocation; cache split unknown')
scene = device_scenario(a, '/data/historical-preflight.json')
compiled = compile_adf(scene)
print(compiled.device_facts)  # known facts, assumptions, unknowns; live_lease=False
```

For the recorded device, factual imports are 110 SMs, register capacity per SM
and per block, shared capacity per SM and per block, and CC12.0. Warp size and
per-block thread bound constrain input applicability. A captured function's
occupancy does not establish the device's maximum threads or CTAs per SM.
Shared initial allocation is derived from the single shared-capacity Param when
omitted; a contradictory explicit alias is rejected. Static allocation is
required for a fact-bound shared capacity because the adaptive cache split has
not been observed. `counterfactual(scene)` explicitly removes the device-fact
claim and labels its retained values synthetic for sensitivity experiments.

Fact claims are a finite semantic mapping from `(native owner, ADF key)` to the
original pointer and unit. Every claimed value, unit, pointer and evidence SHA
must match that mapping; equal numeric values or equal units do not make two
parameters interchangeable. Unobserved fields cannot borrow a fact source.
The same audit supplies component-map classifications, including the SM-count
repeat and the declared static shared-allocation alias. Other repeats remain
explicit scenario topology. File identity is not an execution lease.

The native constraints retained before launch include:

- One core per cluster in this adapter subset; no dummy SM padding. L2
  subpartitions per channel must be a power of two in 1..16, not fixed to two.
- Chiplet count divides both clusters and channels; native IPOLY-MODULO indexing
  requires the power-of-two factor of total subpartitions to be 16, 32 or 64.
- BookSim remains single-stage fly, destination-tag routing and identity endpoint
  mapping; `k = clusters + channels * subpartitions`, bounded to 4096 ports.
  Unsupported topology/route/hash combinations are rejected, not renamed.
- Registers/shared per block cannot exceed their corresponding per-SM capacities;
  the optional shared default must agree. CTA capacity stays within native
  MAX_CTA_PER_SHADER=32. Clock domains need no fixed 2:1 relationship.
- Core MHz is bounded to 4294 because the pinned native frequency getter returns
  uint Hz; other integral-MHz domains are bounded to 1000000. Invalid units,
  booleans, nonpositive frequencies and unsupported compound grammars are refused.

Scalar L1/shared latency fields use core cycles. `gpgpu_l2_rop_latency` and
`dram_latency` are also **core-cycle ready timestamps**, polled by the L2/DRAM
paths; owner names alone do not define their unit. The compiler records that
clock-domain distinction. Native DRAM controller MHz is separate from
`dram_data_command_freq_ratio`; neither is automatically a DDR data rate or
vendor Gb/s. `gpgpu_simple_dram_model=1` is an explicit fixed-service assumption
and bypasses detailed dram_t command counters. Mode0 selects the already
implemented detailed controller. Both retain unknown physical timing. Pinned
sources: `gpu-sim.cc:init_clock_domains/next_clock_domain`,
`l2cache.cc:memory_partition_unit::*dram* / memory_sub_partition::push`,
`shader.cc:shader_core_config::max_cta`, `addrdec.cc/hashing.cc`, and
`intersim2/networks/fly.cpp` at the engine manifest's GPGPU-Sim SHA.

Use `typed_trace.create_request(..., adf=scene)` with the full original typed
trace, then the existing installed API or `archexplorer evaluate` route. A
synthetic trace must retain `kind='synthetic', gpu='none'` and explicit toolchain
and description; a fact-constrained ADF does not turn it into captured execution.
For a pure-CPU scenario-building command:

```sh
python -m arch_explorer.components.accelsim.device assumptions.json historical-preflight.json --output scenario.json
```

No source metadata, successful 110-SM run, or matching API/CLI cycles makes
`hardware-accuracy` pass: same-device cache/network/DRAM/clock and typed-bank
parameters remain unvalidated. Historical preflight evidence is not a live
resource lease. This interface does not authorize device queries or execution.

The source distribution also includes `tools/accelsim_native/calibration/`: pinned public microbenchmark provenance, independently authored new candidates, offline build/static-byte checks, a CPU-only dry-run, conditional service-observable solver and replay. Its raw-record schema and source-reference JSON are shipped with the sources. See that directory's README for explicit budgets and blocked capture/physical-evidence prerequisites. Empty observations remain blocked; synthetic solver exercises are not hardware calibration or a target-specific coefficient table.


## Finite microbenchmark originals and timing

The installed `arch_explorer.components.accelsim.microbundle` module accepts the
versioned SIM30 registry: twelve complete, unrolled dep/quad/chase functions,
one CTA of 32 threads, three fit lengths and one interior timing holdout in each
family. It checks complete ELF instruction bytes, offline resources and an
independent unsigned input/output oracle. This finite registry grants no device
execution permission. Unknown functions, altered input bytes and enlarged budgets
are rejected.

`microcapture` consumes versioned raw SIM30NV1 records plus the lossless original
instruction dictionary/table, actual allocations, outputs and process evidence.
It reconstructs ordinary typed trace v7 through the same renderer as SIM28.
Dynamic global addresses come from the recorded MREF channel and must match an
independent per-lane oracle; no addresses are invented from a shape.
`microrecords` owns raw observation semantics, including actual host/preflight,
loader maps, event handles and float32 milliseconds/bits/nanoseconds closure.
`microexperiment` consumes these original references for fit-only conditional
service intervals. An arbitrary dictionary labelled `hardware` is insufficient.

A CPU-only installed check is:

```sh
python -m arch_explorer.components.accelsim.microcampaign check /absolute/bundle/REGISTRY.json
```

The explicit `run --design ... --handoff ... --output NEW_DIR` operation additionally
requires a newly coordinated handoff, exact installed source/host/tracer identities,
the common lock, no external compute, and all actual functions live before launch.
It is a bounded measurement operation, not a default test. Its append-only journal
freezes all initial native predictions before fit timing, and final intervals and
unchanged native assumptions before heldout timing. Capture and uninstrumented
event timing run in separate processes. Failures retain all originals and the full
attempted/not-started denominator; there are no warmups, retries or replacement
samples. The native engine remains responsible for all simulated cycles.

Event resolution is approximate, and three samples do not establish an error
bound. Service slopes remain conditional composite observations: unknown clocks,
issue mechanisms, caches, queues and instruction-cache effects prevent a physical
L/I or L1/L2/DRAM decomposition. Such intervals are never written into an assumed
native profile as measured parameters. No full-model accuracy claim follows.


The launch environment and the in-process environment are separate observations.
NVBit initialization may remove `LD_PRELOAD` before its helper CPU tools run. A
capture reader accepts only that removal when the exact frozen tracer file is
mapped both before CUDA initialization and before host close; all other variable
changes remain errors. Uninstrumented timing has no such allowance. Both original
environment records remain intact. A failed observation window stays closed even
when a later CPU reader correction can consume its already captured originals.
## Finite measurement outcome replay

`python -m arch_explorer.components.accelsim.microclosure DESIGN WINDOW
--evidence CONTEXT --reader-source SOURCE --output RESULT` is a CPU-only reader.
`CONTEXT` and `SOURCE` are JSON file paths. It never authorizes a device window.
The result separates `replay_status`, `collection_outcome`, every registered
attempt, completed stages, postclosure software recovery and unknown hardware
accuracy. A successfully read failed collection remains failed.

Historical `arch-explorer.sim30-outcomes.v1` needs an explicit closure context
containing hash-bound design/source, journal, observations and existing resource
reaping evidence. New producers write v2 with that same context embedded by
reference. Unknown versions/states, missing/duplicate attempts, orphan scoped
observations, substituted freezes, foreign runs and contradictory cleanup are
rejected. The reader only scans declared campaign roots and the supplied view.
An unrelated sibling directory cannot affect replay. Resource proof consistency
is historical evidence and conveys no current reservation or permission.

Both production `Collector.execute` and installed replay use `microclosure`;
Journal's production methods also replay the complete phase sequence. Prediction
freezes are revalidated against complete native protocols and fit-only inputs.
Fit and heldout observations are closed in both directions to actual process,
start, return and raw-event originals. No new timing can be inferred from a
postclosure native result or from a successful software replay.

`python -m arch_explorer.components.accelsim.microrehearsal --source SOURCE
--bundle REGISTRY --template IMMUTABLE_DELIVERY --output NEW_DIRECTORY
--mode success` drives the same phase orchestration using an explicitly
synthetic process/device boundary from the frozen source archive. Modes
`first_failure`, `timeout` and `quality_refused` stop without replacement samples.
This test command creates no process, device event, GPU lock or real lease and
does not run the native engine. It reads unchanged completed native protocols as
fixtures. The production hardware replay rejects synthetic source/marker roots;
the rehearsal result is always labelled synthetic, with physical parameters
unset and hardware accuracy unvalidated. The approximate event resolution and
conditional interval assumptions remain unchanged.

New finite campaigns use `arch-explorer.measurement-registry.v2`. The registry
declares the work item, compiled function prefix, per-program integer oracle and
bounded budget. Designs, handoffs, leases, receipts, freezes, journal and outcomes
carry that identity with versioned measurement schemas. Stable raw CUDA/NVBit
transport formats remain v1; their version does not identify an experiment.
Historical SIM30 records are still read with their original, bounded contract.
Unknown versions, foreign work items and old handoffs cannot open a new campaign.

New registry rehearsals also require `--native-fixture PLAN`: an explicitly
synthetic plan containing completed public protocols for every new full program,
plus the exact registry, engine, host, tracer and history references. Renaming an
old prediction cannot satisfy it. Complete new static bytes, dynamic addresses,
oracles and synthetic events pass the same parsers used for collection.

Closed new campaigns attach `native_timing_comparison`: every registered fit and
heldout row, original float32 event bits and units, assumed native clock/time,
signed and absolute errors, and summaries with explicit excluded denominators.
Relative errors are fractions (0.10 means 10%). Missing or refused timing stays
visible and never becomes zero error. Conditional composite intervals are
separate from native predictions; physical parameters remain unknown.

`python -m arch_explorer.components.accelsim.microguard selfcheck --root NEW_DIR`
checks CPU descendant stopping/reaping without a GPU lock. Its `run --design
DESIGN --handoff HANDOFF --window NEW_WINDOW --root NEW_SUPERVISION` entry requires
the frozen installation and a fresh actual handoff with an absolute deadline.
It reserves cleanup time, bounds the worker by both wall and monotonic clocks,
and records owned PID/start-tick identities and the final lock recheck. The
handoff deadline never moves when the common lock is later acquired.
