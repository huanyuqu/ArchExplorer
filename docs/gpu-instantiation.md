# 从组件实例化 GPU 并对照原版 Accel-Sim

将主库未知的新图/算子接入已有 GPU 目标，使用[外部目标执行实现](operator-implementations.md)。
模型图、输入和数值参考属于 workload/扩展；GPU provider 消费公开执行产物与能力，不按模型名准入。
下面的 ViT/CLIP 是已有用户场景，不能通过改名或伪装它们来接新算子。

在非 GPU 场景直接创建原生组件，参见[独立组件调用](standalone-components.md)。
独立创建、最小服务和同进程隔离需要另外的实际证据，不能仅由本页的 GPU 行为比较推出。

本流程的目标是用 ADF 中的 SM/cluster、互连、L2 分区和 DRAM 实例组成 GPU，并由通用框架
连接、路由事务和推进时钟。`accelsim_components` v1 是组件路径，`accelsim` v1 保留完整
原版引擎路径作为 oracle。provider 的名字本身不证明组件化；应核对实际事务边界和运行证据。

本页随组件实现接入；执行所需的新 native 引擎与公开 wheel 分开提供。
构造和准备成功不代表原生差分通过，运行后须检查公共比较报告。

## 输入与执行边界

先读用户的 ADF 文件，再读负载和 mapping。ADF 是组件实例、参数及连接的描述，不是一个
必须继承的 GPU 类。组件拥有缓存、发射、队列和控制器机制；框架拥有事务所有权、端口路由、
时钟相位和生命周期。具体 native 配置由组件包的公共构造接口生成，不在用户脚本中复制。

两侧消费同一个新组件引擎 manifest，并分别选择组件模式和保留的原 cycle 模式。
完整输入包括引擎 manifest、组件参数/连接、工作负载、mapping、程序或 trace
及其来源。两侧输出使用独立的新目录，保存原始命令和执行产物。组件路径先运行，再产生 oracle
结果，避免从现成 oracle 输出填统计；共同输入的等价仍由公共比较器核验。

小 ViT 和 CLIP 使用完整前向结构。CLIP 包含视觉/文本双塔、投影、归一化和最终输出；
两种 mapping 必须对应真实程序切分或执行几何，不能只改标签。参考执行生成的动态指令/地址流
是合成 reference trace，不是采自官方框架的 CUDA trace，也不包含真实权重。

trace 固定了指令、地址和同步流。改变数据依赖路径、地址生成或同步语义时应重新准备输入，
不能把同一 trace 任意外推到所有新架构。组件化对照通过只说明已覆盖软件行为一致，不是硬件精度。

## 准备与实际执行

安装匹配版本的库和独立 native 构建产物；公开 wheel 不携带第三方源码、native 二进制或 trace。
原生模拟在 CPU 上执行，不需要开启 GPU 测量或下载权重。

从[用户示例目录](../examples/gpu_instantiation/README.md)执行：

```sh
python workflow.py create --model vit --engine /ABS/NEW/engine.json --output cases/vit.json
python workflow.py topology cases/vit.json --output cases/vit-topology.tsv
python workflow.py run cases/vit.json --output runs/vit
python workflow.py replay runs/vit --output replay-vit.json
```

CLIP 使用 `--model clip` 和另一组新输出路径。每份 case 含 32、64 threads/block 两个
真实程序 mapping；helper 先准备并评估两侧全集，再顺序调用通用 `demo` 接口执行，最后调用
组件包的比较器。可以把 `case.json` 的 candidate/oracle 对象分别保存为 scene JSON，使用
`archexplorer demo prepare --specs scene.json --output ...` 和 `archexplorer demo run ...`。
这两条通用命令成功后仍须调用比较器；它们本身不宣称两条路径相等。

公共 Python 构造与连接接口：

```python
from arch_explorer.archs.gpu_instantiation import case_specs
from arch_explorer.components.accelsim.componentization import gpu_adf, compile_topology

adf = gpu_adf(sm_clusters=4, memory_partitions=8, l2_subpartitions=2)
topology = compile_topology(adf)
case = case_specs('vit', engine='/ABS/NEW/engine.json')
```

`gpu_adf` 是原生配置和连接的唯一权威。本版显式选择单 chiplet、LRC 关闭；不支持的子集
会拒绝。`SpaceMatrix.connections` 使用通用 `adf.Connection`，五字段为 `source`、
`source_port`、`destination`、`destination_port` 和 `protocol`。例如：

```json
{"source":"component_gpu/cluster[0]/sm","source_port":"request","destination":"component_gpu/icnt","destination_port":"request0","protocol":"transaction.v1"}
```

端点是展开 repeat 后的物理路径，repeat>1 用 `[i]`。六类有向连接是 SM→ICNT、ICNT→SM、
ICNT→L2、L2→ICNT、L2→DRAM、DRAM→L2。cluster 自带 L1/shared；框架仅消费端口、
事务和时钟。`compile_topology` 拒绝缺线、多线、错线。已有未连接 ADF 可交公共
`connect_gpu(adf)` 复制并连接；它不会覆盖已有连接或修正不支持的参数。

`--hardware hardware-async.json` 使用示例中的详细 DRAM 与四域异步时钟。case 暴露实际
硬件 JSON，修改后两侧须相同。是否形成跨 SM/内存压力取决于实际机制统计，不能从频率不同
直接推断。接口或引擎缺失时停止并保留原因，不把 joint provider 改名继续执行。

native 运行使用有界 timeout、内存和周期限制，按记录的 CPU 预算串行执行；实例教程的基准
每个 CPU native 子进程组最多 600 s、8 GiB，预算包含 finish，构建只用一个 job。
可以进一步收紧预算；正常结束与失败均须等待并回收整个进程组，不无限重试。
本流程不恢复历史已取消运行，也不采集新的 GPU trace。

## 用公共比较器核对

比较逻辑归组件包的公共接口，用户流程仅组织结果和展示报告：

```python
from arch_explorer.components.accelsim.componentization import compare_results
report = compare_results(component_result, oracle_result)
assert report['schema'] == 'arch-explorer.accelsim-component-comparison.v1'
assert report['status'] == 'passed'
```

比较器先只读核验两份完整结果及其原件，再验证：

- 两侧精确版本、补丁/构建身份、配置、共同程序/trace、输入和 mapping；
- kernel 集合与顺序、整体与逐 kernel 周期、退休指令和 CTA；
- 缓存命中/未命中、DRAM 读写、互连传输、反压及异步时钟等已声明机制；
- kernel 完成与 system_drained，不能把消费者就绪误当物理排空。

整数计数和完成顺序要求精确一致；ns 使用相同源时钟换算。缺失指标、截断、未知 ISA 或没有
排空须失败或明确不支持，不能填零、拟合或用宽泛百分比容差放过。

### 前台完成与物理排空

新 component-engine 的 `Evaluation.latency_ns` 表示物理排空时刻。组件包根据
`final.total_cycles` 和源 core 时钟计算它，用户脚本只复制结果。结果中的
`details.native.gpu_tot_sim_cycle`、`latency_ns`、`latency_ns_exact` 保留原前台值；
`system_drained_cycles`、`system_drained_latency_ns`、`system_drained_latency_ns_exact`
给出物理完成值。helper 在 `run.json` 中分别展示，并标注 `latency_semantics=physical_system_drain`。

比较器保留进入尾部前的 `foreground_final`、尾部 `drain_ticks`、`drain_core_cycles` 和
`final`。前台逐 kernel/总计与尾部/最终机制的差分都必须通过。`drain_ticks` 是额外时钟推进
次数；`drain_core_cycles` 是真实累计 core cycle 的差，二者不可混同。尾部继续推进已有
物理工作，不增加 kernel、退休指令或 CTA，不追加 cache flush；有尾部时 breakdown 单列
`phase='physical_system_drain'`，不能塞入最后一个 kernel。最终物理周期不必等于逐 kernel 合计。

`active()==false` 或进程退出 0 不能单独证明排空。前台必须正常完成，真实队列、管线、响应和
资源必须归还；无进展、截断、总预算超限或未排空仍是失败。timeout/内存/cycle 预算覆盖前台
加尾部，不能在 finish 时重置。只读重放不执行 finish。旧 engine.v1/v2 延迟语义保持原合同；
不能把新 oracle 的显式尾部续跑解释成旧 runner 原本已物理排空。

输入/来源错配抛出错误；合法同输入但计数不同返回 `status='failed'` 与差异。
helper 保存完整比较报告，并将总运行标为失败。只有全部 mapping 的报告为 `passed` 才写
`run.json` 的 `compared`。不匹配时保留两侧原件、身份和 scope。不要通过改写请求摘要、拷贝 oracle 计数或改用户参数
来制造成功。若没有可合法复用的真实 CUDA 应用 trace，报告该覆盖未取得。

报告完整保留 `native`、`counter_differences`、`mechanism_differences`、`identity`、`results`
和 `evidence`。两侧 ADF、负载、mapping 集合和 options 在任何 native 执行前必须相同；
底层源码、输入和执行原件的真实绑定仍由公共比较器验证。

## 重放和结果解释

```sh
python workflow.py replay runs/vit --output replay-vit-again.json
archexplorer evaluate runs/vit/candidate/mapping-000/result.json --output replay-components.json
archexplorer evaluate runs/vit/oracle/mapping-000/result.json --output replay-oracle.json
```

provider 结果重放应只读核验原始请求、实现来源和执行收据，不重新执行 native。
保存两侧引用的原路径和文件；输入、配置或实现改变后应准备新目录。

最终交付同时保留公共比较报告、两侧原始结果、失败/未尝试项及可复现命令。完整小模型、
实采 trace、真实权重、框架数值一致性和设备时延精度是不同范围，分别说明。
用户 agent 可按[运行 skill](../skills/arch-explorer-run/SKILL.md)消费本流程；
组件作者的接口要求见[组件拆分边界](accelsim-componentization.md)。
