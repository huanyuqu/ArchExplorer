---
name: arch-explorer-run
description: 使用 ArchExplorer 在用户指定架构上准备、运行、比较与重放负载仿真，解释结果范围和失败原因。适用于现有组件及负载；新增算子或组件时先查相应扩展接口。
---

# 使用 ArchExplorer 做一次可解释的仿真

从用户的架构文件、负载和目标问题开始。ADF 描述硬件；负载图描述计算依赖；mapping 描述切分、放置和搬运；组件拥有执行机制；simulator生成执行计划。`dmc_adf`是示例描述工厂，不是用户必须继承的架构类。

## 找到安装与输入

在用户环境运行 `python -m pip show arch-explorer` 和 `archexplorer --help`；确认实际导入路径。若未安装，使用用户选定的发行包或checkout。不要靠修改PYTHONPATH掩盖未安装扩展。项目版本号不能代替能力检查。

运行 `archexplorer components`，读取硬件类型下的实现版本、来源、输入与执行域；Agent使用同一
component_catalog。ADF显式绑定实现及其独立参数，不按组件名任选第一个实现。SM roofline是
有明确IO边界的F/B/P/BW解析服务，Accel-Sim SM是包含L1/shared的原生cluster；输入不可互换，
域不兼容不得fallback。见 `docs/component-implementations.md` 及 `examples/component_implementations/`。
新增已有组件实现可后装外包，不应修改CLI/provider名单。组件功能与限制见 `docs/component-inventory.md`。

定位用户的ArchExplorer源码分发或文档目录（下称项目目录），按需读取：
- `docs/tutorials/03-architecture.md`：ADF文件、参数、单位、拓扑。
- `docs/workload-extensions.md`：新图及已安装的负载扩展。
- `docs/component-extensions.md`：组件/provider能力。
- `docs/models.md`：已实现范围。
- `docs/gpu-instantiation.md`：任务涉及 GPU 组件实例化或与原版 Accel-Sim 差分时读取。
若仅有wheel而缺文档，使用匹配发行的公开文档；不猜接口或声称缺失能力已存在。

保留用户给定的shape、dtype、布局、批次、精度转换和mapping。未指定影响结果的参数时，明确假设或询问；教学合成参数需标来源。硬件参数包含值、单位、来源：ns、GHz、B、B/ns，不把时钟数当ns。

## 使用公共入口

需要保留共同计算/映射表达并生成不同目标输入时，读 `docs/compilation.md`。
使用 `archexplorer task-adapters` 读取公开目录，再用 `archexplorer compile specs.json --output program.json`，
再 `archexplorer prepare program.json --output <new-dir> > public-request.json`，
随后公共 `evaluate` 执行/只读重放。Python 对应 `workload.api.compile_workload`、
`simulator.api.prepare_compiled`；使用显式合法tiles和按组件路径的AdapterBinding。
适配器专参写bindings[path].options，GPU线程数不进入公共mapping；执行provider的options另传prepare。
组件只声明唯一兼容项时可自动选择，歧义/不支持须拒绝；不新增target名单或静默回退。
不能把普通Kernel/trace、计数总量和公共编译图互换，不能修改编译产物字段后重标旧结果。
编译/prepare不执行native或mapping搜索；实际执行、数值参考与硬件精度分别取证。
未知外包公共编译默认例为 `python -m ae_component_implementations.compilation`，
先装匹配主库后装独立外包；精确命令在该包README。

已有场景JSON包含 `adf`、`workload`、`mappings`、`provider`、`options`、`scope`。按选中provider的文档检查能力，不擅自替换引擎。

```sh
archexplorer demo prepare --specs scene.json --output run-001
archexplorer demo run run-001
archexplorer evaluate run-001/mapping-000/result.json --output replay-001.json
```

选择新的输出目录，保留生成请求及其引用文件。prepare只准备，不代表仿真通过。更改输入后重新prepare；遇失败或已尝试目录，先解释原因，再按用户意图决定是否新建一次运行，不无限重试。

Python对应入口在 `arch_explorer.simulator.api`：prepare_request、assess_request、evaluate_request。先检查 `supported` 与 `gaps`；options仅None默认空对象，其余必须是对象。

完整GPU程序需要明确的外部引擎及其能力。CPU事件模型的结果不能替代GPU结果；安装一个provider也不意味着它支持所有负载或跨引擎组合。

新图/算子按 `docs/operator-implementations.md` 选择公开目标实现，必要时先完成外包安装。
缺实现、多个兼容实现的歧义及显选不兼容是准备错误，不通过改模型kind、换provider或回退来消除。
保留实际语义/reference/执行ID版本与输入来源；事件时序、数值参考及native比较分别解释。

若任务是 GPU 组件实例化，核对所选版本是否真正通过公开端口连接 SM/cluster、互连、L2 和 DRAM，
并由通用事务/时钟框架推进。仅将同一完整引擎跑两次不是组件实例化。按匹配版本教程构造
`accelsim_components` 场景和独立 `accelsim` oracle 场景；不在 CLI 或脚本中重写 native 机制。
能力或构造接口缺失时明确报告缺口，不通过换名或 fallback 填补。

任务涉及非GPU场景复用L2/DRAM/ICNT时，先核对匹配版本的独立创建工厂、最小服务和所有权合同。
请求源只使用公共事务/端口，不创建或借用GPU整机实例；响应恰好完成与物理排空分别检查。
实际运行同进程不同配置实例的交错推进和独立销毁，保留输入、响应、服务/组件生命周期及包来源。
独立构造/复用和GPU原版行为差分是两组证据，任一通过不能替代另一组；准备或编译成功也不代表完成。

## 比较和交付

非 GPU 原生复用可使用[用户侧双实例调用](../../docs/standalone-components.md)。分别选择
匹配的 SDK manifest 和 ICNT/内存 archive target，按示例运行 `standalone.py`；保留外部调用收据与原生统计，
另用 GPU 公共比较器消费行为证据，不把两类结果互相代替。

绑定结果逐项核对架构、负载、mapping、provider版本及实际任务变化。报告latency_ns、资源/计算/传输范围、参数来源与未建模指标。unknown/null不补0，不把两独立子图时延相加当完整图仿真。

区分两类“重放”：provider结果的evaluate应验证保存的结果且不重新执行；search/agent结果有各自的重评估或回放协议，先检查schema，不能一概宣称零执行。

GPU 软件差分使用组件包的公共比较器，保留两侧版本/manifest、共同 trace/配置/mapping 和 scope。
检查逐 kernel 及整体计数、完成顺序、已声明机制指标与物理排空；缺值不补零，不以总时间相同
替代机制一致。不匹配时保留原件和具体差异；相同原生语义下的整数差分不靠拟合或百分比容差通过。

新 component-engine 的主Evaluation.latency_ns表示物理排空，标注physical_system_drain；
details.native的gpu_tot_sim_cycle/latency_ns(_exact)是前台，system_drained_*是最终值。
foreground_final与尾部drain_ticks/drain_core_cycles分开解释，两项差分均须通过。时钟tick不等于
core周期，尾部不是额外kernel；不可强求物理总计等于逐kernel合计。退出0或active=false不足以
证明排空，未排空仍失败。旧engine.v1/v2保留原延迟合同，重放不重新执行finish。

交付可复现输入路径、实际命令、结果路径和失败/限制。合成事件时间、软件功能通过、原生程序运行和真实设备精度是不同结论。无需下载权重或启用物理GPU来完成纯CPU示例。
