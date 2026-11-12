---
name: arch-explorer-component
description: 为 ArchExplorer 接入或复用计算、存储、互连、有状态组件及外部模拟器provider，通过公开工厂或独立扩展包验证创建、组合与执行能力。
---

# 组件与执行器接入

读取匹配版本的 `docs/component-extensions.md`，普通组件参考 `examples/extensions/synthetic_components/`；状态模型再读 `docs/unified-execution.md`。对外类型从 `arch_explorer.components.api` 导入，使用公开mapping接口，不直接调用私有scheduler实现。

ADF是组件实例、参数和连接的描述，ComponentType是类型声明，build返回每次构建的运行模型。组件应只认识自己的请求与能力，不依赖完整模型名ViT/CLIP。

先运行 `archexplorer components` 并读 `docs/component-implementations.md`，按硬件职责找到
组件和已有实现。ComponentType拥有硬件身份、共同参数及结构规则；新增已有硬件的执行方式用
`components.api.ComponentImplementation/register_component_implementation`，不另建硬件身份或provider。
ADF的ImplementationBinding精确记录实现ID、版本和带单位/来源的实现参数，与共同参数分开。
目录、CLI和Agent均消费公共component_catalog；不要维护实现白名单或按来源分支调度。

SM的roofline要求显式F/B/P/BW及component_io边界，普通Kernel不能推断B；原生SM当前是
含L1/shared的cluster，不因此宣称独立L1工厂。检查每实现自己的必需参数、执行域和输入。
新增计算-only实现可复用外部memory/network搬运；roofline已经计入的同笔IO不能重复收费。
独立可安装示例见 `examples/extensions/component_implementations/`，论文能力边界见
`docs/component-inventory.md`。实现参数/版本必须保留于序列化、fingerprint和来源重放。

公共编译示例经 `compile_graph` 和显式组件关系选择适配器，不复制模型/编译器或新增provider。
实现的adapters声明可复用适配器；唯一兼容项才可自动选，显选使用组件路径到AdapterBinding的字典。
新增转换公开注册 `TaskAdapter`，lower消费组内任务并返回 `AdaptedGroup`，restore仅恢复组件输入。
复用已有执行域；新域才注册 `AdapterDomain`，compose保留同一公共图。组件模型独占时延/资源公式。
读取公共mapped Workload，不调用内置适配器私有lower；旧CompilerTarget入口明确退役。
组件输入/计费边界必须兼容；
roofline component_io已计的流量不能在相同边界再次交外部DRAM/ICNT收费。
详 `docs/compilation.md`，不对任意硬件/目标宣称自动桥接。

## 选择执行边界

新增运算若只需要已有组件能力，先走 `docs/operator-implementations.md` 的语义/目标执行注册。
OperatorDefinition拥有数学规则与reference，ImplementationDefinition拥有lower/support；
普通多步/临时量由公共graph策略处理。不要因此给组件、provider或调度器添加算子/模型名分支。

1. **资源需求模型**：ComponentType声明ParamSpec、Conventions和能力；build返回ComputeModel/MemoryModel/NetworkModel等。计算kernel_demand、访存access、网络route给出Demand，由框架处理依赖和争用。已有基类能表达时无需自写provider。
2. **有状态模型**：内部队列、反压、流水线或状态相关延迟用submit/next_event_time/advance。按文档绑定每次评估的新实例，共享控制器使用同一实例。拒绝submit不能有副作用；consumer_ready、physical_complete、resource_ready、queue_available分别表示不同事件，不能互相替代。
3. **独立引擎provider**：完整引擎或必须联合执行的组件组实现prepare_request/assess_request/evaluate_request，声明id/version、组件范围、single或joint执行域、请求和结果schema。prepare不执行仿真，assess明确拒绝能力缺口，evaluate保存完整请求/来源/结果绑定。

前两项是组件行为表达；provider是执行域边界。外部引擎若能满足状态事件合同，也可通过状态模型接入。
动态组件网络还需要请求ID/父子关系、端口、响应匹配、队列credit、时钟域/同时间phase及排空合同；
先确认匹配版本提供这些公开能力，不能仅凭submit/advance推断支持，也不能私下推进其他组件。

## 注册与验证

直接复用已有 Accel-Sim 原生组件时，先读[独立调用教程](../../docs/standalone-components.md)
及匹配版本的公开工厂合同。[外部调用程序](../../examples/gpu_instantiation/README.md)
分别消费 ICNT 和内存静态库；它们不是新 provider 或替代机制。主 wheel 先安装，公共
`native_sdk.load_sdk/compile_consumer/run_consumer` 验证 SDK 并在仓库外链接执行；
保留构造审计、公共收据及安装前后身份，断言实际执行通过后才记录复用结果。

新增硬件类型用 `register_component`；已有硬件的新实现用 `register_component_implementation`；
新增执行域provider才用 `register_provider`。这些与workload的ImplementationDefinition不同。独立包声明入口点：

```toml
[project.entry-points."arch_explorer.components"]
my_extension = "my_extension:register"
```

无参数register()只注册描述/工厂。用discover_components()检查loaded和failures；失败扩展不应隐式启用替代模型。

参数有单位与来源，validate拒绝越界和不支持的精度。组件拥有执行资源；mapping拥有拆分、放置、搬运与存活。状态执行已拥有的资源不能再由外层Demand重复收费。

仓库外安装后，构建一个实例并执行最小支持请求，检查资源占用与预期事件顺序；再验证并发/拒绝或另一参数配置。实际运行与重放都要检查。明示未建模的存储、数值、功耗或物理精度，null不填0。

Accel-Sim 的完整 joint 路径与组件实例化路径分别核验。采用组件路径前读取
`docs/gpu-instantiation.md` 和 `docs/accelsim-componentization.md`，核对实际 ADF 端口连接、
组件独立驱动能力及通用路由/时钟证据。禁止把各组件分别计时相加，或把整引擎 run 包装成每个组件。
L2/内存应能由非SM请求源驱动，通用协议不暴露mem_fetch指针或GPU全局上下文；内部原生依赖可以留在适配器。
精确整数差分、反压无副作用、事务不丢失/不重复及system_drained均需实证；不能由总时间相同推断。

复用已有原生组件时，核对公开工厂是否真正创建并拥有实例、局部状态和销毁，而非从整机借对象。
读取该版本的服务依赖与生命周期合同；内存/互连调用方只提供其配置与必要服务，不应创建GPU宿主、
SM/kernel/CTA环境或闭包捕获整机。组件与通用Runtime各自保有权威，用户请求源只负责流量和响应记账。
同进程两套不同配置交错推进、销毁一套后另一套继续，才可说明实例隔离；两个进程不能替代这项验证。
GPU重新组装必须使用同一工厂。结合构造调用链与链接隔离/构造探针核验；单个自报零计数不足以证明。

前台完成不等于物理排空；active=false或exit=0后仍可能有在途队列/DRAM工作。保留原前台
foreground_final，再推进真实尾部到排空，单列drain_ticks/drain_core_cycles/final，不改前台
kernel统计、不伪造新kernel或清空队列。新component-engine主延迟取物理结束并标注
physical_system_drain，前台和最终计数分别验证；预算覆盖前台与尾部，重放不执行finish。
