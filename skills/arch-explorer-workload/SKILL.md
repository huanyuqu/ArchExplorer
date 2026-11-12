---
name: arch-explorer-workload
description: 为 ArchExplorer 接入新的算子或负载图，编写独立可安装扩展并在自定义ADF上验证mapping与仿真。适用于新增计算语义或图结构，不要求每个新模型重写硬件组件。
metadata:
  id: skill_1b068ddcfd93
---

# 新负载接入

先找用户安装版本对应的 `docs/operator-implementations.md` 和
`examples/extensions/operator_implementations/`。兼容单lower/外部row策略另见
`docs/workload-extensions.md`。阅读公开API和实际命令后编写扩展，不修改主库模型名单，
也不把自动PyTorch/ONNX导入或任意ISA支持作为已有功能。

区分三类工作：
- 新图由已有算子组成：定义张量、算子与依赖，直接复用语义和执行模型。
- 真正新算子：定义shape/dtype/属性、语义与数值参考，再注册所需目标的执行实现。
- 新硬件机制：另接组件执行模型；同一算子可以有多种硬件实现。

## 公共语义接口

扩展合同位于 `arch_explorer.workload.api`：语义/图类型、OperatorDefinition、ImplementationDefinition、
ExecutionProgram/ExecutionStep、注册与选择接口。先确认匹配版本实际提供这些符号；缺失时报告，
不能绕过校验或猴补白名单。

OperatorDefinition拥有validate、version和可选reference(op, inputs)，输入是只读ndarray；
实现定义拥有稳定ID/version、operator、target、requirements/dtypes/supports及lower(op, context)。
语义不随硬件速率改变，attrs按算子自己的规则验证。旧三参数lower是legacy:<op>兼容适配，
不是同一语义只许一个实现的限制。

`Workload.to_dict()/from_dict()`保存图；`WorkloadSpec('graph', graph.to_dict())`表示已有图。可选注册builder用于参数化图生成：`register_workload(kind, builder)`，builder(params, *, tp=1)返回Workload。自定义kind只是生成图的便捷入口，非每个新图的必选步骤。

普通单设备events图优先用execution_graph：按节点指定implementations，选择first/round_robin、
serialize及可选memory。复用图依赖/放置/存储基础，不复制run_device。两个兼容实现存在时必须
消除歧义；显选不存在、目标/能力/dtype/shape/layout不兼容都应在准备输出前拒绝。
旧GSM/DMC适配保留各自限制，不能因算子未知静默跳过。复杂布局/切分可扩展专门策略，
固定tile或未试跑的选择不能称最优。

具体schema、图输入输出及拓扑规范以公开教程为准。保持分支、共享权重、精度与依赖；模型中无依赖的部分才可并行。拒绝未知引用、重复生产者和依赖环；不把一张图拆成加权独立workloads来绕过图限制。

## 实现和接入

涉及不同目标共同编译表达时，使用 `workload.api.compile_workload/validate_compilation`
及 `simulator.api.prepare_compiled`。`compile_graph_v2` 的region计算、索引、K循环、
tile张量/assembly、访问与inspection由注册OperatorMapping回调拥有；公共编译器只编排，不在用户脚本复写。
公共任务通过按组件路径的 `AdapterBinding` 选 `TaskAdapter`，由其lower/restore负责组输入转换。
`ImplementationDefinition` 仍是算子lower，`ComponentImplementation` 仍拥有模型工厂和计时资源。
公共mapping的v3包含compute/memory/rules/operator_rules；逐实例覆盖优先于kind默认和唯一规则。规则ID/version/options由各规则验证，专参属于适配器options，执行选项另传prepare。
同执行域支持的组保留共同依赖/最后读者，缺跨域桥时拒绝，不独立计时相加。示例及FP32/连续matmul边界见
`docs/compilation.md`和`docs/attention.md`。新编译是v3，v1/v2显式迁移；保留显式输入和原DAG两个读者，不把图压成FLOPs后猜指令。
公开编译支持连续物化FP32/FP16/BF16 storage、FP32 accumulation和逐物化输出舍入；复用workload.api.round_storage，不在扩展/用户脚本复制舍入实现。
完整pre-LN block由公开transformer_encoder_block构图；多层用transformer_encoder和compose_workloads。CORE场景只生成显式数据/配置并调用公共函数，不预计算层间activation。默认reshape/transpose物化；layout_candidates给同一图的materialized/views完整mapping，不能共享的reshape显式保留物化，不自己计算strides或复制假装view。
同dtype只读view保留位值，不新加舍入；不同dtype需cast。resolve_layout给根storage、元素strides/offset和完整allocation_bytes，BF16 host FP32数组仍按2B存储计费。只读正步长slice/transpose/可共享reshape需适配器声明readonly_strided；旧independent-only外包在lower前明确拒绝别名任务，不冒充具备strided能力。任意as_strided、零/负步长、dtype重解释、alias写入仍拒绝。
当前设备仍声明FP32算术服务，低精度promotion/rounding另计转换需求，不猜加速倍率。view没有复制但真实读者仍付访存；根只分配一次，小输出view可能保留完整根。读写服务看access，分配看allocations，运行峰值看真实分配/释放，不用逻辑张量求和替代。旧非编译范围保持；native精度/ISA另经公开assessment，本轮不新跑native。
精确输入、norm/GELU定义及场景见docs/precision-transformer.md和examples/transformer_block/README.md；未知扩展保持自己的精度声明。
多层布局比较和十二预算配置见docs/multilayer-encoder.md及examples/multilayer_encoder/README.md；单布局控制六预算。保存的有效规则、root布局和resolver/builder/adapter来源必须严格校验，不能靠替换字段迁移旧已绑定结果。
完整公共ViT入口见docs/vit-compilation.md及examples/vit_compilation/README.md；只传图像与完整根参数，构图/布局/数值使用公开workload API，不复制patch或GELU公式。ViT保持erf精确GELU、epsilon=1e-12及其声明的存储边界，不能把tanh encoder改名替代；旧ViT/CLIP原生结果不迁称公共编译结果。布局/逐实例mapping/硬件比较走同一既有搜索器，未知外包保留其FP32/连续能力范围。
完整公共CLIP入口见docs/clip-compilation.md及examples/clip_compilation/README.md；同一图保留图像塔、causal文本塔与两个矩形相似度输出，只供图像/token IDs/完整根参数。QuickGELU、epsilon=1e-5、first-argmax EOT、无epsilon的L2与FP32 logit_scale顺序归公开语义，不能套ViT/tanh块或移动低精度舍入边界。token/EOT保持int32，不截断/浮点化；外部非法NaN/inf不能借内部mask sentinel放行。查composition身份和layout_candidates，不猜塔/层名字或复制索引流量/布局公式。输出共享根的保留、全embedding容量与重复索引访问分别核对；新BF16存储不宣称旧CLIP或官方混精度等价。
探索仍调用既有Env/search；compilation配置接公开编译流程，每个候选尝试计预算，保留失败；保存compiled-result作为只读依据。
新编译搜索结果replay不执行；需要新执行时显式rerun。旧ADF结果保留原再评估行为。

从独立示例复制最小所需部分到用户包，使用自己的命名空间。入口点 `arch_explorer.components` 的无参数register()可注册算子、builder、mapping与组件。注册只声明能力，不运行仿真。

已有Kernel足够时复用；多步实现返回有显式deps、内部Tensor与输入/输出的ExecutionProgram。
临时量容量、最后消费者和释放走公共存储计划，分支临时量由框架隔离。特殊策略才扩展mapping；
无法表达的操作明确拒绝，不伪装成matmul或填固定时延。仅计算型示例明示没有存储网络模型。

GPU目标使用组件公开emitter合同注册target=accelsim的实现；图/输入/reference仍属于workload或
外包。GPU provider不识别新模型名，不能改kind/metadata假装ViT/CLIP，也不新建专用provider绕过。
先核FP32/ISA/单设备/拓扑等真实限制；native组件路径与oracle使用同输入、独立输出和公共比较器。

## 证明外部可用

安装主包与扩展wheel，离开源码目录，使用实际公共API和CLI运行。至少覆盖用户需要的新图与算子、一个可信小输入、mapping变化及硬件参数变化。检查变化是否进入真实任务，不能只修改报告标签。检查不支持的dtype/shape/mapping会在执行前拒绝，并保存可重放结果。

数值参考与时序/资源检查分别保存证据；性能结果不证明张量值。绑定实际语义/参考/执行实现的
版本和源码、mapping及输入字节，插件变动后旧结果只读重放应拒绝。核注册冲突及失败回滚，
不能静默换实现。TP/多设备/外部引擎能力分别确认，不对任意新图默认支持。
native前台/物理尾部分别比较，主延迟取物理final，尾部不是新kernel；只读重放不执行native。
交付外包、ADF、图、映射、来源和可执行命令，不修改通用scheduler或CLI的身份分支。
