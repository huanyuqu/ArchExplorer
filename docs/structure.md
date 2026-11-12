# ADF 结构编辑与三层搜索

`arch_explorer.structure` 在共享 `SpaceMatrix` / `SpacePoint` 树上执行结构编辑。
调用返回新树，原树及其所有子对象保持不变。ADF 契约本身是可变 dataclass，调用方不应
修改已作为检查点或候选交给其他代码的树。每步编辑都通过组件注册表、参数 domain 和
网络角色检查；不合法的编辑抛出 `StructureError`，其 `to_dict()` 包含 code、message、details。
组件 evaluator、映射模板与事件后端由各自模块注册。

```python
from arch_explorer.archs.adf_seeds import dmc_adf
from arch_explorer.structure import split, replay

seed = dmc_adf(4)
result = split(seed, "chip/dram", 4)
assert replay(seed, [result.record]) == result.adf
```

运行前须注册相应组件；种子只描述硬件，不提供仿真模型。`dmc_adf(num_cores=4)` 包含
vector core、本地 SRAM、mesh 与 DRAM，参数来源为 `synthetic seed`，用于非 GPU 教学场景。
GPU 采用 [原生程序 demo](tutorials/05-full-models.md) 的 ADF，不使用吞吐公式种子。
CGRA 具名种子 `vectorcgra_4x4` / `vectorcgra_4x4_diag` 直接使用 `devices.get`，
保留其 RTL 来源、场景参数及 unknown clock 标记，不复制器件常数。

## 六个基本原语

路径是类型级路径，如 `chip/core/sram`。同一 repeat 元素的所有实例共享参数。

| 原语 | 参数和效果 |
| --- | --- |
| `split` | `point, num, algo="even"`。调用组件 `split`，原点成为同名容器，内含 num 个分点。父容器继承原点的 repeat。 |
| `merge` | `points, algo="sum", name=None`。调用组件 `merge`，合并同级、同种类型、相同 repeat 的点。可再 `remove` 去掉包裹层。 |
| `insert` | `parent, children, name="group", group_size=None`。将直接子元素放入新容器；网络不能作为 children。 |
| `remove` | `level`。子元素和网络提升至父级，各子节点 repeat 乘被移除层的 repeat；名称冲突时报错。 |
| `create` | `container, spec` 或 `container, component`，二选一。spec 为完整 ADF 节点 JSON；component 为已有路径，表示移动。网络自动进入 networks。 |
| `delete` | `component`。删除该路径的点或整个子树；不允许删除根。 |

`split` 仅允许 compute/memory 点；通信点不能变成 networks 中的容器。
extensive 参数依组件实现划分/相加，int 资源必须整除，clock 等 intensive 参数保持不变。
来源说明由组件的 split/merge 实现维护。根节点保留；合并/分割不能跨兄弟集合。

对 repeat 的分组使用 `insert(..., children=["chip/core"], name="chiplet", group_size=4)`。
当 core.repeat=16，结果为 chiplet.repeat=4、core.repeat=4。此表示等价于每四个 core
组成一组，不实例化 16 份重复 JSON。group_size 必须整除 repeat。需要分别编辑不同组时，
可用 create 创建不同名字的显式节点，再 delete 原重复节点。跨不同父级物理倍率的
component 移动被拒绝，以免静默复制或丢弃硬件。

## 高层程序

高层程序返回 `ProgramResult(adf, records)`，records 完全由基本编辑构成。
`replay(seed, records)` 可以独立展开，不需要再次执行高层程序。

| 程序 | 结构效果 |
| --- | --- |
| `chiplet_transform(chip, cores_per_chiplet, noc_spec, nop_spec)` | 删除 chip 原网络，按组插入 chiplet，为组内创建 NoC、组间创建 NoP。支持一条重复 core 或多个显式 core。 |
| `gsm_to_dmc(chip, shared_memory, parts)` | split 共享内存，展开直接 compute-bearing core 的 repeat，为每核插入一个含私有内存分片的 tile；添加局部 bus 与全局 mesh。parts 必须等于 core 数量。 |
| `pim_transform(memory, level, nums, spec)` | 递归分割 memory，插入指定层级，最底层放入 spec.compute 与 spec.network。level/nums 可为单值或等长列表。 |
| `bond_memory(core, memory_spec)` | 为每个 core 绑定一份 technology=`bonded` 的 dram；若 core 是点，先按 group_size=1 包裹。 |

NoC/NoP/总线 spec 必须指向 network 角色。GSM→DMC 默认的 bus、mesh 参数都标为
synthetic seed，可通过 `bus_spec`、`noc_spec` 显式替换。`pim_unit` 已注册，合法规格可用于
`pim_transform`，并由 `pim_offload` 执行受支持的 GEMV；缺少必需参数时报告具体参数错误。
真正未注册的组件仍返回 `unsupported_component`。`pim_transform` 在私有副本中批量构造，
最后验证整树，保留逐条 split/insert/create 记录供独立重放；失败不修改输入。论文三个变换
例子的手写基本编辑与高层程序的规范化等价性见 `tests/core/test_structure.py`。

## 规范形式

`normalize` 从叶子向根处理，保持根不变；一个子容器满足 repeat=1、无网络、只有一个
元素，且提升该元素不会造成同名冲突时，将其展开。所有兄弟元素及网络按名字排序。
保留有网络、重复、多个子元素、空容器以及会造成名称冲突的层级；保留参数值、source
和元素名字。规范化是幂等的，不承诺检测任意图同构或别名架构。

`normalized_fingerprint` 使用共享契约的 SHA-256 fingerprint。该指纹包含参数和
provenance，因此它是保守的完整候选去重；参数或来源不同的树不会被错误地合并。
环境保留用户编辑后的路径供下一步引用，搜索在评估前对规范形式去重。

## 三层策略

`arch_explorer.search.run_search` 的 SA、greedy、random 都调用
`ExplorationEnv` 公开工具。默认外层只从组件目录和当前图生成基本编辑：split/merge、
insert/remove、create/delete、跨容器移动，以及分类参数的原子 delete/create 替换。
复制可以用于整棵子树；已注册但种子没有的组件也可进入候选。所有容器使用相同规则，
名称、预设家族、阶段和配额不控制可达范围。高层变换仍可显式调用，但不进入默认集合。

新组件使用当前树中同名同单位的参数值或声明域中的代表值，标记为 synthetic 来源。
不能从目录或现有参数确定的必需固定参数不会编造；此时由用户提供完整 spec。
代表值只是有限的一步提议，不是对所有合法图的穷举，也不保证可执行。
`search.proposals` 是用户明确限定的编辑/编辑组集合；`--basic-only` 过滤其中的高层变换。

树/组件校验失败不调用后端；实际后端尝试（含失败）计入评估预算。
CGRA 与其他组件使用同一候选机制，也允许显式参数域和 set_parameters。
CGRA 后端仍要求受支持的4×4标量uint32模型、16个PE、两个SRAM bank、容量和相关
时钟/带宽关系；无效候选会返回真实后端原因，不代表新增物理支持。
默认未知时钟是周期归一化单位，不能把调时钟的结果解读为测量精度。
详见 [CGRA](cgra.md)。checkpoint/restore 不退还已消耗额度。

中层包含当前参数基线，并再采样至 `parameter_samples` 个参数版本。每个版本修改一个
已声明 domain 的坐标：数值区间在当前值 ±`parameter_radius * abs(value)` 内均匀采样，
int 最少允许一步；当前值为 0 时用 domain 宽度确定窗口。离散数值列表选当前值最近的
候选及相邻项；字符串列表均匀采样；domain=None 保持固定。所有结果再次校验。
每轮中层先以 `arch_explorer.cost.estimate` 估计当前基线 ADF。只要任一已知且启用的
成本/面积约束超限，就以 `shrink_probability`（默认 0.75）概率优先选容量、lane 数、
阵列行列、tensor/SIMT/PIM 运算单元规模，且只在当前值及更小值中采样。
这些参数会影响当前成本公式；时钟和模拟带宽没有经过面积标定，不列入缩减优先项。
整数边界和离散选项可能产生不变候选；偏置不是保证每次都降低成本。
若所有启用指标均可用、无超限且至少一项尚有余额，以 `growth_probability` 概率优先选
容量、算力或带宽参数并只向上采样。若某项超限，缩减优先于其他约束的余额；未知指标
不产生方向判断。未触发偏置时仍做对称局部探索。每个参数版本都从同一基线出发，
方向不随上一份随机样本改变；同配置、随机种子和工具序列可重放。
`parameter_radius`、`growth_probability`、`shrink_probability` 和 `cost_model` 在合成配置中显式列出。
已有的 domain=None 默认仍保持固定。调用者可在 `parameter_domains` 中逐路径明确列出
正的有限数值选项及 source（必须包含种子值），例如 `{path:"chip/core/sram", param:"capacity",
values:[524288,1048576,2097152], source:"synthetic private SRAM search space"}`；此时该坐标使用
离散邻居采样。所有选项都先通过现有组件校验；显式选项可以缩小、不能扩大已有 domain。
组件定义不变，未声明坐标仍遵守原规则。结构删除了对应路径后，该域不再参与采样。
示例 `examples/adf-budget.json` 在真实 events 后端上演示非 GPU 私有 SRAM 容量参数与成本约束。

内层枚举小的 `MappingTemplate.space()` 笛卡尔积；大空间采样扁平索引，避免物化整个积。
每 workload 总共至多 `mapping_samples` 个 mapping 候选，多个适用模板共享这个额度。
各 workload 独立选最低延迟 mapping；成本/面积由同一份 ADF 成本估计提供，不随 workload 改变。

目标为 `Σ weight*latency_ns + cost_penalty*max(0,cost-cost_budget)
+ area_penalty*max(0,area-area_budget)`。`best` 和中层 winner 按这个目标排序；约束超出
量、feasible 与占用比同时输出。罚系数过小可能选中超预算候选，调用方必须检查 feasible。
启用的约束缺少指标时目标为 null，search 报 `unavailable_constraint`；未启用该约束时
可仅以延迟搜索。未知指标始终保留 null。
`cost_model.basis` 选择 raw（默认）或 full；口径、来源、未支持项见 [cost.md](cost.md)。
后端 Evaluation 中的面积/成本保留为原始后端输出，搜索目标使用独立成本包的估计。

greedy 只接受目标不增的移动；random 接受所有已知目标的合法候选；SA 对目标下降直接
接受，对上升量 Δ 仅在 `Δ <= max_relative_degradation*abs(current_score)` 时，以
`exp(-Δ/temperature)` 概率接受。每次提议后温度乘 cooling。search 设置是算法参数，
不进入仿真模型。max_steps 防止无效或重复提议无限循环，因此返回的实际评估量可能小于
预算。比较策略时以 trace 中实际 backend 调用数为准。
