# ADF 面积与成本模型

`arch_explorer.cost.estimate(adf)` 返回 `CostEstimate`；它是公开参数驱动的分析模型，
不依赖注册组件的 area/cost 回调，不执行 CACTI 或任何在线服务。数值不是当前报价，也
没有用验证设备的面积/价格调整常数。外部参考代码、论文与实验数据不随包分发。

```python
from arch_explorer.archs.adf_seeds import dmc_adf
from arch_explorer.cost import estimate

adf = dmc_adf(num_cores=1)
cost = estimate(adf)
print(cost.area_mm2, cost.raw_cost_usd, cost.full_cost_usd)
print(cost.by_category, cost.unsupported)
```

此例使用显式合成非 GPU 架构，默认工艺仍是未经器件验证的参考场景。逻辑面积和 raw/full 成本
可返回，DRAM die 密度/面积缺失仍保留在 unsupported，不能将其解释为完整硅面积已知或实际报价。

## 输出与口径

- `area_mm2`：逻辑 die 面积之和。包括逻辑侧内存接口，不包括 DRAM die，也不是封装占地。
- `raw_cost_usd`：未计逻辑 die 良率的 wafer 分摊 + 内存价格；对应 raw-die+memory 口径。
- `full_cost_usd`：额外考虑逻辑 die 良率、bump、organic substrate 及封装失败损失的 **RE 场景**。
  不是完整 Chiplet Actuary 工具，不包含 NRE、CoWoS/interposer、供应商利润或未知实际装配步骤。
- `cost_usd`：选择的口径，`basis="raw"` 默认，或 `basis="full"`。两份估计总是分别保留。
- `by_category`：arithmetic、control_rf、sram、interfaces_network 的已知逻辑面积小计。
- `by_path`：路径、展开后实例数、分类面积、内存成本、按逻辑面积比例分摊的 raw die 与总 raw 成本。
  这种分摊不表示每个 SpacePoint 都独立制造。
- `memory`：内存价格、DRAM 硅面积之和与 bonded stack 占地；都与逻辑 die 面积分开。
- `provenance`：输入参数、模型参数及公式系数的 value/unit/source；`assumptions` 是使用的参考配置。
- `unsupported`：未知组件、缺少的输入、未使用的 override 及其影响指标。未知组件的行面积和总成本为 null，
  不能被当作免费组件。类别及 memory 中的和只代表已知项，不是缺项为零的完整总量。
- `omitted`：当前公式范围之外但未独立刻画的细节，例如 SFU、物理布线；这些限制不伪装成测量值。

DRAM die 面积未知时，逻辑 die 面积和已知的内存价格仍可用。不要把“逻辑面积可用”解释成
“封装中全部硅面积已知”。模型对同一 ADF 的费用不随 workload 改变。
缺少或无效的 `dram_usd_gib` 仅使总成本和该行内存价格为 null，不清空逻辑接口面积；
已知的 raw die 分摊仍保留。`memory.cost_usd` 在有未知价格时为 null。

## 配置与物理参数

环境、Agent、SA/greedy/random 均通过同一成本包评估。JSON 配置示例：

```json
{
  "cost_budget_usd": 100,
  "cost_penalty": 10,
  "growth_probability": 0.75,
  "shrink_probability": 0.75,
  "cost_model": {
    "basis": "raw",
    "process_nm": {"value": 7, "unit": "nm", "source": "LLMCompass 7nm reference scenario"},
    "die_count": {"value": 1, "unit": "count", "source": "one-die design assumption"},
    "overrides": {}
  }
}
```

此片段合并到完整探索配置；`process_nm` 支持 5 和 7，不插值其它节点。默认值为 7nm 与
一个逻辑 die；`die_count>1` 表示等面积、同工艺逻辑 dies 的明确场景，不从容器层级猜测
die/chiplet/package 边界。repeat 按全部祖先相乘，仅表示 ADF 实例数。
未显式提供节点时，`assumptions` 带 `status="unverified_default"`，说明默认节点未经设备确认。
保留默认计算是为了支持未定工艺的合成探索场景，不能据此宣称某个器件的实际工艺或面积。
显式传入未支持的 12nm 会拒绝，不做未刻画的缩放。传入来源只记录使用者
选择的节点，不代表工具已验证该来源。`scope_notes` 可携带字符串列表，在 assumptions 中保留配置的已知限制。

完整预算示例（包含明确的非 GPU 私有 SRAM 离散容量域）可直接运行和重放：

```sh
archexplorer search --config examples/adf-budget.json --policy sa --output results/budget.json
archexplorer evaluate results/budget.json --output results/budget-replayed.json
```

`overrides` 形式为 `{"完整/ADF/路径":{"参数名":{"value":...,"unit":...,"source":...}}}`。
Python API 可使用 `Param`；环境 JSON 中使用上述对象。参数必须有来源，不能静默吞掉
拼错的路径或未使用参数；结构变换导致 override 失效时，报告不完整估计。常用 cost-only 参数：

| 位置 | 参数与单位 | 含义 |
| --- | --- | --- |
| 任意点 | physical_instances / count | 该类型路径的总物理实例数，不得小于执行 ADF 展开数量 |
| 网络 | endpoint_count / count | 物理端点总量（当前网络实例内），用于非执行资源与实际资源不同的情况 |
| 网络 | nvlink_links、pcie4_lanes / count | 当前网络实例额外承载的逻辑侧外部接口数；分别按 NVLink3 link、PCIe4 lane 计费。p2p_switch 已计 NVLink，拒绝额外 nvlink_links 防止重复 |
| DRAM | interface_bytes / B 或 interface_count / count | 接口容量比例或明确接口数；默认 HBM 每16GiB一个接口，DDR每GiB一个32-bit接口（场景假设） |
| DRAM | dram_usd_gib / USD/GiB | 给定的内存价格；GDDR 没有内置价格，不自动借用 DDR5/HBM 价格 |
| DRAM | dram_density / MiB/mm2 | 已知 die 容量密度；未给出时普通 DRAM 硅面积为 null |
| bonded DRAM | bonded_layers / count | 1–4层，默认4层场景 |

cost override 不修改 ADF、不改变仿真结果；它与用于实际参数探索的 parameter_domains 是两个不同配置。单位不匹配报错；价格
用 GiB，是为了明确沿用来源把标称 80GB HBM 当作80个容量单位的模型约定，不能混用十进制字节。

## 逻辑面积公式

常数存于 `cost/constants.py`，全部使用 `Param`。主要来源为固定版本
[LLMCompass 面积模型](https://github.com/PrincetonUniversity/LLMCompass/blob/2e015fd2ee750e6cad8d7152df52551e5b41ef20/cost_model/cost_model.py)
及其 [GA100 配置](https://github.com/PrincetonUniversity/LLMCompass/blob/2e015fd2ee750e6cad8d7152df52551e5b41ef20/configs/GA100.json)。
实现独立重写公式，没有复制或运行外部仿真器。

浮点单元面积为 `N × 685300 × (mantissa_bits/52)^2 / (density×10^6)`；FP64/32/16 的 fraction bits
为52/23/10。BF16 扩展采用7，明确为同一缩放法的扩展；INT32 用177690晶体管。7/5nm密度为
96.3/138.2百万晶体管每mm²。systolic_array 用 rows×cols 个FMA；vector_unit 用 lanes个FMA与
参考 lane control。未知精度或未刻画的组件（包括当前 cgra_pe）报告不支持。

原生 `accelsim_*` GPU 组件没有独立面积/成本估计，相关指标保持未知。
非 GPU 的 systolic/vector/PIM 估计继续按各组件契约执行。

SRAM：`A = blocks × block_bytes × 8 × bitcell_area / efficiency`。block至多512KiB，
大容量按完整block向上取整，小容量至少4KiB；efficiency取来源的CACTI表（按容量log2向上索引）。
7/5nm bitcell area为1.70e-7/1.25e-7 mm²/bit。此处是静态CACTI-derived参考表，
不是在每个候选上运行CACTI。

mesh/bus/crossbar 使用来源的44300000晶体管/展开端点通信代理，不声称覆盖拓扑布线面积。
HBM2e/DDR5 接口采用来源的 controller晶体管 + 固定PHY面积；p2p_switch用NVLink3每链接代理。
显式 PCIe4 lane 使用同一来源的0.48mm² PHY和3962500 controller晶体管；GDDR接口目前借用
DDR5 32-bit代理，并在 assumptions 中注明，尚未独立刻画GDDR PHY/controller。
频率、模拟带宽不直接转成面积；需要物理模型才能扩展它们。增加网络层级会显式增加接口开销。

## 制造和封装

raw die 使用300mm wafer，`gross_dies = π150²/A − π300/√(2A)`，
`raw_die = wafer_price/gross_dies`。7/5nm wafer价格为历史模型的9346/16988美元。
参见 [LLMCompass所用供应链模型](https://github.com/PrincetonUniversity/ttm-cas/blob/d98ac0586faaf63f8f4eac94aa7cd50f37421c1f/supply_chain_model.py)。
该解析gross-dies近似不进一步加入scribe/edge-loss；过大的单die不会被悄悄拆分。

full RE 场景遵循 [Chiplet Actuary](https://arxiv.org/html/2203.12268) 的负二项良率及organic封装费用分解，
参数固定到 [42dd5dee版本](https://github.com/Yinxiao-Feng/chiplet-actuary/blob/42dd5deeec2b2f2daf13b1f3ac6042df2d4e3b62/parameter.ini)。
`Y_die=(1+A×D/10)^(-10)`，7/5nm D为0.0009/0.0011每mm²。
full逐项报告raw_die、die_defects、memory、bumps、substrate、package_defects、wasted_KGD。
organic substrate面积为逻辑总面积4倍，单位面积0.005美元，bump同为0.005美元/mm²，
bonding yield为0.99^die_count；多die substrate层数因子按来源的面积阈值计算。
内存按已采购的良品计价，不再次施加逻辑工艺的良率，但计入封装失败的报废成本。
这不是A100 CoWoS成本，不能拿full与排除封装的参考价做同口径精度结论。

## Bonded DRAM 与 PIM

bonded 使用25.6MiB/mm²/层、最多4层。容量C的总DRAM硅面积为`C_MiB/25.6`，
stack占地为该面积除以层数。1GiB对应40mm²总硅，4层占地10mm²。
价格使用HBM参考7美元/GiB；计入每层95% bonding yield后为`C_GiB×7/0.95^layers`，
1GiB四层为约8.594美元。该内存费用已经含stack bonding损失，full不重复计这道损失。
上述归一化参数来自任务给定的论文方法；[原始3D-DRAM研究](https://arxiv.org/abs/2604.08044)
的单位归一化尚待审计，provenance 保留该证据等级。

`pim_unit` 与唯一DRAM处于同一bank组时，DRAM面积/价格加20%，PU不再重复作为逻辑die上的计算单元计价。
HBM-PIM：容量按HBM2e的7美元/GiB乘1.2。DRAM-PIM的AiM等效die数取容量、PU数和计算吞吐所需die数的最大值：
`max(C/512MiB, PU/16, total_ops_per_cycle×GHz/(16×32×1))`。
来源为 [SK hynix AiM Hot Chips 34](https://hc34.hotchips.org/assets/program/posters/hc34.SKhynix.YongkeeKwon.v03.pdf)：
4Gb/die、16PU、1GHz、每PU16 MAC/cycle。价格按等效容量×明确的单位价格×1.2，
不从bank内部带宽猜测外部I/O规模。DDR场景的内置价格2.4美元/GiB来自LLMCompass；这不是AiM的GDDR报价。
GDDR必须提供dram_usd_gib。若无公开DRAM面积密度，**不能由512MiB容量推导mm²**；提供密度d后，
每个等效die面积为`512/d×1.2 mm²`。该缺失不会被填0或伪装成测量。

`pim_transform`/`pim_structure` 的同构 DRAM 组按所有展开 part 汇总为一行：容量、PIM单元数、
吞吐和已显式给出的接口数都计入祖先repeat，组内全部容量只计算一次接口和价格。组内 bank bus
属于 DRAM 内部结构，不按独立逻辑 die 网络收费；仍只有20% PIM开销假设，没有新增内部布线标定。
默认接口数仍是容量比例场景（HBM每16GiB一个1024-bit接口，DDR/GDDR每GiB一个32-bit代理），
**不是由bank数推断实际引脚数**。例如 NeuPIMs 默认32GiB得到2个HBM接口单位；如按其32个128-bit
channel建模，应提供有来源的组级 `interface_count=4`，不能把默认容量比例当成该设备的物理事实。

组级override指向 `neupims/hbm` 或 `hsndi/dram` 等原DRAM容器路径。`by_path` 提供
`member_paths`、`part_instances`、`pim_units`、`interface_count`。part级override不会被静默丢弃，
会报unused override；异构技术/价格/密度或只在部分bank放置PIM的组保留逐part处理。
相同原因及受影响指标的不支持项合并，`affected_path_count` 是受影响类型路径数（不是repeat展开实例数），
`example_paths` 最多列四条；完整成员路径在组行中。

## 验证与重放

非 GPU 成本可与同一 ADF 的执行结果分开检查，复核时使用开头的 `dmc_adf(num_cores=1)` 示例。

原生 GPU 组件当前不提供独立面积/成本估计，未知保持 null。静态厂商资料或参考微架构事实
不构成可执行 GPU 时延模型；不再用旧 GPU 预设运行成本/时延示例。

提交文件保留cost_model及完整cost_estimate；replay_result重新计算。历史结果中没有成本模型配置时
采用当前默认模型，旧版本失败trace/目标可能不同；严格trace重放需固定同一模型版本。
`adf.evaluate` 已接入该模型：后端未提供两项指标时填入面积/成本；`cost_model=False` 可关闭。
`details.cost` 保留模型、口径、分项与 unsupported；完整 assumptions/provenance 请查看 `estimate` 输出。
