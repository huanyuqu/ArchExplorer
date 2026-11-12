# 完整来源的逐样本硬件时间比较

`arch_explorer.archs.hardware_comparison.compare_hardware(document)` 实现 `arch-explorer.hardware-pairs.v1`。输入严格包含 schema、samples；样本严格包含 sample_id、split、measured、predicted。ID 全文件唯一，split 为 calibration/development/holdout。重复项全部拒绝，不仅丢弃后出现的一行。

每个 observation 恰为 kind/source/conditions/unit/value/repetitions/raw_values。source 恰为 locator、sha256、license、redistribution、evidence_class、audit。locator 可追溯到原始文件行，sha256 为小写64位，许可和再分发范围显式；audit=reviewed/unreviewed。工具不验证自报审核标志和摘要的真实性，来源准入需要核原件。

conditions 两侧必须在规范 JSON 值层面完全一致，恰含以下对象：

| 对象 | 全量字段 |
| --- | --- |
| device | vendor/sku/architecture |
| software | sdk/driver/kernel |
| clock | mode/core_hz/memory_hz |
| operation | op/inputs/outputs/attributes |
| precision | inputs/outputs/accumulate/rounding |
| mapping | 完整有效映射、布局、实现选择 |
| timing | start/end/synchronization/warmup/statistic |

inputs/outputs 是原名字典，值为完整正 shape，或含 shape 与角色的对象；显式标量为 `{shape: [], role: 'output'}`。attributes 可以是已知空对象；其他身份拒绝 null、空容器或 unknown/unspecified/todo。clock 正且有限；warmup 为非负整数；precision 输入/输出为非空 dtype 数组。不能从预测补全实测一侧缺项。

unit 为 s/ms/us/ns，显式换算 ns。repetitions 是正整数，raw_values 长度精确相等，value 必须等于 mean/median/min/max 所声明的统计量（相对浮点容差1e-12、绝对容差0）。原始值和计算的时间/误差必须有限非负；signed error 可为负。不能把论文均值复制多次冒充原始重复数据。

仅 raw-device-timing/simulator 配对为 hardware-pair；synthetic-control/synthetic-control 单列。paper-digitization 与 simulator 值不能充实测。每行给 P-M、绝对误差、相对误差；M=0 时相对两项为 null，计 zero_reference，不纳入 MAPE 分母。按 split、证据类别及两侧 audit 组合分别聚合，给样本数、零数、分母数、均值与最坏样本 ID。工具不拟合，也不跨 split 混报。

```sh
archexplorer compare-hardware samples.json --output ERRORS.json
```

合法行与具体拒绝原因一起写出，存在拒绝行时 CLI 返回2。合成控制仅证明比较器行为；硬件精度需要来源审核通过的同设备、同精度、同形状、同映射、同频率及同计时范围真实记录。完整冻结样本合同由 VAL63 独立维护，生产不引用 QA 实现。
