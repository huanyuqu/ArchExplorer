# 硬件比较样本 v1

冻结协议 `arch-explorer.hardware-pairs.v1`。CORE34 的公共比较器可以独立实现此协议；QA 参考实现为同目录 `hardware_comparison.py`，不依赖生产代码。完整合成例由 `test_hardware_comparison.pair()` 提供，其零 SHA 和 synthetic 来源仅用于比较器测试。

输入对象严格包含 `schema` 和 `samples`。每个 sample 严格包含 `sample_id`（全文件唯一字符串，重复ID的全部出现均拒绝）、`split`（calibration/development/holdout）、`measured`、`predicted`。不混合拟合集、开发集与留出集聚合。两个 observation 各包含以下全部字段：

| 字段 | 完整值/类型 |
| --- | --- |
| kind | measured 或 predicted，与所在槽一致 |
| source | 对象：locator、sha256、license、redistribution、evidence_class、audit |
| conditions | 下述完整匹配对象 |
| unit | s/ms/us/ns；只支持时间，显式换算 ns，不由吞吐反推时间 |
| value | 非负有限数，等于 raw_values 声明统计量 |
| repetitions | 正整数 |
| raw_values | 长度等于 repetitions 的非负有限数列表，单位同 unit |

source 的 locator 是原始文件与行/样本的可追溯位置；sha256 是该来源文件的小写64位摘要；license 与 redistribution 必须如实填写许可及可公开范围，私有论文输入不进入公开包。evidence_class 为 raw-device-timing、paper-digitization、simulator、synthetic-control。audit 为 reviewed/unreviewed。文件摘要和自报 reviewed 不构成鉴权或真实性认证，必须独立核原件。没有原始重复记录时不能将论文均值复制 repetitions 次。paper-digitization 保留为待配对材料，不能冒充本协议 raw-device-timing。

conditions 的字段精确如下，两边在规范 JSON 值层面全部一致；不同 unit 可以显式换算。不能用预测一侧填写缺失实测身份。

| 字段 | 必填子字段与语义 |
| --- | --- |
| device | vendor、sku、architecture（确切设备/硬件修订） |
| software | sdk、driver、kernel（精确实现/版本或摘要；确实无驱动需写明不适用原因） |
| clock | mode、core_hz、memory_hz（正数；真实采样/固定范围另保留原件，不能填峰值默认频率） |
| operation | op、inputs、outputs、attributes；inputs/outputs 包含全部形状与根角色，attributes 可为空对象 |
| precision | inputs、outputs、accumulate、rounding，不能只写一个 dtype |
| mapping | 完整有效映射、布局及实现选择对象，不仅名称 |
| timing | start、end、synchronization、warmup、statistic；statistic=mean/median/min/max，warmup明确次数 |

所有身份字段拒绝 null、空字符串/容器及 unknown/unspecified/todo。显式标量使用带 `shape: []` 的对象，未知形状保持 null 并拒绝。不支持的身份应先拒绝，不自行补默认值。空 attributes 表示确定的无属性操作；warmup 是非负整数。

有效硬件配对必须 measured=raw-device-timing 且 predicted=simulator。合成对两侧都必须 synthetic-control，单列报告。逐点输出 signed_error_ns=P-M、absolute_error_ns=abs(P-M)、signed_relative_error=(P-M)/abs(M)、absolute_relative_error=abs(P-M)/abs(M)。M=0 时 relative 两项均为 null，单列 zero_reference；不以 epsilon 替代分母，不把零参考纳入 MAPE。分 split、证据类与审核状态聚合，给出有效数、零数、相对分母数和最坏绝对/相对样本 ID。拒绝项保留具体原因，不能静默丢弃后宣称全部通过。

接口源码尚未证明硬件精度。已核 paper-fig7 原件明示 not raw measurements；MetaDSE 架构配置的 latency 字段也不等于原始采样。待来源审核和真实预测身份齐备后才可作为硬件比较输入。
