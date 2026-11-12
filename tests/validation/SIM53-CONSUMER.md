# SIM53 v1 的独立消费者接线

`static_control_observations.classify(description, types, oracle, model)` 以原始 ModelProto、封存独立原值参考及正式 describe_onnx 的 values/nodes/outputs 划分物化值与静态控制。每个条目核原值 dtype/逻辑 shape；原节点和消费者的输入/输出身份也与 ModelProto 对齐。

`tensor=null` 单列 static，记录 source attribute 摘要、消费者和已知 constant_folded 标记，`cpu_value_returned=false`、`folded_value_numerically_verified=false`。这个名单不进入 CPU 数值通过计数。`cast_observability` 只在两端均有 tensor 时做配对转换检查，其余静态 Cast 单列；物化目标仍可以作为原节点输出与独立参考比较。全部原始公开输出必须有物化 tensor 且顺序一致。

`inspect_consumer_shapes` 从正式普通 Graph 核消费节点的输出 tensor shape 与可直接匹配的 shape 属性。它明确保留未核对属性，不能将源 attributes 的摘要、编译期折叠或 shape 一致性扩大成所有静态值/offset/axis 已数值核对。消费者保存两类名单和每类覆盖；真正产品状态仅由唯一候选运行填写。

`array_access_expectations.expected(op, tensors, values)` 根据冻结 ARRAY-API 在小输入上独立列出逐逻辑坐标读取与字节/抽象 issue 预期。Gather 索引各读一次，重复索引的数据读取保留重复；Where eager 读取两个分支；同一 tensor 在两个输入槽出现会累计两次；ArgMax 沿轴顺序读取，并区分逻辑标量索引和物理[1]。`compare_counts` 与 `compare_indices` 分别核正式返回的计费字段和读取坐标/顺序；物理地址须再经实际布局/访问展开独立核对，不能把逻辑坐标当物理字节地址。

`block_expectations` 提供：

- `partition_grid(shape, block_shape)`：末块裁切、starts 字典序、每片FP32字节。
- `partition_coverage(shape, blocks)`：在不超过一百万元素的小输入上逐元素核覆盖、越界/重叠/漏片和字节，返回每片真实 canonical flat 地址。矩形切片不能被当作一个连续 flat 区间。
- `payload_digest(value)`：独立 FP32 量化后 little-endian payload 摘要，保留正负零。
- `load(source, starts=..., shape=..., guards=...)`：矩形源读取一次，完整读取每个栅栏，生成独立新输出，源/栅栏字节分列。
- `zero(shape)` 与 `matmul_accumulate(A,B,C)`：完整写零或从 C 开始按 K 顺序 FP32 FFMA，累计源读取和新输出写入，禁止隐含原地更新。参考使用有理数舍入；手工有限 dyadic 小样本与声明的 FP64乘加后FP32舍入次序一致，未外推为所有浮点边界的设备等价。

小型 INPUTS 的 root 字节按槽顺序独立核为[564,272]，合计836B，cache-pass boundary 为1200B。该核对仅依据无答案根/分片边界和元数据；workspace 上界、聚合峰值、actual Events/容量和论文 MAC16×16 硬件均未借此认定。原四行尺度不构图、不编译、不执行，不占规模锁。

通用数学 oracle 比当前生产准入域更广。例如已有 grouped Conv 与负步长 Slice 控制用于独立标准参考，SIM53 v1 生产明确只接 group=1、显式非负 pads 与正步长非空 Slice。六份实际模型按正式支持域消费；域外输入应记录正式拒绝，不能把数学 oracle 能计算当作产品已经支持。

源码阶段只验证这组独立工具，并校验接口/输入闭包。完整 partition_input 返回 JSON、describe_onnx 静态/运行/Conv/Cast metadata 样本已向作者一次请求，用于后续精确适配；没有读取或安装作者生产源码来猜字段。实际分片恢复、Graph/编译/CPU/组件/CLI/fresh与有效重封全部继续等待负责人唯一候选。来源、数值、物理、组件和安装分列，既有原件不重标。
