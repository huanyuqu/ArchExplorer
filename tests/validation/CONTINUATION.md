# 真实导出与物理分片的独立参考

这些测试工具从实际 ONNX ModelProto 和原始根生成参考，独立于产品导入、lowering、布局和执行器。`framework_numpy_interpreter.evaluate` 的默认行为保留；新模型显式调用 `framework_continuation_numpy.evaluate(model, feeds)`。模型名不参与分派。

## 逻辑接口

`onnx_index_expectations` 提供 `first_argmax(data, axis_value=0, keepdims=1, select_last_index=0)`、`gather(data, indices, axis_value=0)`、`gather_elements`、`shape_of`、`concat`、`slice_tensor`、`expand`、`squeeze`、`unsqueeze` 和 `integer_binary`。索引与 shape 保持有符号 int32/int64；float/bool 不自动转成索引。整数算术使用 Python 精确整数，除法向零截断；有符号64位溢出在此参考的准入域外，不能据此声称 ONNX 要求 wrap 或拒绝。

`onnx_conv_expectations.conv_geometry(input_shape, weight_shape, *, strides=(1,1), pads=(0,0,0,0), dilations=(1,1), group=1, auto_pad='NOTSET')` 接受正维度 NCHW 2D Conv，返回完整输出 shape 和实际填充。`conv2d(data, weight, bias=None, *, dtype='fp32', **options)` 按 channel/kernel 行列顺序执行 FP32 FMA、FP32 bias 加法及目标输出存储舍入。填充乘零的逻辑 FMA 数只用于数学描述，实际内存读取与组件计费须从产品执行原件核对。

`framework_continuation_numpy.evaluate` 消费 IR8/opset17 的真实节点类型，扩展整数/shape、Conv、索引、广播、归约、Exp 和多输出。它保留每个原值名、中间结果和输出类型；浮点使用 float64 容器保存已舍入的声明存储值，整数值始终保留整数容器。显式整数到浮点 Cast 直接从精确整数舍入，归约采用顺序 FP32 累加；BF16 使用有理数到二进制网格的 ties-to-even 参考。并行框架归约和融合 GELU 可能与逐 ONNX 节点的舍入不同，应保存差异。

```sh
python tests/workload/framework_continuation_reference.py \
  --package FROZEN_INPUT_DIRECTORY --case FROZEN_CASE_ID --output NEW_REFERENCE_DIRECTORY
```

可选 `--framework-module FROZEN_GENERATOR.py` 仅调用其真实 CPU 模型构造和 forward，不重新导出，不调用产品。生成 `INDEPENDENT.npz`、独立 `OFFICIAL.npz`、可选 `FRAMEWORK.npz` 和 `RESULT.json`。FP32 每个原值按预设 atol/rtol=5e-5 比较，整数/bool 精确比较。低精度不以官方 evaluator 作 oracle，状态为 `reference-generated`；真实框架输出误差逐项保存且不调阈值。该阶段不包含 Graph/组件/安装通过结论。

## 物理观测协议

`physical_shard_expectations.audit(document)` 接受 `arch-explorer.qa-shard-observations.v1`，所有坐标是非负整数：

| 字段 | 类型与含义 |
| --- | --- |
| tensors | 名称到 `{elements:int>0,dtype:fp32/fp16/bf16/int64/int32/bool,root:bool}` |
| capacities | tile 名称到容量字节，正整数 |
| links | 有向链路 `[source_tile,target_tile]` 列表 |
| buffers | `{id,tensor,tile,kind,start,stop,offset,bytes,born,ready,dead}` 列表；kind 为 root/output/scratch/copy |
| transfers | `{id,source,target,start,stop,begin,end,bytes,route}` 列表；source/target 是 buffer ID，route 为 tile 序列 |
| reads | `{buffer,tile,at,start,stop}` 列表 |

逻辑元素片段为 `[start,stop)`，分配存活为 `[born,dead)`，`ready` 是内容就绪时刻。相同时刻先释放再分配，读时刻必须严格早于 dead。每个逻辑根/输出/临时张量的原始片段恰好覆盖完整张量；复制另有独立分配，不能充当原始片段。此协议限定稠密连续元素片段；任意 strided view 须先提供可审计的独立地址展开，再由适配器归入该域。

检查逐 tile 实际物理地址及同时存活峰值、活跃别名冲突、各片段末读者、目标复制完整覆盖、源/目标存活与就绪、有向路径及每跳传输字节。copy 的读取必须有对应目标片段，远程读取不能免费引用源 buffer。返回峰值、峰值时刻、末读者、复制分配字节与逐链路字节。

`test_physical_shard_expectations.sample()` 是手工小型控制：缓存两片、两跳复制及输出复用已释放的复制空间；12B exact-fit、11B 容量不足。它检验审计工具，实际 Events 适配、节点数值、观察完整性及执行身份仍需唯一产品候选与原件。协议不会认证调用方自报的时间/路由。

## 标准来源与参考分歧

以本地 ONNX1.16.2 的 `onnx.defs.get_schema(op,17)` 固定实际版本，官方文档支持语义：[ArgMax13](https://onnx.ai/onnx/operators/onnx__ArgMax.html#argmax-13)、[Gather13](https://onnx.ai/onnx/operators/onnx__Gather.html#gather-13)、[Conv11](https://onnx.ai/onnx/operators/onnx__Conv.html#conv-11)、[Slice13](https://onnx.ai/onnx/operators/onnx__Slice.html#slice-13)。opset17 Conv 不接受 BF16；实际模型中的 FP32 Conv 与显式 BF16 存储转换按原类型计量。

ONNX1.16.2 `onnx.reference.ops.op_conv._conv_implementation` 的 auto_pad 路径使用 `X.shape[i]` 代替空间维 `X.shape[i+2]`，控制保存其真实差异并列为 xfail，不能计为官方数值通过。显式 pads 的官方控制与手算 SAME_UPPER/LOWER 几何分别保留。BF16 官方参考既有分歧继续保留原分类。

所有探测用 `tests/validation/stage_capture.py` 落盘退出、资源和进程组清理。小型120s/8GiB；规模600s/16GiB且调用方显式传入同宿主已约定锁。完整规模首发属于模拟器作者，QA 优先审核其原件。
