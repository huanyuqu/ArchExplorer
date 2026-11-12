# 实际框架导出独立参考边界

本参考逐节点解释原始 IR8/opset17 文件。浮点存储是 FLOAT/FLOAT16/BFLOAT16；BF16 的16位载荷显式解码，数组使用 FP64 承载已舍入的数值，不把 uint16 整数当作浮点值。期望值不调用生产 importer、reference、lowering 或 layout resolver。

`typed_onnx_expectations.py` 使用既有有理数 IEEE ties-to-even 舍入器。FP16 的有效位11、最小正规指数−14；BF16 的有效位8、最小正规指数−126；FP32 的有效位24、最小正规指数−126。包含半途偶数、正负值、次正规数和负零控制。普通有限数的单次正规舍入相对界分别为 2^-11/2^-8/2^-24，次正规数改用半个最小格距的绝对界。溢出或非有限比较独立拒绝；不将通用相对界用于零值。

本次准备的逐元素合同是 FP32 计算后显式存入目标 dtype；矩阵内积采用按 K 顺序的 FP32 FMA 累加。MatMul 结束存一次；Gemm 的 alpha 与 beta 缩放及偏置在 FP32 中结合后存一次。不同框架融合、归约顺序及 BLAS 实现可能改变最终低位。原始框架输出、官方参考输出、本参考及后续映射 CPU 必须分别给出误差，不能将其中任何一个默认解释为逐位标准答案。该选择尚待与 SIM52 冻结的产品精确计算合同逐项绑定。

LayerNormalization 的 axis 归一化区间为 axis 至最后维，stash_type=1 在 FP32 内做均值、中心化、平方、方差、加 epsilon、sqrt、reciprocal、乘中心值，Normalized 显式 cast 回原 dtype 后分别 Mul Scale、Add B。Mean/InvStdDev 为 FP32 且保留归约维。SIM52正式合同允许有限非负epsilon与省略B；Scale/B必须恰为归一化suffix形状。零epsilon非零方差、可选B的精确手算控制独立验证，旧12文件均为正epsilon/显式B且不受扩域影响。只准备明确的 stash_type=1；其他值给出诊断。对照 [ONNX LayerNormalization 17 规范](https://onnx.ai/onnx/operators/onnx__LayerNormalization.html#layernormalization-17) 及实际安装 ONNX1.16.2 的 schema；Cast 的边界与 [ONNX Cast 13 规范](https://onnx.ai/onnx/operators/onnx__Cast.html#cast-13) 分别核查。本文公式依据所选17版本，不依赖网页当前总版本。

真实 CORE34 输入的支持节点是 Constant、Cast、Identity、Gemm、MatMul、Add、Sub、Mul、Div、Erf、Sqrt、Tanh、Transpose、Softmax、LayerNormalization。其中 Sub/Identity/Sqrt/Tanh 是独立辅助域，未冒称 CORE34 原件一定包含；未实现节点直接拒绝。CORE34 decoder 的两次 LayerNorm 和 Softmax 在文件内显式 FP32，Cast 回低精度；没有将整个 block 仅标 metadata。

GELU 对照真实 Div/Erf/Add/Mul/Mul 分解，每个中间节点有存储舍入。torch GELU 作为融合框架算子另列。误差必须逐节点报告最大绝对/相对误差、零参考数、shape 和 dtype；全模型统一容差无法证明上述舍入边界。独立准备结果不证明产品地址/流量、组件服务或硬件时延；这些需要唯一候选实际消费。

准备执行使用本地 Torch2.7.1+cu128 的 CPU 路径、NumPy1.26.4、ONNX1.16.2、protobuf4.25.8，未调用 CUDA。原件真实导出环境 Torch2.5.1+cpu/NumPy2.2.6 依旧按原身份保留；没有把不同版本改标为同一次执行。BF16 官方 ReferenceEvaluator 已实际探测，其错误/非有限/巨大误差逐节点留证，不能当成官方通过。FP16 辅助输出的 stash dtype 不符也单列；参考器进程返回0不等于数值合格。
