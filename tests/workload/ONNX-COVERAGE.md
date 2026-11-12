# ONNX 有界导入独立验收

标准文件域为 ONNX 1.16.2、IR 8、标准空 domain、opset 17、静态正尺寸 FP32 推理 DAG。模型来自确定性数值的 MLP 与显式 Q/K/V attention，包含矩形乘法、batch/标量广播、重复或空节点显示名、两返回根、共享中间量和共享权重。int64 只用于编译期 Reshape literal。

`onnx_import_cases.py` 生成真实 ModelProto 和外部权重，`onnx_generate_inputs.py` 生成无答案清单。`onnx_import_expectations.py` 为两个自有模型独立写出逐节点 NumPy 公式；`onnx_numpy_interpreter.py` 另外实现十种标准算子，解释正式输入。两者均不调用 ArchExplorer reference、lowering、layout resolver 或编译器求期望。原模型官方 ReferenceEvaluator、独立 NumPy、产品原 Graph、产品映射 Graph 和实际 Events 服务分别报告。

| 合同 | 独立输入/检查 | 候选消费者 |
| --- | --- | --- |
| MatMul/Gemm 参数、broadcast、Softmax 任意轴 | closed form、手算矩形乘法、独立解释器与官方逐节点对照 | test_onnx_import_edges.py、test_onnx_numeric_storage.py |
| Transpose/Reshape/Identity/Constant、稳定原身份 | 实际文件、literal int64、非默认 permutation、多输出 | test_onnx_import_contract.py、test_onnx_import_refusals.py |
| 外部权重完整字节、offset/length、明确根 | 非零 offset、搬移/重命名、缺失文件 | test_onnx_import_refusals.py、snapshot 移植 |
| FP32、有限根、符号特化、initializer 覆盖策略 | 标准合法控制与产品额外域限制分列 | 具体 ValueError/ImportError 文本，禁止任意 Exception 冒充拒绝 |
| 实际地址、服务、共享根、返回值持有、峰值 | 既有独立 root walk、标量 FMA、实际 allocate/free 透传观察 | 两模型×两 mapping×两 layout；views 峰值不预设下降 |
| 恰好容量与少一字节 | 消费已观察真实区间的峰值，拒绝重复估计 | test_onnx_numeric_storage.py 独立容量入口 |
| 公共用户闭环、失败记账、可选依赖 | 完整普通 WorkloadSpec、API/CLI/Env/Agent/search 入口 | test_onnx_user_consumers.py、onnx_search_consumer.py |
| fresh 只读/显式 rerun、来源拒绝 | 纯编译校验单列；prepare/engine/solver/native/network/write 设实际 guard | onnx_readonly_probe.py、重封篡改拒绝、独立新 rerun 目录 |

独立 oracle 与输入准备可在未安装产品的环境执行。候选消费者只在负责人统一构建的主 wheel/sdist 与同源扩展上运行；源码阶段仅进行 AST 检查，collection 不作为通过。需要的环境变量、依赖顺序与真实执行记录写入私有验收交付。

该有界域不关闭框架导出、全 ONNX 算子/动态形状、硬件精确时延或 IPU 成本。旧失败与旧矩阵保持原身份。
