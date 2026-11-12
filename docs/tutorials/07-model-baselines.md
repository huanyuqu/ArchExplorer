# 7. 基线工具与执行边界

完整 GPU demo 使用[原生程序入口](05-full-models.md)。旧自建 GPU 时延模型、GSM 种子和依赖
它们的基线配置已移除。已有实验结果、失败与采集记录保留在原工作目录，不改写成新原生证据。

`tools/run_model_baseline.py` 仍组织通用非 GPU ADF/Agent 实验，记录实际调用、资源上限和
独立进程重放。它不实现时延公式。输入需要由当前受支持的组件和 mapping 构成，
不接受旧 GPU 预测种子或 `gpu_sm` / `gpu_kernel_service` 组件。
`--max-evaluations` 仅降低实验预算，不缩小负载、不补偿失败、不保证得到可行最佳候选。

新 GPU demo 的两个 mapping 在同一负载/ADF 下分别生成完整程序并执行，不用旧搜索胜负或
历史预测替代真实程序结果。完整前向、数值参考、执行覆盖、硬件精度需分别核对。
