# 2. 将自己的负载变成可解释的输入

先从模型代码或已有 profiler 记录中列出一个固定输入场景的算子、张量形状、精度、
数据依赖和重复次数。ArchExplorer 不会自动下载模型或把任意 ONNX/PyTorch 图转换成受支持仿真。
下面只需要形状；不需要输入值或权重值，也不计算模型预测或准确率。

## 视觉基础模型：token 投影

假设一张 224×224 图像已经按 16×16 patch 转成 token，且不含 CLS token，
则 token 数是 `(224/16)² = 196`。为使教程轻量，选择合成 embedding 宽度 192。
一个 Q/K/V 合并投影可以抽象为：

```text
X[196,192] @ W[192,576] -> Y[196,576]
m = token 数 = 196
k = 输入通道 = 192
n = 三倍通道 = 576
```

这不是完整 attention：只算投影，尚未包含 QKᵀ、softmax、AV、输出投影、残差或 MLP。
更不是对某个视觉基础模型的官方配置描述。patch 编码的开销也在本例边界之外。
该形状对应 `examples/tutorials/vision-projection.json` 的：

```json
{"kind":"operator","name":"synthetic-vision-projection","params":{"op":"matmul","batch":1,"m":196,"k":192,"n":576,"dtype":"fp16"},"weight":1.0}
```

改变分辨率时先重算 token 数；如果模型带 CLS、重叠 patch、动态裁剪或多尺度输入，
应按实际图中的形状计算，不能固定套用 196。batch 与 token 是不同维度。
批量图像共用一份权重时还应明确 `shared_b: true`；本例 batch=1 无此区别。
输入/权重/输出的最低数据体积可由 `(m*k+k*n+m*n)*2` 字节估算，
但实际流量取决于复用、mapping、padding、容量和通信模型，不能直接拿这个式子当访存计时。

## 具身 AI：单次决策的融合特征投影

假设上游编码器已生成一个 256 维融合特征，它可以来自视觉、机器人本体状态等；
这里不模拟这些上游编码器。一个合成策略头将它投影为 64 维中间表示：

```text
X[1,256] @ W[256,64] -> H[1,64]
m = 当前控制步的一条观测 = 1
k = 融合特征维度 = 256
n = 中间表示维度 = 64
```

配套 `embodied-projection.json` 模拟这一 GEMM。64 不是指定机器人的关节数，
还没有偏置、激活、动作输出层或动作执行。它也不是闭环强化学习或机器人环境仿真。
本例使用合成非 GPU DMC；切分、局部存储与通信共同影响结果，更大硬件未必更快。

如果真实计算是矩阵向量乘，可另建 `operator`/`mvm` 场景，使用
`params={"op":"mvm","batch":1,"m":64,"k":256,"dtype":"fp16"}`。
这里 MVM 的 `m` 是输出维度，与上方 GEMM 的 `m` 含义不同；模型调度也不同，
应记录实际采用的计算形式，不能把二者中更快的结果事后当作实测实现。
把 GEMM 的 m 改为 8 表示一起处理 8 行，不会自动表达 8 个相互依赖的控制步。

控制系统的完整截止时间还要包含采样、图像处理、传输、编码器、策略余下部分、
运行时与执行器。不能用这个片段的 ns 结果证明系统达到了某个控制频率。

## 多片段与真正的图

`workloads` 中的多项是分别评估的场景或片段；目标为 `Σ weight * latency_ns`。
它们不共享中间 tensor，不建立相互依赖，也不会自动获得跨算子融合、缓存复用或流水重叠。
weight 不会自动归一化。如果把 layer 重复次数作为 weight，结果只是相应的加权估算，
不是完整网络仿真。

完整自定义图通过 `arch_explorer.workload.api` 的 Workload 明确 tensor/producer 关系，
用 `graph.to_dict()` / `Workload.from_dict(data)` 保存为标准 JSON，并以
`WorkloadSpec('graph', graph.to_dict())` 运行，无需修改主项目或增加模型 kind。
新算子通过注册的 OperatorDefinition 提供 validate/lower；有限浮点属性由算子语义检查，
不能直接把任意 Python IR 对象写进 JSON。
见[外部负载教程](../workload-extensions.md)：从 ADF 文件读取硬件，在真正仓库外安装的
新算子包中构建分支图，再用公开 mapping 和现有 events 执行。
现有 Transformer 或单查询 attention 的接口有自身假设，不能因名称相似就替代任意视觉
attention、diffusion、卷积、稀疏算子或动态控制流。新算子还需要资源需求与 mapping 支持。

记录一份实验说明：输入形状/精度来自哪里、计时从何处开始到何处结束、权重和激活驻留哪里、
哪些算子已建模、哪些仍未知。随后再[建立目标架构](03-architecture.md)。

具名完整ViT/CLIP的入口、支持版本和命令见[完整模型教程](05-full-models.md)。上面的两个合成投影片段保持原有边界。
