# 非 GPU 合成架构与保留负载生成器

这些生成器描述 systolic/vector 资源、负载和显式设备互连，使用 `backend="events"`。
所有硬件数值均为合成假设，结果不代表 GPU 性能、物理时延精度、真实权重数值推理或机器人闭环。
原生 GPU 的 ViT/CLIP demo 继续使用 [Accel-Sim 程序入口](tutorials/05-full-models.md)。

旧 GPU 专用 `arch_explorer.devices.scans` 的 matmul_sweep、softmax_sweep、a100_operators、
prefill_sweep 及模块 CLI 已退役，论文图默认配置不再作为公共 sweep 入口。
通用 evaluate、search、显式 workload/mapping 和保留的非 GPU presets 继续使用原接口。

## 明确的硬件构造接口

```python
from arch_explorer.archs.adf_seeds import non_gpu_adf

adf = non_gpu_adf(num_cores=2, matrix_precision="bf16",
                  dram_capacity_bytes=64 * 2**30)
print(adf.to_dict())
```

`num_cores`、`dram_capacity_bytes` 须为正整数，拒绝 bool；`matrix_precision` 仅 fp16/bf16/fp32。
每核包含 32×32、1 GHz、WS/double-buffered systolic array，矩阵操作数使用所选精度；
另有 16-lane、4-SFU、1 GHz 的 FP32 vector unit，显式开启 FP32 GEMM/MVM，承担 FP32 矩阵、
向量、转换及归约能力。每核 4 MiB SRAM、128 B/ns 局部 bus，chip 通过按端点数构造的 mesh
连接各核与 DRAM。DRAM 默认 64 GiB、64 B/ns、100 ns；参数逐项附 synthetic 来源和单位。
这些值没有按旧 GPU 预测配平，不是任何厂商器件参数。

算子精度由负载契约指定。BF16-only 阵列不能接收 FP32 attention；FP32 vector 保留该能力。
缺少相应矩阵精度时拒绝执行，不把 FP16、BF16、FP32 当作可互换标签。
原 `dmc_adf` 不变；不提供 `gsm_adf` 或旧 GPU 别名。

## 从安装包与公开源码生成

在源码 archive、export 或解开的 sdist 中，安装相应 wheel 后运行：

```sh
python tools/generate_pi05.py /tmp/pi05-inputs
python tools/generate_multi_device.py /tmp/multi-device-inputs
python tools/generate_multi_device_execution.py /tmp/multi-device-execution-inputs
```

三个脚本都保留原 `OUTPUT_DIRECTORY` 参数及 `--help`。输出文件名和 Python
`configuration()` / `cases()` 入口保留；产物使用新 seed，不只是修复脚本导入。
脚本属于公开源码工具，wheel 提供生产包；脚本不会向 Python 路径注入某个开发工作树。

π0.5 生成器先调用既有 `normalize`，默认 dtype 为 **bfloat16**，对应 BF16 阵列；float32 对应 FP32
阵列，两者始终保留 FP32 vector。输入示例的三路视觉、prefix KV、动作专家、flow、mask、缓存与
输出 shape 来自 `openpi.pi05.v1`。FP32 初始化权重、共享 BF16 转换副本、算子级精度、累积/归约及
FP32 输出保持原图定义，不能把整个图强制设为 BF16 或 FP32。64 GiB 是显式合成容量，用于容纳
初始化与转换后的权重，不是推断出的设备事实。

```python
from tools.generate_pi05 import configuration

bf16 = configuration(profile="openpi_pi05_dummy", num_steps=1,
                     action_dim=1, action_horizon=1, max_token_len=1)
fp32 = configuration(profile="openpi_pi05_dummy", dtype="float32", num_steps=1,
                     action_dim=1, action_horizon=1, max_token_len=1)
```

生成和图/组件能力核对不等于完整 events 运行。完整 π0.5 执行要求映射的 applicable/apply 与
events 声明的 `openpi.pi05.v1` 一致；未支持时明确报错。执行与账本边界见 [π0.5 文档](pi05.md)。
π0.5 原生 Accel-Sim、多设备 π0.5 或 PIM 扩展不由本生成器提供。

## 多设备、切分与 FP32 通信

`generate_multi_device.configuration(names, workload)` 保留任意合法的唯一设备名称，邻接方向
和非对称带宽由显式 topology_edges 决定。默认完整 ViT/CLIP 为 FP16，其他输入由 workload IR 的
dtype 选择阵列精度，FP32 vector 始终存在。空/重复设备名和非法负载会被拒绝。

`generate_multi_device_execution.configuration(kind, count, profile, tp)` 保留逐算子 placement，
TP 案例将真实 K 维按 rank 切分，并记录 FP32 tree allreduce。输入矩阵与最终 FP16 输出不因
FP32 partial sum/归约而被改为全图 FP32。count 须为正整数、tp 须为 bool，未知 kind、
不适用的 profile 或非法切分/拓扑由生成器及正式请求校验拒绝。

可以从完整执行生成器的 `matrix-tp.json` 做小型 CPU 验证，或通过公开对象独立评价：

```python
from arch_explorer.adf import MappingSpec, WorkloadSpec, evaluate, from_dict
from tools.generate_multi_device_execution import configuration

config = configuration("operator", 2, tp=True)
result = evaluate(from_dict(config["seed_adf"]),
                  WorkloadSpec(**config["workloads"][0]),
                  MappingSpec("multi_device", config["mapping_selectors"]["multi_device"]),
                  backend="events", cost_model=False)
print(result.to_dict())
```

检查 K 分片覆盖、FP32 partials、归约 additions、每条消息的 bytes 与输出 dtype。拓扑可达、容量、
精度和执行支持是独立条件；不能仅凭生成了 JSON 就宣称整个负载执行通过。
