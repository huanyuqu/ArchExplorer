# 1. 安装、搜索与重放

以下命令均从包含 `pyproject.toml` 的仓库或干净源码导出目录执行。
教程文件随源码提供；仅安装 wheel 不会在当前目录生成这些配置。

```sh
python -m venv .venv
. .venv/bin/activate
python -m pip install .
python -m arch_explorer --version
```

需要 Python 3.10+。基础依赖只有 NumPy；本教程不需要 MCP、SCALE-Sim、GPU 或模型下载。
开发时修改了 Python 源码可用 `python -m pip install -e .`，本教程只修改输入文件。

## 视觉投影：第一次搜索

```sh
python -m arch_explorer search --config examples/tutorials/vision-projection.json --policy sa --output results/tutorials/vision.json
python -m arch_explorer evaluate results/tutorials/vision.json --output results/tutorials/vision-replayed.json
```

第一个命令在合成非 GPU 4-vector-core DMC 上搜索两个片上网络布局和 GEMM mapping。
第二个命令读取保存的完整 ADF、负载和 mapping，独立仿真同一个最佳候选，**不重新搜索**。
输出目录自动建立。结果保存在 `results/`，不要把它们当作源码或公开校准数据提交。

```python
import json
from pathlib import Path

original = json.loads(Path("results/tutorials/vision.json").read_text())
replayed = json.loads(Path("results/tutorials/vision-replayed.json").read_text())
best = original["best"]
print("实际后端尝试数:", original["evaluations_used"])
print("目标和约束:", best["score"])
print("mapping:", best["mappings"])
print("时延(ns):", best["workloads"][0]["evaluation"]["latency_ns"])
print("独立成本估计:", best["cost_estimate"])
print("诊断:", best["workloads"][0]["diagnostics"])
assert best["fingerprint"] == replayed["best"]["fingerprint"]
assert best["mappings"] == replayed["best"]["mappings"]
assert best["score"] == replayed["best"]["score"]
assert best["workloads"][0]["evaluation"]["latency_ns"] == replayed["best"]["workloads"][0]["evaluation"]["latency_ns"]
print("重放一致")
```

把代码保存为临时 `.py` 后运行，或粘贴到激活环境的 Python 中。
同版本、相同后端应得到上述一致性；跨版本重放发生差异时保留两个结果和代码版本再排查。
时间单位是 ns；除以 1e6 才是 ms。`score.objective` 不一定是纯时延，见第 4 节。

## 具身策略投影：替换输入，保留流程

```sh
python -m arch_explorer search --config examples/tutorials/embodied-projection.json --policy greedy --output results/tutorials/embodied.json
python -m arch_explorer evaluate results/tutorials/embodied.json --output results/tutorials/embodied-replayed.json
```

两个输入只改变负载名称和矩阵形状，可用相同方式检查结果。
这里选用不同策略只是展示接口，不能据此比较 SA 与 greedy 谁更好：两者负载不同。

## 常见问题

| 现象 | 处理 |
| --- | --- |
| `No module named arch_explorer` | 确认使用已安装包的同一个虚拟环境；用 `python -m pip` 安装 |
| 找不到 config | 在源码根目录运行，或传配置的绝对路径 |
| `evaluation_failed` / `invalid_mapping` | 先恢复原始例子，检查错误中的 trials/attempts、算子与模板兼容性 |
| 实际调用数少于 16 | 16 是上限；重复结构会跳过，max_steps 也会终止搜索 |
| 面积/成本为 null | 指标未覆盖，保留未知；不能改成零来通过约束 |
| 结果出乎预期 | 核对精度、padding、memory/网络、mapping 和计时边界；勿先调系数拟合结果 |

继续阅读：[如何从自己的模型提取负载](02-workloads.md)。
