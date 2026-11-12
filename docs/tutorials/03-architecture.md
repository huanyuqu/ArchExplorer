# 3. 建立自己的非 GPU 架构并仿真

前两章的投影例子使用合成 DMC：4 个 vector core，每核私有 SRAM，共享 DDR DRAM 和 mesh。
参数来源是教学假设，没有真实 GPU 含义。原生 GPU 组件见[完整程序教程](05-full-models.md)。

已有 ADF 文件时先读文件，再转换为内存描述；不要把文件名当作硬件模型：

```python
import json
from pathlib import Path
from arch_explorer.adf import from_dict

adf = from_dict(json.loads(Path("my-dmc.json").read_text()))
print(adf.to_dict())
```

下面演示初次构造并保存该文件。这个 ADF 对象本身尚未实例化资源或执行算子；
evaluate/prepare 根据注册组件和 mapping 生成执行模型。更改文件后需要重新读取和准备。

```python
import json
from arch_explorer.adf import Param, MappingSpec, WorkloadSpec, evaluate
from arch_explorer.archs import dmc_adf

adf = dmc_adf(num_cores=4)
workload = WorkloadSpec("operator", {"op": "matmul", "m": 196, "k": 192, "n": 576})
mapping = MappingSpec("dmc_partition", {"split": "n", "weights": "local", "activations": "local"})
result = evaluate(adf, workload, mapping)
print(result.to_dict())
adf.child("core").child("sram").params["capacity"] = Param(
    2**21, "B", "synthetic tutorial: 2 MiB per core")
print(evaluate(adf, workload, mapping).to_dict())
with open("my-dmc.json", "w") as file:
    json.dump(adf.to_dict(), file, indent=2)
```

比较 core 数时重新调用 `dmc_adf(num_cores=...)`，让网络端点数与拓扑匹配。
所有参数保留单位和来源；容量不足时保留失败，不能关闭检查冒充原场景。
构造新 ADF 时显式声明 compute、memory、network，模板按结构检查合法性。
[组件契约](../components.md)和[结构编辑](../structure.md)说明复用方式。

想同时接入新算子图和自定义阵列，可使用[独立负载包教程](../workload-extensions.md)。
其中的硬件来自独立 ADF JSON，lanes 和 clock_ghz 是两个显式参数，mapping 的不同 tile
会产生不同任务；组件不根据完整模型名称分支。

继续阅读：[探索与结果分析](04-exploration.md)。
