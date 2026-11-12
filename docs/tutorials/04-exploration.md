# 4. 控制探索、解释结果和接入 Agent

## 先固定问题，再运行搜索

配套 JSON 明确限定了教学实验：

| 层次 | 本例搜索空间 |
| --- | --- |
| 结构 | 保持 4 个 vector core、私有 SRAM 和 DDR DRAM，顶层 `chip/noc` 在原 mesh2d 与 bus 之间替换 |
| 硬件参数 | `parameter_samples=1`，每轮只有当前参数；容量、算力、带宽不随机增长 |
| mapping | 仅非 GPU `dmc_partition`，每轮最多抽取 4 个合法 mapping 组合 |
| 目标 | 单 workload 的时延；不启用面积/美元约束，罚系数为零 |
| 停止条件 | 最多 16 次实际 backend 尝试、8 个结构提议步骤；seed=17 |

`search.proposals` 是两个原子 delete/create 编辑组，均替换同一路径并保留其余资源。
回到已见 fingerprint 的结构会跳过，因此实际尝试通常少于上限。
这只是演示“选定架构变化后观察结果”，不是完整架构空间的穷举或公平的策略性能实验。
bus 和 mesh 的共享方式不同；网络输入值是合成假设，不代表面积等价的物理实现。

`mapping_samples=4` 不是穷举全部组合；各候选来自模板声明的 split/weights/activations 空间。
`max_evaluations` 统计环境层 backend 尝试，不等于所有内部模拟次数或墙钟成本。
失败的后端尝试也消耗额度；checkpoint/restore 不退款。
仅改变 seed 不会保证新硬件更快。同一源码、配置和操作序列便于复核，不代表统计显著。

## 进入参数探索

先按第 3 节生成几个固定参数场景，可清楚归因容量或带宽的影响。
之后可在配置中加 `parameter_domains`，例如明确授权 私有 SRAM 容量选项：

```json
[{"path":"chip/core/sram","param":"capacity","values":[524288,1048576,2097152],"source":"synthetic tutorial assumption: 0.5/1/2 MiB private SRAM"}]
```

将该数组放在 `parameter_domains` 字段，并把 `parameter_samples` 增至 2 或更多。
这会增加每轮参数候选和后端调用；包含基线，不保证每次抽到不同值。
**该数组不是所有参数的白名单**：组件已有 domain 的参数也仍可被采样，例如 DRAM technology。
当前实现每个新参数版本修改一个可调坐标；若要求其他参数严格固定，采用多个显式固定
配置、`parameter_samples=1`，或经环境工具设置明确参数再评价。不要把未限制的实验描述为
“只扫描 SRAM”。完整规则见 [三层搜索](../structure.md)。

## 成本与最佳可行结果

`best.score` 包括 objective、feasible 和约束信息。目标是加权时延加面积/成本超限罚项；
罚项太小可能使被提交的 `best` 超预算，不能仅看 CLI 退出码判断设计满足资源约束。
若要求严格预算，必须检查 `best.score.feasible`；不满足时报告“本次没有交付满足约束的最佳候选”，
并从保留的尝试记录中重新审计可行候选，而不是隐瞒超限。

`best.cost_estimate` 是独立成本模型，`raw` 与 `full` 口径不同，mm²、USD 和旧模型 mm²eq
不可互换。本例的 7nm 是合成参考场景。默认不设成本门槛不等于资源没有成本。
启用预算前阅读 [cost.md](../cost.md)，确认所用组件和工艺有估计支持；未知仍为 null，
启用缺少指标的约束会产生不可用目标，不能填零。

## 怎样读反馈

先看每个 workload 的 `evaluation.latency_ns`、`breakdown`、`details` 和 `diagnostics`。
任务累计耗时可能重叠，不能直接求和作为端到端时延；资源利用率依赖后端定义。
小 GEMM 的切分、内存容量/路径、局部暂存和网络竞争都可能决定结果。
为验证一个解释，应只改一个假设、保留其余条件并重放，不按期望速度调参数。

保存配置、代码 commit、seed、实际 evaluations_used、完整结果及 `search.trace_path`
指向的 JSONL。默认 trace 位于临时目录，应在清理前复制到自己的实验目录；也可预先配置
一个尚不存在的 `trace_path`。相同配置重复运行需用新的 trace 文件名。
`search.trajectory` 记录结构提议/接受/错误；完整工具 trace 可审计评价返回和预算。
保存最佳结果允许重放最佳候选，并不能替代保存全程 trace。

## Portal Resident / 外部 Agent

同一配置可作为有状态工具环境：

```sh
python -m arch_explorer env serve --config examples/tutorials/vision-projection.json
```

在该进程 stdin 逐行发送下面的 JSON，每行会得到一份 JSON 响应。这里固定架构只优化 mapping：

```jsonl
{"id":1,"tool":"describe_environment","args":{}}
{"id":2,"tool":"get_state","args":{}}
{"id":3,"tool":"evaluate","args":{"optimize":"mapping"}}
{"id":4,"tool":"submit","args":{"selection":"best","path":"results/tutorials/tools.json"}}
```

发送 EOF 结束服务后：

```sh
python -m arch_explorer evaluate results/tutorials/tools.json --output results/tutorials/tools-replayed.json
```

每轮都读取反馈再决定下一次操作。`search.proposals` 只限制内置搜索策略；它不是外部 Agent
工具调用的权限或硬约束，外部 Agent 仍须遵守约定的编辑集合和资源限制。
MCP 安装/Claude Code 配置、Portal Resident 的 subprocess 示例和完整工具定义见
[Agent 文档](../agent.md)。`agent run --policy scripted` 是离线脚本，不是调用 LLM。

可给 Agent 的任务模板：

> 用给定配置评估合成视觉投影；先读取环境和状态。固定 workload、精度、core 数及存储，
> 只允许配置中列出的两个 chip/noc 替换组，硬件参数固定，每轮仅优化 mapping。
> 总 backend 尝试不超过 16；不删除必要通信或存储。每次调用前记录简短选择理由，
> 读取返回的真实时延、约束、错误和剩余额度后再决定。保存 trace，检查 feasible，
> 提交 best 并独立 CLI 重放。报告未覆盖算子和模型边界，不声称完整视觉模型速度或真实设备精度。

完成这些步骤后，再按自己问题逐步增加负载、参数域和结构变化，并保留一份固定基线。
