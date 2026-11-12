# 多设备拓扑与探索

通用 ADF 层次、重复实例、互连、结构编辑和显式 placement/sharding/routing 契约继续保留。
这些机制独立于 GPU 物理时延实现。原来包含 `gpu_sm` 的公开多 GPU 配置已移除，
不会把这些配置静默解释成非 GPU 硬件。新的 GPU 完整程序路径见[原生 demo](tutorials/05-full-models.md)。

创建非 GPU 多设备 ADF 时，在各设备子树声明 compute、memory 和本地网络，再在父容器声明互连。
通过 `describe_environment().result.mapping_spaces` 发现当前负载与设备适用的模板及条件分支，
不能因拓扑合法就假设任意模型可执行。结构实例数计入所有祖先的 repeat。
面积、成本与物理执行能力分别检查，未知指标不能补零。

隐式发现不凭空请求 `multi_device`；显式请求须携带完整 placement、sharding 和 routing。
[执行约束](multi-device-execution.md)、[通用接口](integration.md)和[Agent 工具](agent.md)说明入口。
