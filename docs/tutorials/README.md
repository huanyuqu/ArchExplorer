# ArchExplorer 负载驱动教程

原生 GPU demo 从完整小模型走到真实程序执行；通用搜索教程使用合成非 GPU vector many-core。
两者都不下载权重或调用付费 API。原生模拟还需要显式安装外部引擎。

1. [安装、首次搜索与独立重放](01-quickstart.md)：非 GPU 合成算子闭环。
2. [描述自己的负载](02-workloads.md)：形状、依赖及覆盖范围。
3. [建立非 GPU 架构](03-architecture.md)：DMC 组件、参数和评价。
4. [受限探索与 Agent](04-exploration.md)：预算、约束和可行结果。
5. [ViT/CLIP 原生 GPU demo](05-full-models.md)：完整小前向、ADF、mapping、原生结果与新形状。
6. [标准模型语义来源](06-full-model-sources.md)：不等于标准模型原生支持。
7. [基线工具的适用范围](07-model-baselines.md)：非 GPU 搜索和历史结果边界。
8. [接入自己的组件或引擎](../component-extensions.md)：独立包、注册、mapping、生成与重放。

`examples/tutorials/` 的视觉/具身例子只是单个合成投影，不是完整模型或机器人闭环。
默认原生 demo 的 tiny 配置由核心场景明确声明，不冒充标准 ViT/CLIP。
