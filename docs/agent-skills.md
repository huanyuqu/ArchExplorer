# 面向外部 agent 的 ArchExplorer skills

本发行的 `skills/` 提供三个独立、可复制的skill目录：

| Skill | 用户任务 |
| --- | --- |
| [arch-explorer-run](../skills/arch-explorer-run/SKILL.md) | 准备、执行、比较与解释仿真结果 |
| [arch-explorer-workload](../skills/arch-explorer-workload/SKILL.md) | 接入新算子、新图、mapping及自定义架构 |
| [arch-explorer-component](../skills/arch-explorer-component/SKILL.md) | 接入组件执行模型或已有模拟器 |

将所需目录复制到外部agent支持的skills目录（例如Codex的 `$CODEX_HOME/skills`，未设置时 `~/.codex/skills`）。如该环境不自动发现skills，也可以明确让agent读取相应SKILL.md。无需Portal账号、内部工作区或团队运行工具。

skills引用的文档路径均相对匹配版本的ArchExplorer项目目录，复制skill后仍需让agent定位公开源码分发或对应文档。skill不会安装依赖或下载权重；实际操作由agent依据用户任务执行。可给出这样的请求：

- “读取arch-explorer-run，使用我的架构和场景，比较两个mapping并解释结果。”
- “读取arch-explorer-workload，为一个带分支的图新增算子，在我提供的ADF上执行。”
- “读取arch-explorer-workload，在独立包中为同一算子提供两个目标实现，其中一个含多步和临时张量，再接既有 GPU 目标。”
- “读取arch-explorer-component，将我的SRAM模型接入现有事件框架。”
- “读取arch-explorer-run，按匹配版本的组件实例化教程运行完整小 ViT/CLIP，并与独立 Accel-Sim oracle 做软件差分。”

GPU 组件任务先阅读 [GPU 实例化流程](gpu-instantiation.md)，检查所选版本实际支持的组件、
事务与时钟能力。skill 的指导不构成该版本已经实现或验证这些能力的证据；两侧原始结果及
公共比较报告才承载本次差分结论。接口未具备时明确报告，不将 whole-engine 重复执行解释为组件化。

新执行实现流程见[外部算子教程](operator-implementations.md)。先区分语义、目标实现、mapping 和
硬件机制；普通图复用公共策略，不让 agent 为每个算子复制调度状态机或新增 provider 分支。

公开源码导出和sdist携带skills；wheel提供执行库，不保证携带这些项目级资料。skills不会把合成示例解释为真实设备精度。新增负载是否可在特定GPU引擎执行，仍由其能力合同决定。
