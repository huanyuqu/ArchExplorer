# Accel-Sim 组件拆分：边界与验证状态

以下保留基线版本的源码审查与设计依据。该基线仅支持 components.accelsim 的 joint provider，
尚无可独立连接的 SM/L2 实现。新的组件实例化和原版差分流程见
[GPU 实例化](gpu-instantiation.md)；未完成实际安装与机制验证前，不能把目标组织当作已实现。
新负载扩展本身也不扩大原生引擎的 ISA、精度或图支持范围。

## 审查基础

检查了项目当前引擎声明的GPGPU-Sim基线 `c24e8858eee552b3bc2919b147f14e2a88571775`（本地构建另含显式变体补丁），并对照上游代码。网页dev分支可能变化，实际提取必须绑定确定版本与补丁。

- [gpu-sim.cc](https://github.com/accel-sim/gpgpu-sim_distribution/blob/c24e8858eee552b3bc2919b147f14e2a88571775/src/gpgpu-sim/gpu-sim.cc)：`next_clock_domain`和`gpgpu_sim::cycle`调度CORE/L2/DRAM/ICNT；缓存接收前检查队列，响应经互连回流。
- [l2cache.h](https://github.com/accel-sim/gpgpu-sim_distribution/blob/c24e8858eee552b3bc2919b147f14e2a88571775/src/gpgpu-sim/l2cache.h)：memory_sub_partition暴露full/push/top/pop/cache_cycle及L2到DRAM队列；仍引用gpgpu_sim/config/stats/mem_fetch，尚非独立库接口。
- [shader.h](https://github.com/accel-sim/gpgpu-sim_distribution/blob/c24e8858eee552b3bc2919b147f14e2a88571775/src/gpgpu-sim/shader.h)：SM包含warp、scoreboard、执行流水线、load/store与同步机制，依赖GPU上下文。
- [gpu-sim.h](https://github.com/accel-sim/gpgpu-sim_distribution/blob/c24e8858eee552b3bc2919b147f14e2a88571775/src/gpgpu-sim/gpu-sim.h)：sst_gpgpu_sim有SST_receive_mem_reply和SST_cycle等外部内存接入边界。它说明拆接具有先例；本次没有运行或验证SST集成。

## 判断

可行。复用对象是原有状态机、队列和周期推进机制。仅给SM/L2分别包装run函数或分别计算一次总延迟，无法保存相互作用。模块的独立测试能力与完整系统中的闭环协同需要同时满足。

L2/DRAM/互连的事务边界通常比SM内部机制更适合跨硬件复用。重用L2时仍须明确请求粒度、地址映射、sector/line、写策略、MSHR/合并、原子与一致性范围；原GPU缓存不自动成为带CPU一致性的通用cache。SM适合其他SIMT或GPU相近架构，不能仅靠改名复用为任意NPU矩阵阵列。

## 目标组织（提案）

```text
组件包：SM/warp执行 → L1与访存合并 → 互连 → L2分区 → DRAM控制器
                               ← 响应与反压 ←
通用框架：实例连接、事务路由、时钟域、事件排序、生命周期
ADF：组件实例与参数、端口连接、时钟/地址映射声明
mapping/前端：operator图 → 分块/程序或trace → SM可接受的执行输入
```

组件机制保留在components中。可通过C++适配层向Python暴露公开合同，避免每个内部周期都跨Python调用；只有证明无可见外部事件的区间才允许批量推进。提取第一阶段优先保持原tick序与时钟相位，不能先引入乐观跳时再用近似总延迟补偿。

## 需要补的框架合同

当前submit/next_event_time/advance主要消费预先声明的任务并报告完成/可用事件，尚无完整的动态子请求端口与响应路由。SM执行时产生的cache miss及其返回会影响后续发射；用静态TaskGraph预先枚举并不足够。

需要定义独立于GPU名称的事务接口，至少包括：

- 请求ID、父请求、源/目的端口、读写/原子类型、地址/长度/byte mask、排序约束和必要执行上下文。
- 接收/拒绝以及credit/queue可用事件；拒绝无副作用，所有队列有界；所有权转移与响应匹配唯一。
- 时钟域、同时间phase顺序、跨域可见时间、最早安全事件时间；发生同时间链式事件时推进至稳定状态。
- 请求产生与响应派发由框架协调，组件不可越过调度器私自advance另一个组件。
- 消费者就绪、物理完成、资源归还、kernel完成和系统排空分别建模；当前事件图消费者全部完成不代表全部物理事务排空。
- 每次评估独立的上下文/状态与统计，消除必须共享整个GPU单例的依赖；一个缓存只拥有一份真实状态，不另加外层Demand重复收费。

接口需在两个不同组件族上验证通用性，而非把mem_fetch指针或GPU全局上下文作为通用框架协议。mem_fetch可以留在原生适配层内部。

## 建议实施顺序与验收

1. 固定原生版本、配置、输入和小程序，保存原始完整引擎结果作为差分基线。
2. 提取L2+DRAM子系统，使用可控请求源驱动，检查命中、未命中、合并、写回、队列满、返回顺序和排空；再用非SM请求源证明复用。
3. 拆出互连与时钟适配，验证争用和反压形成闭环，无请求丢失、重复完成或无进展。
4. 接SM/cluster以及trace/kernel生命周期，保留完整SIMT与指令依赖。重新组装与原配置等价的GPU，比较逐边界请求、周期、缓存统计、停顿及完成顺序；相同配置下的偏差应解释和修复，不用拟合系数盖住。
5. 在行为等价之后替换一个组件或拓扑，研究新的硬件组合。功能输出与物理精度分别验证。

Trace驱动保持记录的动态指令和地址流；会改变程序路径、地址、同步行为的映射或架构变化可能需要重新生成输入，不能将一份trace视为所有新硬件都成立。原生整体结果一致只证明提取没有改变已覆盖行为，真实硬件精度仍需独立测量。

上述内容是先前可行性评审，不是新组件路径的完成声明。joint 执行保留为已知路径及独立差分
基线；新路径的可安装接口、实际覆盖及限制由实例化教程与本次比较原件说明。
