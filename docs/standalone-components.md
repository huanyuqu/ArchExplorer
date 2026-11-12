# 在非 GPU 场景中调用原生组件

从公开工厂创建组件，调用者拥有它的生命期。原生组件处理自己的队列、仲裁、缓存或控制器机制；
通用 `ae_dynamic::Runtime` 处理端口连接、事务所有权、时钟相位和完成阶段。调用者提供外部请求，
无需继承 GPU 类。应从工厂的实际参数和依赖闭包核对独立性，再运行同进程隔离用例。

本页的双内存与双 ICNT 调用程序位于
[用户示例](../examples/gpu_instantiation/README.md#standalone-transport-caller)。它使用实际库接口，
运行所需的匹配原生静态库与公开 wheel 分开构建。代码或头文件可用不等于实际运行通过。

## 非 GPU 内存请求源

公开 `memory_system.hpp` 中的 `MemorySystem` 是固定 source↔L2↔DRAM 拓扑的便捷组装层。
它调用与 GPU 组装相同的 `make_l2` / `make_dram`，并拥有本组的配置、周期视图、事务分配域、
统计、端点及 Runtime。工厂之间借用各自所需的窄服务，没有 GPU 程序或整机对象传入。
需要审查直接工厂的服务字段和销毁顺序时，使用匹配版本的组件公共接口与构建依赖说明。

外部调用程序只提供内存参数和请求：

```cpp
namespace native = ae_accelsim::standalone;
native::MemoryOptions options;
options.channels = 2;
options.subpartitions_per_channel = 1;
options.cache_sets = 4;
options.cache_ways = 1;
options.controller_queue = 32;
options.simple_dram = false;
options.cache_period = 1.0;
options.dram_period = 2.0;
native::MemorySystem memory(options);
auto request_id = memory.submit(0x1000, 32, false);
memory.tick();
ae_dynamic::Transaction response;
while (memory.receive(response)) {
    // Correlate response.id with the submitted request metadata.
}
```

`native` 指 `ae_accelsim::standalone`。这段片段仅演示接口，一次 tick 不代表请求完成。
完整[调用程序](../examples/gpu_instantiation/standalone_memory.cpp)继续推进至请求全部收到且
`drained()` 为真，期间不创建 SM、kernel、CTA 或 warp 服务。`receive` 只取回已产生的响应，
不推进缓存或控制器。`snapshot()` 保留原生 L2/DRAM 指标及独立的 consumer、physical、resource
完成计数；不能以响应到达代替整个物理系统排空。

请求大小为 32B 的整数倍，单次不超过 128B，地址按 32B 对齐且请求不跨越 128B 行。
响应保留行内字节掩码，读回复为 `read_reply`，写确认为 `write_ack`。`transfer_bytes` 是真实
传输长度，不能把它无条件等同于逻辑请求的 `bytes`。示例校验这些关联，不声称模拟了内存数值。

多 sector 逻辑请求须等所有实际 sector 回复后才返回一次凭据，ID 与逻辑提交相同、parent_id=0。
byte_mask 合并有效字节，transfer_bytes 累加真实响应：32B sector 读回复为40B，写确认为8B；
因此64B读的聚合流量为80B。submitted/received计逻辑请求，三个生命周期计实际wire及内部流量，
不能把物理计数强行设为逻辑请求数。已就绪凭据未被用户取走不阻止物理排空。

示例使用详细 DRAM、单 chiplet 和 LRC 关闭的组装范围。它按阶段发出热行读取、写入和超过缓存
容量的冲突流，检查实际命中/未命中、读写、脏行写回及反压。两个系统用不同 channel/cache/时钟
配置交错推进，销毁或构造失败的另一个实例不能改变 B；B 最终还要与单独运行的同配置控制组相等。
固定便捷拓扑不代表已覆盖所有直接工厂的任意连接或所有原生配置。

便捷配置使用 sector cache、LRU、writeback、fetch-on-write 和 linear index。用户设置显式几何，
不传一串通用 GPU 配置项：channels 为 1..64 的 2 次幂，subpartitions/channel 为 1/2/4，
cache_sets 为 1..65536 的 2 次幂，cache_ways 为 1..1024，组合缓存容量最多 1 GiB；
controller_queue 为 1..65536，时钟周期有限且为正。更广的原生策略需使用底层工厂的明确配置。

底层工厂的借用与生命期要分别核对：

| 依赖 | 所有者与使用方式 |
| --- | --- |
| 配置与周期视图 | 组装层拥有；组件读取自身参数及 `CycleView`，不从 GPU 取时间 |
| 请求 ID 与事务域 | 每个组装单独拥有；负责关联、分配和归还，取消不标为完成 |
| 内存统计 | 组装层拥有；工厂得到明确的统计引用，不要求外部 source 配置 SM/warp |
| L2 | `make_l2` 返回拥有实际实例的 `unique_ptr` |
| DRAM | `make_dram` 返回 `unique_ptr`，显式借用本分区的 L2 |
| 程序/CTA 执行服务 | 仅 SM 工厂需要；内存与 ICNT 不创建这些服务 |

先销毁借用 L2 的 DRAM，再销毁 L2，之后释放 statistics/config/context，最后关闭请求 arena。
`MemorySystem` 管理这组依赖；直接工厂的调用者须遵守同样次序。匹配版本的原生依赖说明应逐项
列出保留底层类型的字段用途，不能只把整机宿主改名为 Context。

## 创建和组合 ICNT

公开头文件 `components/accelsim/component_native/icnt_api.hpp` 提供：

```cpp
namespace native = ae_accelsim::standalone;
native::IcntConfig config;
config.mode = 2;
config.request_nodes = 1;
config.memory_nodes = 1;
config.local.input_buffer = 2;
config.local.output_buffer = 2;
auto network = native::make_icnt(config);
```

工厂返回调用者持有的 `std::unique_ptr<Icnt>`，不接收 GPU、kernel 或指令执行服务。
mode 2 使用原生 LocalInterconnect；mode 1 使用 BookSim，另需它的配置文件。示例只选择
mode 2、一对请求/内存端点，不将这一用例外推为所有 BookSim 拓扑已验证。

`has_buffer(input, transfer_bytes)` 是准入查询；`push(input, output, transaction, packet_type)`
提交请求或回复。拒绝时调用者保留事务并重试，不能创建重复逻辑请求。`advance()` 推进一步原生互连，
`peek(output)` 查看输出头；仅在下游接受后 `pop(output, result)`。请求端点编号在前，内存端点编号
在后；示例使用请求 0、内存 1，回复走相反方向。`Transaction` 保存 ID、地址、逻辑长度、传输长度、
字节掩码和顺序；读写请求/回复类型由公开 `IcntPacketType` 指定。

外部网络适配器实现通用 Runtime 的被动 `can_accept/accept/drained`，把准入和入队委托给原生对象。
接受回调只接收事务，下一次显式相位再发送回复，避免递归路由。示例通过四条普通连接形成：

```text
source.request → icnt.request → echo.request
source.reply   ← icnt.response ← echo.reply
```

每个实例注册自己的时钟和相位，Runtime 及端点在组件使用期间保持有效。相位回调仅借用对应节点，
不把拥有全部配置和状态的宿主对象传给工厂。示例中的 echo 只产生传输回复，不模拟 L2 或 DRAM。

## 运行、来源和完成

先安装主库 wheel，再从仓库外调用示例的 `standalone.py`，传入匹配的独立 SDK manifest 和
`--target memory` 或 `--target icnt`。SDK manifest 使用
`arch-explorer.accelsim-standalone-sdk.v1`，保存实际 archives、各自构建收据及依赖；
`native_sdk.load_sdk` 验证安装头 SHA 和这些引用。保留被引用源文件和依赖的路径与哈希。
构建 SDK 本身依赖受控上游提取源码及工具链；编译这两个外部调用程序仅使用公开头与静态库。
保留上游来源、许可与 NOTICE，不能用整 GPU 可执行文件替换 SDK，wheel 也不携带第三方二进制。

runner 使用公共 `native_sdk.compile_consumer` / `run_consumer`，不复制链接与进程管理协议。
组件包保存源码/二进制/SDK 绑定、ELF 构造审计和 `PROCESS.json`；示例保留这些公共收据与
安装包前后文件哈希。默认每个 CPU 子进程组最多 600 秒、8 GiB，编译一个 job，失败或超时回收
整个进程组并留下原件。`--memory-mib` 可降低运行预算，编译预算保持 8 GiB。接口或依赖缺失时失败，
不改用 Python 互连模型、手工 header alias 或整引擎包装。结合库的来源和实际构造证据检查
整机宿主依赖，不能仅凭文件名含有 standalone 就认定独立。

调用程序创建缓冲和时钟不同的两个互连实例，交错推进，使用重叠事务 ID 和部分相同地址验证命名空间
隔离。它在 B 仍有流量时销毁 A，保存 B 的前后原生统计，并验证无效的第三次构造不会改变 B，随后
完成 B。下游延迟接收制造反压；程序核对响应 ID、地址、长度、掩码、读写类型和恰好一次完成。
最终同时要求原生包、flit、credit 和 Runtime 事务排空。

纯传输用例在回复交付后记录自己的消费/物理/资源完成。这一条件不能照搬为 DRAM 的完成条件：
内存可能还有写回、控制器或资源归还尾部，应使用内存组件的实际完成与排空接口。

`run.json` 是这个外部调用程序的运行收据，保留其断言和原生指标，不是 provider 的 `Evaluation`。
没有执行或失败时不能声称独立库复用成功。GPU 重组的行为保持仍用
[组件与原版的公共比较器](gpu-instantiation.md#用公共比较器核对)单独验证，不能由 ICNT 用例推断。
