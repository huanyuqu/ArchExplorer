# 尺寸请求与受限阶段运行

`cases.py` 只生成参数完整的请求，不默认执行大模型：

```sh
python cases.py --kind llm --batch 1 --seq-len 2048 --dtype fp16 --tp 1 --output LLM.json
python cases.py --kind mla --width 64 --query-rank 16 --content-dim 64 --output-width 64 --dtype fp32 --output MLA.json
python cases.py --kind export --directory exported --output EXPORTED.json
```

LLM 使用公开注册定义的 Llama2-7B/70B、Llama3-70B、Qwen-72B 全部声明层数，prefill/decode 分开。工作量是层级性能图，不含数值权重、embedding 或生成准确率。batch/context/dtype/tp 是本次实验明确选择；不能把一层短序列控制叫完整模型规模。

MLA 记录四行 B={1,4}、S={2048,8192}、H128、latent512、position64，其余投影宽度/dtype 必须明确给出并列为实验选择。64 tiles、每私有 tile 1MiB、MAC16×16 与4×16/8×8/64共享候选单列；共享容量必须给精确 bytes，不把论文 MB 当 MiB。请求生成不分配 cache。

export 请求读取真实 EXPORT/INPUT-SHAPES，记录文件大小和原导出节点数，不载入数值根或权重。声明摘要不是来源准入，文件大小不是峰值内存估计。三个入口后续都必须先走权威 workload preflight，构图、编译、服务与数值参考分别记录。

`preflight.py --cases CASES.json --index N [--resource-budget JSON] [--build-budget JSON] [--cache-replicas N] [--tiles JSON] --output NEW.json` 只调用正式预检：导出模型用公共 onnx_preflight，MLA 用公共 mla_preflight，LLM 性能图用公共 transformer_preflight。MLA 的 build_budget 与通用编译 resource_budget 是独立账本；LLM 从该行 workload.params 透传完整 model/phase/batch/seq_len/layers/dtype/iterations，以该行 tp 为实际并行度，显式 --resource-budget 覆盖 params.resource_budget，否则保留行内预算。输出原样保存生产者返回的完整结果；admitted=true 对应 admitted/退出0，false 对应 budget-refused/退出2。参数非法时保存 failed 和原诊断。CORE 不复制估计公式或裁减完整层数。reference_mode=inputs 只验证根值，不能代替整图数值。

例如已有 LLM.json 的第0行可运行 `python preflight.py --cases LLM.json --index 0 --resource-budget BUDGET.json --output PREFLIGHT.json`。新增规模阶段应由 scale-run 在共享锁内执行，并等待当前持锁窗口结束；这条命令不会自动启动构图或服务。预算使用公共 v1 字段，生产者返回完整规范化 budget 和逐项 violations，不在示例预先填写资源答案。

MLA 分片使用显式 `--streaming-plan STREAM.json`，STREAM 恰含 `chunk`、`slots`、`capacity`（bytes）、`cache_mode`（sharded/external）。例如 `python preflight.py --cases MLA.json --index 0 --streaming-plan STREAM.json --output STREAM-PLAN.json`。该行必须声明 fp32；其完整 batch/sequence/heads/ranks/width 透传正式 `mla_stream_plan`，不裁剪维度。返回状态为 plan-estimated，退出0仅表示拿到静态估计；原始 root_fits 和 conservative_plan_fits 分列，不将保守上界当编译/容量或服务通过。旧命令不加该选项时仍调用 dense/canonical mla_preflight。stream plan 没有 host/build/replica 预算参数，混用旧 `--resource-budget/--build-budget/--cache-replicas/--tiles` 会拒绝；真正构图的完整 resource_budget 放在下述参数 JSON。

`block.py` 把正式流式图接到普通编译路径。`PARAMETERS.json` 是 `mla_stream` 的完整关键字对象，包含全部 canonical 根、完整维度、chunk/slots/cache_mode、dtype/epsilon 及明确的 resource_budget；根名和实际分片均由公共 workload API 决定。ADF.json 是调用者的真实硬件。compute/memories 按槽顺序给原路径，不按名字猜 rank/group，不重排列表。

Python 配置入口是 `arch_explorer.archs.block_compilation.block_scene(parameters, *, adf, compute, memories, external_memory=None)`，返回(adf,graph WorkloadSpec,mapping)。预检入口继续是 `arch_explorer.archs.paper_scale.preflight_scale_case`，新增可选关键字 streaming，其余签名兼容。

```sh
python block.py --parameters PARAMETERS.json --adf ADF.json \
  --compute board/tile0/mac board/tile1/mac \
  --memories board/tile0/sram board/tile1/sram --output block-inputs
archexplorer compile block-inputs/SCENE.json --output BLOCK-COMPILED.json
archexplorer prepare BLOCK-COMPILED.json --output block-request
archexplorer evaluate block-request/COMPILATION-REQUEST.json --output BLOCK-RESULT.json
```

上例路径是占位示意，必须来自实际 ADF/topology，数量必须由正式 block_mapping 核到 slots。external 模式另给 `--external-memory` 实际外存路径及有容量、有路由的硬件；搬运、读栅栏和工作区沿共同语义计费。输出保存完整普通 graph WorkloadSpec、v4 MappingSpec 与 SCENE，不复制构图/布局/资源公式。RESULT 的 scene-emitted 只表示构图/mapping 返回并写出，后续 compile、prepare、服务分开留证。小型 shared_memory_hardware 是 vector-lane 机制例；论文 MAC16×16/64slot/1MiB 和显式外存候选以 SIM 冻结硬件为准。

正式物理候选可用 `python block.py --parameters PARAMETERS.json --hardware HARDWARE-OPTIONS.json --output NEW`。hardware JSON 原样传给公共 `components.api.block_memory_hardware`；该 factory 返回的 compute/memories 槽序和 external_memory 原样传给 block_mapping，不解析路径或重排。`HARDWARE.json` 保存完整原 metadata，含 candidate、全部实际槽、容量、MAC/SIMT/SFU、时钟、dataflow、网络声明与 hardware_accuracy。`--hardware` 与 `--adf` 互斥；factory 模式不接受手动路径覆盖，定制路径使用显式ADF模式。

Python 薄入口 `arch_explorer.archs.block_compilation.block_hardware_scene(parameters, *, hardware)` 返回(adf,spec,mapping,metadata)。hardware 为实际 factory kwargs；64tiles/1MiB/MAC16×16 的 on-chip 选项明确没有 root compute/cache 或 DRAM，external_bytes=134217728 是另一个显式128MiB外存候选。时钟、SIMT/SFU、带宽和延迟是未标定实验选择，原 metadata 原样保留；硬件声明和 scene-emitted 均不代表服务或实测精度。

SCALE-INPUTS 中逐行 parameters 属于 `mla_stream_plan`（含capacity、无数值根），不能直接当构图参数。构图 PARAMETERS.json 必须另含完整 canonical inputs、dtype/epsilon、resource_budget，并仅给正式 mla_stream 接受的关键字；不把 capacity 或缺失根静默删改/补零。原 on-chip B4/S8192 根容量超64MiB仍为 required-incomplete；external 是独立配置，不能替换原结论。四行两个chunk候选的原plan、累计逻辑字节、每槽保守峰值与实际阶段分别保存。CORE仅接线，规模窗口由SIM持固定锁推进。

LLM 结果中的 metrics 将逻辑 weight_bytes/kv_bytes/root_bytes 与 materialized_root_bytes、root_python_bytes、estimated_host_bytes 分开；性能图没有数值权重，物化根为0不能说明硬件容量满足。plan.status=mapping-required，replicas/physical_peak_bytes 为 null；真实放置、复制与容量沿既有 MappingTemplate/prepare_request 落计划。stages 保留正式构图/编译/服务边界，graph_executed/service_executed 始终为 false。compact KV 性能图仍不支持 compile_graph 的完整数值根路径，预检成功不改这一边界。

单阶段执行器为 `archexplorer scale-run PLAN.json --output-dir NEW_STAGE`，PLAN 恰含 argv/cwd/wall_seconds/memory_bytes/stage/dimensions/estimate。调用者先保存权威资源估计，随后显式执行一个阶段。stage 是 preflight/graph/compile/service/reference，600s/16GiB 为上限，CPU BLAS单线程、GPU不可见。三方共用固定 `/tmp/archexplorer-paper-scale.lock` 非阻塞锁，不受 TMPDIR/hostname 影响；忙时不启动子进程，持锁至受管进程组结束，不能与 SIM/QA 同时新增规模执行。

运行器要求 Linux `/proc` 和 child-subreaper。独立监督进程继承锁，接管孤儿后代；调用者的 subreaper 设置保持不变。主进程退出后仍有运行后代时，阶段状态为 failed，returncode 保留主进程实际退出码。正常退出、非零退出、取消、异常和超时均先清理同组后代，再释放锁。终止信号无效时升级为 SIGKILL；超时或组内资源超限立即 SIGKILL。清理暂未确认时返回 cleanup-incomplete，独立监督进程继续持锁、检查和清理，SUPERVISION.json 保存后续状态，不能视为已结束。

每进程继承 RLIMIT_AS 和 RLIMIT_CPU 硬限制。整个受管组共用 wall_seconds 截止时间，并每20ms检查成员 RSS 之和与累计 CPU；超过 memory_bytes 或 CPU 预算即终止整个组。组内检查是采样约束，共享页可能重复计入 RSS；它不提供内核 cgroup 总量配额，也不约束主动逃离该组的进程。这些限制应随结果解释，不能把采样峰值当作精确宿主峰值。

RESULT.json 包含完整 argv/环境/预算/尺寸和预检记录、阶段状态、wall/RSS/CPU、主进程退出码、输出大小与原日志。cleanup 单列确认状态、终止信号、剩余 PID 与锁是否保留；group_peak_rss_bytes/group_observed_cpu_seconds 是组内观测，max_rss_bytes/child_user_seconds/child_system_seconds 是主进程 wait4 统计，reaped_processes 保存回收记录。监督进程异常退出时，调用者保持锁并做应急组检查；无法取得的主进程退出码保留 null。容量拒绝和宿主内存/时间预算失败不同；超预算结果保持未完成，后续先检查具体瓶颈，不盲目增加资源重试。尺寸生成、runner 控制或预检成功均不能替代实际图、编译或组件执行。
