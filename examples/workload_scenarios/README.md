# 有界性能场景与结构探索

先安装发行主 wheel，将本目录放到仓库外运行。示例调用公开数据结构、注册映射、Env 和 provider API。默认输入是 Llama2-7B 的一层、batch 1、序列长度 4，使用合成非 GPU 硬件，构造性能任务图。

生成完整配置与两个具体映射：

```sh
python scene.py emit --input llm.json --output ./llm-input
python scene.py prepare --scenes ./llm-input/SCENES.json --index 0 --output ./llm-request
archexplorer evaluate ./llm-request/REQUEST.json --output ./llm-result.json
archexplorer evaluate ./llm-result.json --output ./llm-readonly.json
archexplorer search --config ./llm-input/CONFIG.json --policy greedy --output ./llm-best.json
archexplorer agent run --config ./llm-input/AGENT-CONFIG.json --policy scripted --max-steps 2 --output ./llm-agent.json
```

第一次 evaluate 执行组件，第二次读取来源绑定的原 receipt。输出目录必须是新目录；保留 provider 子目录。搜索最多四次 attempted evaluations，两个映射共享额度。显式 SRAM 参数变化也消耗同一预算。查看结果中的实际 ledger、失败和 termination，不能仅凭返回 JSON 推断预算已全部用完。

普通 LLM 搜索 BEST 的 `archexplorer.adf-result.v1` 读取会重新评估。若要保存 winner 的只读 receipt，使用结果 `best.adf`、`best.workloads[i].workload` 和 `best.mappings[i]` 准备一个新的 events request，再保存其结果。编译图搜索的 replay / rerun 用法见下方 MLA 示例。

生成分组 SRAM 和原语记录：

```sh
python structure.py --output ./sram-structure
```

四个 private SRAM 合并为两个组内池，compute 数量和 SRAM 总量保持，网络是明确的合成配置。输出的 SCOPE 声明这里只生成结构；共享访问、广播复用和争用必须由正式运行组件支持后再执行。不得把普通 bus 视作已实现硬件组播。

三层联合探索示例复用同一 Env：

```sh
python joint_search.py --output ./joint-input
archexplorer search --config ./joint-input/CONFIG.json --policy greedy --output ./joint-best.json
```

这个独立配置使用 seed 7、三次评估和两个搜索步骤，先给出一层 Llama2-7B 的合成 GSM 结构。候选包括 DRAM bandwidth 64→128 B/ns 和原语组成的 GSM→DMC 结构变换；每次只从公开 gsm_tiled/dmc_partition 域采一个实际映射。结果记录被评估但未接受的结构、参数及完整映射，不要求结构变换一定胜出。原始 private→GSM 编辑记录在 ORIGIN.json。普通搜索格式仍按其原有重新评估合同恢复。

Python API 可使用 `llm_workload`、`llm_scene`、`llm_search_config`。生成的 AGENT-CONFIG 使用同一硬件和 workload、列表中的第一个具体映射，每 workload 一次预算，明确设置两步 evaluate / submit 脚本。离线 Agent 调用既有工具，不接在线模型服务。

更换模型、phase、batch、seq_len、layers、dtype 时修改 llm.json 中全部显式维度。Llama2-70B、Llama3-70B、Qwen-72B 的已注册配置用于构造性能任务图。每次探测先限制单进程 120 秒、8 GiB 和单线程 BLAS；超限保留失败后拆分。

## 小型 MLA 的编译搜索

MLA 场景从注册 builder 生成实际图，根值由公开 shapes 决定。下例默认 FP32、四 heads、短 compressed cache 和两个计算组。shared/private 与 multicast on/off 都是显式参数。

```sh
python mla.py --sharing grouped --multicast on --output ./mla-input
archexplorer search --config ./mla-input/CONFIG.json --policy greedy --output ./mla-best.json
archexplorer evaluate ./mla-best.json --output ./mla-readonly.json
archexplorer rerun ./mla-best.json --work-dir ./mla-rerun --output ./mla-rerun.json
archexplorer agent run --config ./mla-input/AGENT-CONFIG.json --policy scripted --max-steps 3 --output ./mla-agent.json
```

这里的 evaluate 读取新 compiled-search winner，按其 schema 只读校验。rerun 使用新的执行目录。完整源码/API 消费示例为：

```python
from arch_explorer.archs.mla import mla_scene
from arch_explorer.workload.api import compile_workload
from arch_explorer.simulator.api import prepare_compiled, evaluate_request

adf, workload, mapping = mla_scene(shared=False, multicast=False, candidate=1, layout='views')
compiled = compile_workload(adf, workload, mapping)
request = prepare_compiled(compiled, output_dir='./mla-api-run', options={'trace': True})
result = evaluate_request(request)
same = evaluate_request(result)
```

`mla_decode` 使用 absorbed projection 和完整显式 cache，执行单次 decode。v4 支持 external Events；示例使用 FP32 数据与显式 cache 初始化。
