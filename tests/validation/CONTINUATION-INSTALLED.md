# VAL64 唯一候选的追加消费者

这份工具准备不导入新生产源码；`continuation_installed_consumer.py` 只在负责人唯一 wheel/sdist 安装环境运行。调用方先按唯一 BUNDLE 核对安装来源与冻结输入闭包，再从仓外 cwd 运行，每个 stage 用 `stage_capture.py` 限120s/8GiB/单线程。`--candidate-commit` 记录调用方提供的身份，消费者记录实际 site-packages 文件摘要；此字符串本身不认证安装包。

```sh
python QA_TESTS/continuation_installed_consumer.py \
  --package FROZEN_CORE35_PACKAGE --case FROZEN_CASE \
  --reference-record EXISTING_QA_REFERENCE/RESULT.json \
  --candidate-commit SOLE_CANDIDATE_COMMIT \
  --candidate 0 --layout views --stage numeric --output NEW_DIRECTORY
```

`numeric` 做真实公共 import_onnx、describe、compile、原图和映射图 CPU reference。按原值名与声明 dtype/逻辑 shape 逐节点核对，int32/int64/bool 精确比较，拒绝浮点容器中的索引。每个显式 Cast 用独立转换检查，包括零符号；FP32 记录预设5e-5误差门槛，低精度漂移与存储网格分别记录，最终通过判定须结合 SIM53 正式精度合同。原始输入、初始化权重及 Constant 字节从真实 ModelProto 类型计算，不能把全部根误算成模型浮点 dtype。BF16 模型里的 FP32 Conv 权重按原类型保留。

`component` 保存真实编译/prepare/求值结果与调用计数，实际 Events/物理分片审计仍须正式观察适配。该 stage 不借服务延迟宣称 CPU 数值通过。`search` 和 `agent` 各使用一组显式 placement/layout，max_evaluations最多3，保存配置、逐尝试账本和赢家。独立核对消费单位、失败单位、实际执行标记、原索引、最优目标与赢家 workload/mapping 对 bound source 的身份；不会用结束后新实验补造超时运行的最终计数。

`continuation_observations.bind_reference` 复用原参考与冻结包清单的来源摘要，不重算整个节点图。已有 FP16视觉包装失败继续记录 raw_reference_status=failed 与 reference_requires_review，保存的数据可作观测参考，但最终验收需关联先前明确的补充分类，不能将原失败文件改成通过。

源码阶段仅测试独立比较与账本鉴别器，并检查已封存六份参考/输入元数据的接线。它没有模拟产品返回值执行消费者，不表示新算子、组件、搜索或安装已经通过。

CORE35 正式接口 ENTRYPOINTS 还规定真实 CLI 的 `export-torch --input-names`、多输出、`emit --candidate/--layout` 和保存/replay/rerun；该部分协议已有作者原件，QA 后续仅在唯一候选做真实端到端消费。既有 `onnx_readonly_probe.py` 与 `saved_binding_contract.py` 保留为 fresh/有效重封工具；新生产边界到达后核对拦截覆盖，纯编译/重建与 prepare/engine/solver/写入分列。
