# SIM52 typed 正式消费顺序

冻结输入 SHA256 `4465f47136b92df34847b6002d96324d9dfe18e69e47d3b9bc6cff756e17ceb1`，PUBLIC-API SHA256 `9d40652ca0de41333482201668f9cb7b3daa909a70bd89bdf33c70d8a2135398`。框架文件和完整根按原CORE34身份关联，数据资源复制不包含双方生产拼装。

先以 `typed_onnx_contract_cases.imported` 接公共 `onnx_preflight`、`normalize_budget`、`import_onnx`、`describe_onnx`，显式保存13字段完整预算。检查根原名、shape、dtype、elements、物理字节、预算和 graph.meta 的一致。旧省略预算四字段兼容仍须在原包验，不能由新五字段示例代签。budget_refusals 覆盖bool假整数、rank/dimension/host超限、未知模式/字段的产品 ValueError。

数值观测脚本 `typed_onnx_consumer.py` 必须在负责人唯一安装候选的 site-packages 环境运行。按原ModelProto摘要绑定已经封存的独立oracle，分别保存原图每节点和映射CPU输出误差，逐dtype存储网格及显式Cast有单独精确控制。所有一般节点误差均是观察结果，numeric_acceptance保留 pending，待逐节点计算/归约合同审查后由最终验收驱动收口；脚本返回0不能当成完整数值验收通过。BF16官方evaluator原失真不改变此规则。

MatMul访问按每次FFMA各输入的实际payload计费，BF16/FP16各2B，FP32为4B；同一根读取两次必须累加。同dtype Cast仍有读取和写入。公共 plan/address_map 为实际记录；独立算术控制不能冒称组件已执行。

`--component` 才执行实际 prepare/evaluate。内存观察先绑定到facts，实际结果和原始观察在audit前落盘，最外层finally保存错误和已有数据。真实返回根集合必须完整，与固定“双输出”模型名无关；旧共享根/alias/末读者审计复用其typed字节规则。views峰值不预设下降。容量just-fit/少1B只从实际已观察峰值推导，不做阈值搜索。

`reference_mode='inputs'` 只证明完整根初始化与预算编译，零数值算子必须额外用原包计数器/实际导入引用观察核对。不能用显式 execute_reference 的执行替它代签。fresh正常与有效重封 source/model/external/root/shape/budget/mapping/output 拒绝、API/CLI/search/Agent、noextra、Python3.10与sdist、新目录rerun仍沿后续唯一候选阶段完成，不在本准备脚本中冒称已验证。

MLA Fig16四行依子件正式预算输入与实际阶段产物审核；H128、完整S/B与实验投影尺寸分列。MLA budget构图账本、resource_budget通用编译守卫与实际SRAM容量是三项不同核对。当前没有因收到接口重跑规模或旧矩阵。

有效篡改先用 `saved_binding_contract.reseal_layers` 明确列出所有改变的封装层；由内到外移除旧 binding 后 seal，再对每层调用 compiler.check。摘要合法记录与后续产品语义 ValueError 分列；保留原 binding 后 seal 所产生的摘要拒绝只算摘要负控。该工具的合成控制不构成产品篡改验收。

公共规模工具必须显式传 `run(...,scale=True,scale_lock="<ABSOLUTE_SHARED_LOCK_PATH>")` 或 CLI `--scale --scale-lock <ABSOLUTE_SHARED_LOCK_PATH>`。缺参数抛稳定 ValueError、不spawn。实际共享路径由私有运行配置绑定，小型参考/拒绝不占规模窗口。
