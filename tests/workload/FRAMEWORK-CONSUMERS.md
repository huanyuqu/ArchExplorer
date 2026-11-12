# VAL63 唯一候选的用户消费者

准备依据：CORE34 `ENTRYPOINTS.md`，SHA256 `d71b07dce66d21a6384e04380961c3b415a107bcf2b0b9308f03db5e8ffee5c1`。这些脚本等待负责人唯一候选后实际调用公共生产入口；源码 AST 检查与运行脚本准备不构成产品通过。

`framework_export_consumer.exercise_export` 接受真实 Torch 模块、完整 `inputs`、按模块位置参数顺序的 `input_names`、`output_names` 和新的输出目录。冻结 MLP 为 `X,R`，decoder 为 `X,mask`。函数故意通过排序 JSON 重排输入对象，再显式传 `input_names` 给 API；实际 console `export-torch` 也传必填的 `--input-names` 与 `--output-names`，不依靠字典或文件顺序。检查真实 IR8/opset17/根顺序/根值/shape、双输出、调用者模块状态，以及实际 TorchScript CLI。调用者须在有界单线程120s窗口运行，导出环境与最低版本导入环境分别记录。

`tests/validation/hardware_comparison_consumer.py NEW_OUTPUT_DIR` 在仓外安装环境实际调用 `arch_explorer.archs.hardware_comparison.compare_hardware` 及 `archexplorer compare-hardware`。独立比较器给出正/负误差、单位换算、零参考、split 分组和最坏样本期望；产品 mape 是比值，独立参考 mape_pct 是百分比，显式除100核对。合法CLI返回0，含拒绝行的完整报告返回2。重复ID的所有出现均拒绝。缺项/设备/计时/映射/精度/统计量/非有限/数字化来源与重复ID逐项为负控。完整合成样本仅证明比较器，不证明真实硬件准入。

两个消费者在断言前保存已经取得的原始结果，并以 finally 保存后续失败。独立脚本不引用同伴生产源码，不给 compiler 或成本模型加入替代实现，不调用开发树拼装候选。安装阶段的数字参考、storage/address/traffic/conversion、容量、search/Agent、fresh只读/有效篡改与新目录rerun另依 SIM52 正式 typed 合同接入，不能用导出/比较控制代替。

公共规模运行器要求显式 `scale_lock` 参数或 CLI `--scale-lock <ABSOLUTE_SHARED_LOCK_PATH>`。缺参数抛稳定 ValueError、不spawn，不内置宿主默认；调用方在私有配置中绑定实际共享路径。临时锁行为控制与原数值证据按实际依赖复用。
