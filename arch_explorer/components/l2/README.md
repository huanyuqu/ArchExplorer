# l2

Hardware role: memory.

The authoritative API catalog is `components.api.component_catalog()`. Hardware parameters and execution implementation parameters are validated separately.

* `l2.accelsim@1`: accelsim, input `native.transaction.v1`. Native software cycles and mechanism counters; no hardware calibration claim.
  Factory: `arch_explorer.components.native_bindings:native_factory`.

Unsupported operations, input shapes, domains and native policies raise validation errors.
