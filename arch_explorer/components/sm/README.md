# sm

Hardware role: compute.

The authoritative API catalog is `components.api.component_catalog()`. Hardware parameters and execution implementation parameters are validated separately.

* `sm.accelsim@1`: accelsim, input `native.instruction-trace.v1`. Native software cycles and mechanism counters; no hardware calibration claim.
  Factory: `arch_explorer.components.native_bindings:native_factory`.
  Contains l1, shared_memory; cluster granularity, not independently created L1/shared models.
* `sm.accelsim-sm120@1`: accelsim, input `native.instruction-trace.v1`. Native software cycles and mechanism counters; no hardware calibration claim.
  Factory: `arch_explorer.components.native_bindings:native_factory`.
  Contains l1, shared_memory; cluster granularity, not independently created L1/shared models.
* `sm.roofline@1`: events, input `roofline.work.v1`. Analytical estimate under ideal overlap; not native cycles or calibrated timing.
  Factory: `arch_explorer.components.sm.implementations.roofline:factory`.

Unsupported operations, input shapes, domains and native policies raise validation errors.
