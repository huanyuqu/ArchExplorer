# pim_unit

Hardware role: compute.

The authoritative API catalog is `components.api.component_catalog()`. Hardware parameters and execution implementation parameters are validated separately.

* `builtin.pim_unit@1`: events, input `events.component.v1`. Existing event model; original declared limitations unchanged.
  Factory: `arch_explorer.components.pim_unit.implementations.bank_timing:PimUnit.build`.

Unsupported operations, input shapes, domains and native policies raise validation errors.
