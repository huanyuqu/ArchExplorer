# cgra_pe

Hardware role: compute.

The authoritative API catalog is `components.api.component_catalog()`. Hardware parameters and execution implementation parameters are validated separately.

* `builtin.cgra_pe@1`: events, input `events.component.v1`. Existing event model; original declared limitations unchanged.
  Factory: `arch_explorer.components.cgra_pe.definition:CgraPe.build`.

Unsupported operations, input shapes, domains and native policies raise validation errors.
