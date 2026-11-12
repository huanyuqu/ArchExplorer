# bus

Hardware role: network.

The authoritative API catalog is `components.api.component_catalog()`. Hardware parameters and execution implementation parameters are validated separately.

* `builtin.bus@1`: events, input `events.component.v1`. Existing event model; original declared limitations unchanged.
  Factory: `arch_explorer.components.bus.definition:Bus.build`.

Unsupported operations, input shapes, domains and native policies raise validation errors.
