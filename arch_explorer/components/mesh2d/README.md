# mesh2d

Hardware role: network.

The authoritative API catalog is `components.api.component_catalog()`. Hardware parameters and execution implementation parameters are validated separately.

* `builtin.mesh2d@1`: events, input `events.component.v1`. Existing event model; original declared limitations unchanged.
  Factory: `arch_explorer.components.mesh2d.definition:Mesh2d.build`.

Unsupported operations, input shapes, domains and native policies raise validation errors.
