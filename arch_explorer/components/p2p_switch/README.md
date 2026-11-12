# p2p_switch

Hardware role: network.

The authoritative API catalog is `components.api.component_catalog()`. Hardware parameters and execution implementation parameters are validated separately.

* `builtin.p2p_switch@1`: events, input `events.component.v1`. Existing event model; original declared limitations unchanged.
  Factory: `arch_explorer.components.p2p_switch.definition:P2pSwitch.build`.

Unsupported operations, input shapes, domains and native policies raise validation errors.
