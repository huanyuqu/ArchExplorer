# Explicit multicast tree

`multicast_tree` is a synthetic balanced binary network, with declared
`link_bandwidth` and `port_bandwidth` in B/ns and `hop_latency` in ns.
Endpoint order is the direct ADF element order after repeat expansion. If N
endpoints are present, L is the next power of two: endpoint e is leaf L+e,
the gateway is root 1, and internal node i has children 2i and 2i+1. Unused
leaves carry no padded data. Ports are distinct even for the N=1 zero-hop case.

Directed `edgeU_to_V` resources and source `outE` / target `inE` ports model
contention. `route_many` charges the union of actual directed edges, one source
port, and every target port. Multiple unicast calls charge each path in full.
Each transfer reserves its paths as a fluid cut-through flow with a longest
hop latency tail. Packet buffering/store-forward and hardware calibration are
outside this model. No paper traffic ratio or speedup is a model parameter.

The generic compile_graph_v4 storage contract supplies actual root locations
and explicit immutable replicas. Actual copies, remote reads, merge outputs,
allocation capacity, aliases and last readers remain in the public event DAG.
