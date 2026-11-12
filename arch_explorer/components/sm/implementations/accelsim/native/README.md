# Accel-Sim SM cluster factory

`shader.cc` owns `make_shader_cluster` and `ShaderCluster`. The cluster owns
the real trace cores, L1 caches and shared memory. L1/shared are contained
mechanisms; this package does not claim a separate L1 factory.

`shader_api.hpp` and `shader_services.hpp` retain their flat public include
names. Explicit clocks, dispatch, counters, transport and cache services still
belong to the caller. Fixed upstream mechanism extraction is performed by the
existing shader transformer; directory migration changes no transformation.

Common configuration, transactions and license notices are under
`components/_support/accelsim/native`; GPU wiring is under `assemblies/gpu`.
