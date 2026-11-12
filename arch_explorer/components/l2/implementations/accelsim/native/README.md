# Accel-Sim L2 factory

`l2.cc` owns `L2Component`, `make_l2`, sector admission, response scheduling,
L2 metrics and physical drain queries. Its method bodies are moved from the
former mixed `memory.cc`. The real cache algorithms remain the pinned native
inputs transformed by the existing memory generator.

Public `memory_api.hpp` and the shared memory services/statistics remain under
`components/_support/accelsim/native`. They expose the original flat headers.
DRAM is explicitly connected by the caller and owns its own factory directory.
Shared license notices are in the support directory's `NOTICE`.
