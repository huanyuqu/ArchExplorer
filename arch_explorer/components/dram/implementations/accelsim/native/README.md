# Accel-Sim DRAM factory

`dram.cc` owns `DramComponent`, `make_dram`, native tick selection, admission,
DRAM counters and physical drain queries. Its original method bodies are
separated from L2. DRAM borrows the caller's explicit L2 set and never creates
or destroys L2 instances.

The existing memory transformer retains the fixed upstream controller and
FR-FCFS algorithms. Public `memory_api.hpp`, common configuration validation,
statistics, services and license notices live under
`components/_support/accelsim/native`. No second algorithm copy is retained.
