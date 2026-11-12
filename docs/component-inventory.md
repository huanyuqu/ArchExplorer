# Component inventory

The executable [component catalog](component-implementations.md) describes the
installed version. A hardware type, an execution implementation and an operator
lowering are different identities.

This inventory distinguishes **runnable**, **limited implementation**,
**not implemented** and **pending evidence**. A limited implementation is usable
only inside its stated scope. The listed tests are reproducible acceptance
entry points, not a claim that they were all rerun in this iteration. Current
source is the reference; the initial integration report predates several of
these implementations and is not a current missing-feature list.

## Hardware responsibilities

| Responsibility | Status and actual scope | Authoritative code directory | Acceptance entry / detail |
| --- | --- | --- | --- |
| SM compute | Limited implementation: explicit roofline and Accel-Sim share hardware type `sm`; source and installed evidence are tracked separately. Native granularity is a cluster containing L1/shared, not an independently factored L1. | [sm](../arch_explorer/components/sm/) | [Selection and input contracts](component-implementations.md); [GPU assembly](gpu-instantiation.md) |
| Systolic array | Limited implementation: closed-form SCALE-Sim-v3 dataflow timing; explicit stationary buffering. Not an arbitrary systolic RTL simulator. | [systolic_array](../arch_explorer/components/systolic_array/) | [Component library](components.md#systolic-arrays-and-scale-sim), `tests/sim/test_components.py` |
| Vector unit | Limited implementation: MAC/vector/SFU resource demand with declared precision and issue rates. Not a general instruction pipeline. | [vector_unit](../arch_explorer/components/vector_unit/) | [Component library](components.md), `tests/sim/test_components.py` |
| SRAM / ordinary DRAM | Runnable demand ports, bandwidth/latency and capacity contracts. Native detailed L2/DRAM are separate implementations with their own input and execution domains. | [sram](../arch_explorer/components/sram/), [dram](../arch_explorer/components/dram/) | [Component library](components.md), [standalone native memory](standalone-components.md) |
| L2 / DRAM native mechanisms | Runnable standalone factories and GPU assembly; request/response identity, queue pressure, writeback, physical drain and multiple instances. Convenience memory assembly has its documented fixed topology. | [l2](../arch_explorer/components/l2/), [dram](../arch_explorer/components/dram/) | [Standalone consumer](../examples/gpu_instantiation/README.md), `tests/accelsim/test_standalone_wire_views.py` |
| ICNT and ordinary interconnects | Limited implementations: native ICNT and analytic bus, mesh, crossbar, FIFO and point-to-point switch have distinct protocols. An echo endpoint demonstrates transport, not memory behavior. | [icnt](../arch_explorer/components/icnt/), [mesh2d](../arch_explorer/components/mesh2d/), [bus](../arch_explorer/components/bus/), [crossbar](../arch_explorer/components/crossbar/), [fifo_link](../arch_explorer/components/fifo_link/), [p2p_switch](../arch_explorer/components/p2p_switch/) | [Standalone consumer](../examples/gpu_instantiation/README.md), [component library](components.md) |
| PIM unit / shared memory controller | Limited implementation: bank-level GEMV and explicit transaction/controller command paths, with profile-specific timing, precision and addressing. Not arbitrary in-memory compute or calibrated silicon timing. | [pim_unit](../arch_explorer/components/pim_unit/), [controllers](../arch_explorer/components/controllers/) | [Component library](components.md), [shared transaction subsystem](hsndi-detail.md), `tests/sim/test_pim.py`, `tests/sim/test_reusable_detail_components.py` |
| Scalar CGRA PE | Limited implementation: supported 4x4 uint32 DFG timing and memory/routing choices. | [cgra_pe](../arch_explorer/components/cgra_pe/) | [CGRA scope](cgra.md), `tests/sim/test_cgra.py` |
| IPU tile / Tianjic core / tile exchange | Limited implementation: tile-private memory, declared matrix/vector service and explicit BSP/spatial dependencies. Poplar, full spike processing and unmodeled mesh timing are not implemented. | [ipu_tile](../arch_explorer/components/ipu_tile/), [tianjic_core](../arch_explorer/components/tianjic_core/), [tile_exchange](../arch_explorer/components/tile_exchange/) | [Many-core scope](manycore.md), `tests/sim/test_manycore.py` |
| Service/transaction controllers | Runnable within each declared state protocol, ownership and lifecycle. Queue or refresh behavior is not interchangeable with aggregate bandwidth demand. | [controllers](../arch_explorer/components/controllers/), [service_ceiling](../arch_explorer/components/service_ceiling/), [serial_service](../arch_explorer/components/serial_service/) | [Unified execution](unified-execution.md), [shared controller](hsndi-detail.md) |

## Framework mechanisms and outstanding boundaries

| Mechanism | Current status and acceptance entry |
| --- | --- |
| Six structure primitives | Runnable split/merge/insert/remove/create/delete, atomic validation, repeat handling and replay. [Structure reference](structure.md), `tests/core/test_structure.py`. The initial integration report's missing-primitives statement is historical. |
| Reusable structure transforms | Runnable chiplet, shared-to-distributed memory, PIM and bonded-memory programs in the documented scope; records reduce to basic edits. `tests/core/test_structure.py`. |
| Structure / parameter / mapping search | Runnable bounded nested search and explicit user proposals. [Search](structure.md), [implementation](../arch_explorer/exploration.py), `tests/core/test_structure_search_neutral.py`. |
| Task DAG, mapping, capacity and resource contention | Runnable within each mapping/implementation contract. Operator execution implementations and temporary tensors remain distinct from hardware implementations. [Open operator execution](operator-implementations.md), `tests/workload/test_target_implementations.py`. |
| Contention-stash scheduling | Implemented by `ConsistentScheduler`: jointly evaluate a contention set, split unfinished work, stash tentative results, commit against future competing tasks' earliest-start bounds, and roll back uncommitted results when a late competitor arrives. The conservative stateful driver uses its own event protocol. [Unified execution](unified-execution.md#consistent-and-conservative-scheduling), [scheduler](../arch_explorer/simulator/events/engine.py). |
| General cross-domain bridges | Not implemented unless the selected implementations explicitly supply a bridge. Matching hardware identity does not make traces and operation counts interchangeable. |
