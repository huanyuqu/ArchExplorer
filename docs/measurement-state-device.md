# Finite state measurements on a device

The device-state campaign uses the registry v3 role plan, shared C++ executor
and shared read-only observation/lifecycle validators. The legacy Collector
still rejects state registries. The explicit device-state design binds its
plan, actual ELF catalogs, CPU readiness executable, snapshot policy, source
archive and complete launch budget.

Every case has one diagnostic, two preparation and three formal executions.
Preparation is a finite condition; the diagnostic is the first registered
execution of that host/context. Neither establishes cold hardware or steady
state. All outputs, API returns, float32 event bits and failed/not-started roles
are retained. First failure stops every later case and role.

For new idle-device runs the final installed controller freezes a local request
binding the run, design, registry, plan, full role budget, GPU UUID and absolute
deadline. It takes the shared nonblocking lock and checks actual occupancy before
issuing a local admission and host lease. An idle GPU needs no owner approval,
reservation original or handoff reply. Busy or unknown occupancy stops the entry
before any application host; resolve a resource conflict through coordination.
See [GPU resource admission](gpu-resource-admission.md) for the public entry.

Historical resource handoff v3 and its owner/coordinator evidence remain readable
for closed experiments. A separate marked synthetic handoff is accepted only by
explicitly injected CPU capabilities. The actual entry rejects synthetic sources
and roots. Old handoffs and closed windows cannot authorize new roles. Locks,
queries and production hosts are not opened by import, design validation,
building or replay.

At most eight programs and nine measurement hosts are permitted. One capture
per case and six uninstrumented executions per case give at most 56 application
launches. The whole window includes CPU native work and bounded queries, with a
fixed final sixty seconds reserved for cleanup. Each host also has a 120 second
limit. CPU native subprocesses are listed independently from measurement hosts.

All structures and initial public native predictions precede any uninstrumented
observation. Only fit observations precede the final freeze. Every heldout role,
including diagnostic and preparation, follows that final freeze. Fit-only does
not adjust physical parameters, program constants, role counts or thresholds.

Bounded before/after host state snapshots are outside event intervals. Missing
values remain unknown. Instantaneous clocks cannot replace assumed clocks,
certify a mean clock or prove cache/steady-state equivalence. Unless equivalence
is established independently, physical MAPE and worst error remain null.

CPU rehearsals use actual C++ CPU Driver processes and the same installed stage
machine, with explicit synthetic device/resource capabilities. They demonstrate
software boundaries only. Final hardware execution uses the frozen entry and a
new local idle admission, with coordination needed only for occupancy/conflicts;
no retry, warmup or replacement is implied.
