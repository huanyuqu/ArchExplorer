# GPU resource admission

An idle shared GPU can be used directly. The executor takes the common lock,
checks the selected device's actual occupancy, and starts only when the check
establishes idle status. Owner approval, a per-run reservation, an owner reply
and a historical release record are not prerequisites for idle use. Occupancy
or resource conflicts require coordination. Existing processes are never stopped
or preempted by this admission path.

The common lock and a fresh local evidence record prevent concurrent starts and
bind each run to its actual observations. They are execution records, not a
replacement approval workflow. An experimental task still freezes its programs,
source installation, role counts, byte budgets and deadlines before execution.
This admission rule does not reopen a closed run or permit extra measurements.

## Installed entry

Use the source-bound installed interpreter and an explicit device-state design
whose CPU preparation is complete. The lock file is an existing shared file,
the same one used by other users of that GPU. Supply its absolute path and the
expected physical GPU0 UUID. A fresh request starts the fixed window clock;
delayed execution cannot move its deadline.

```sh
python -m arch_explorer.components.accelsim.state_idle plan \
  DESIGN.json NEW-REQUEST.json --lock-file /absolute/shared/GPU.lock \
  --gpu-uuid GPU-00000000-0000-0000-0000-000000000000
python -m arch_explorer.components.accelsim.state_idle run \
  DESIGN.json NEW-REQUEST.json NEW-WINDOW NEW-SUPERVISOR
```

`plan` is CPU-only. It creates an immutable request, with no resource authority,
owner/coordinator fields or claim of an idle GPU. `run` invokes the existing
absolute supervisor and shared campaign. The worker then:

1. Takes the common lock with nonblocking exclusive `flock`.
2. Creates a persistent one-attempt claim under
   `<lock-directory>/.measurement-runs/<run_id>/CLAIM.json`. A consumed run cannot
   be retried through a replacement output directory.
3. Runs one bounded `nvidia-smi --id=0 -q -x` query while holding the lock. Its XML
   must identify the expected UUID, contain an available empty process list
   covering compute and graphics processes, and report zero GPU utilization.
   Memory usage is retained as a raw observation; driver bookkeeping bytes do
   not themselves constitute a process. Missing, invalid, timed-out or foreign
   device output stays unknown and does not authorize a host.
   UUID, process list, utilization container and GPU utilization must each occur
   exactly once at the expected location. Repeated values, including identical
   copies, misplaced fields and nested content in critical scalar values stay
   unknown. Unrelated normal management fields remain permitted. The acquisition
   path and public reader use the same parser for these checks.
   Across the entire selected GPU subtree, every `process_info` must be a direct
   child of that process list, and every `pid` must belong directly to one of
   those process records. A process or PID under another field is unknown;
   an empty list cannot hide an unowned process observation elsewhere.
4. Saves stdout/stderr, return code, realtime/monotonic bounds and lock
   path/device/inode/fd. Only an idle result produces `IDLE-ADMISSION.json` and
   the existing host lease, retaining the same lock without a release gap.
5. Runs the existing live preflight, complete structures, initial native freeze,
   fit roles, final freeze and heldout roles. The shared C++ executor, role
   ordering, quality checks and physical unknowns remain the measurement rules.

The request and admission have separate schemas. A request cannot be passed off
as a completed admission; the public reader reconstructs idle status from the
hash-bound XML and binds it to the same lock and subsequent host times. New
idle evidence uses no fabricated owner or coordinator. Historical v3 handoffs
continue through their original validator for old records.

## Occupancy, failure and closure

An occupied common lock causes no device query. A query showing a process,
nonzero utilization or unavailable occupancy causes no application host, lease
or retry. Save the failed prefix, release the acquired lock and coordinate the
conflict with whoever is using the resource. Do not kill a foreign process or
poll repeatedly until a trial passes.

The existing finite budget remains at most eight cases, nine measurement hosts,
56 application launches and 1800 seconds including admission/query and CPU work.
The last sixty seconds remain reserved for reaping and release; each host is
also bounded by 120 seconds and the remaining work cutoff. No automatic warmup,
measurement replacement, quality relaxation or native parameter fitting is added.

After execution, reap owned hosts and descendants and release the shared lock.
Retain the local process, resource and supervisor close records. Normal idle use
does not require an owner's close acknowledgement. If an actual conflict was
coordinated with another user, communicate the resulting release or residual
processes to that party. Unknown cleanup stays incomplete.

The outer supervisor retains its final lock recheck even when it reports busy:
another user may already hold the lock, or acquire it after our release. The
reader verifies our own reaping and release independently and never requires a
foreign holder to stop merely to produce a globally idle closing snapshot.

Closed historical runs, reservations and failed outcomes remain immutable. New
resource availability never changes a past outcome or supplies missing samples.
