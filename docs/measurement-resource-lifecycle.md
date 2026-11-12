# Measurement transcript resource lifecycle

The CPU and CUDA adapters use the same C++ measurement executor. Host transcript
`arch-explorer.measurement-state-host.v2` adds observed API operands and outputs
to each ordered trace entry. Each entry carries its run, case and context, and
each allocation or event carries its category, name and actual numeric handle.
Allocations also retain requested sizes; transfers retain the host buffer SHA256.
Failed create calls retain a nonzero output handle when one was returned.

The installed public `microclosure` reader derives ownership from those create
outputs and checks every upload, reset, launch, event operation and cleanup
against live objects in the same scope. Input sizes and bytes come from the
selected complete program registry. Chase programs also require their auxiliary
allocation and upload. Each sample reuses the created event pair with a new
record/synchronize generation. A file reference or reported remaining list
cannot supply a missing Driver call.

Normal execution stops on its first failure. Cleanup still attempts the required
context synchronization, then acquired events and allocations in reverse order.
Only successful destruction/free removes an object. A cleanup synchronization
error remains a cleanup failure even if later releases succeed. An early failure
that acquired no objects can finish cleanup successfully while the experiment
remains failed. Error returns with nonzero handles acquire objects that must be
cleaned up. The reader derives remaining handles and cleanup status before
comparing those values with host, process and window claims.

Earlier SIM33 host v1 transcripts lack resource observations and cannot certify
this contract. They are rejected as insufficient evidence, without supplying
invented metadata. The separate SIM32 replay path and its original failed
outcome retain their existing meaning. Raw per-slot schema v1 remains unchanged;
its case/context, pair, codes, bytes and references must agree with the v2 host.

CPU Driver injections and mutation replays demonstrate software semantics only.
They provide no GPU resource authority, hardware cleanup guarantee, stability or
physical accuracy result. Quality remains three formal samples with the frozen
3% range/mean threshold; unknown physical parameters remain null.
