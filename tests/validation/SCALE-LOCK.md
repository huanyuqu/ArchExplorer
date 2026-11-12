# Shared Lock for Bounded-Scale Execution

Scale graph construction, compilation, and service running in parallel collaboration on the same host must specify the same advisory flock file by the caller. Public call example:

`run(command, cwd=cwd, dest=dest, scale=True, scale_lock="<ABSOLUTE_SHARED_LOCK_PATH>")`

Public tool calls must explicitly provide `scale_lock`; the CLI uses `--scale --scale-lock <ABSOLUTE_SHARED_LOCK_PATH>`. The caller replaces the placeholder with the actual absolute path and binds it uniformly in the private run configuration; the tool does not hardcode host paths or TMPDIR/hostname defaults. Missing parameter raises `ValueError('scale=True requires an explicit absolute scale_lock path')`, saves `status=rejected`, `reason=missing-scale-lock`, `returncode=null`, and does not call Popen; relative or empty paths raise the same error. CLI argument errors return 2.

The runner acquires the specified lock with `LOCK_EX | LOCK_NB` before creating child processes. When the lock is busy, save `status=busy`, `reason=scale-lock-held`, `returncode=null`, do not call Popen, and do not start waiting scale tasks. The lock holder is responsible for running and reaping child processes; normal exit, failure, cancellation, or timeout all clean up descendants in the same group. After cleanup, scan `/proc` to confirm no group members are still running and reap direct child processes, then close the lock. The lock file must not be deleted or replaced to circumvent contention.

`returncode` is the exit code of the direct child process, `parent_reaped` is its reaping status; `process_group_closed` also requires that actual group observation has confirmed no running members. `process_group_cleanup` separately lists signal sending, members before/after cleanup, and their Linux states. Z/X are terminal states; retained descendant zombies must not be claimed as reaped. Merely sending kill does not constitute cleanup completion. Cleanup observation waits at most 2s; when unconfirmed, save `cleanup-unconfirmed`, raise an error, and keep the lock and child process handles in the runner's `_unconfirmed_groups` for explicit recovery by the caller; a surviving caller must not automatically open scale slots. This tool only checks the same process group; processes started in another session require additional management.

Small 120s reference and rejection checks do not use `--scale` and do not occupy the scale window. 600s/16GiB is the scale upper bound; calls not marked as scale must not bypass the window or budget. The caller must not rewrite the already agreed lock path.

Self-checks use an independent temporary lock and a lightweight Python control process to verify no spawn when busy, the lock is held while the child process is alive, and the lock is released after timeout reaping; these checks do not occupy the real scale lock and do not constitute model-scale execution.
