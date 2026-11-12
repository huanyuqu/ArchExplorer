# 有界规模执行的共享锁

同一宿主上并行协作的规模构图、编译和服务必须由调用方指定同一个 advisory flock 文件。公共调用示例：

`run(command, cwd=cwd, dest=dest, scale=True, scale_lock="<ABSOLUTE_SHARED_LOCK_PATH>")`

公共工具调用必须显式给 `scale_lock`，CLI 则用 `--scale --scale-lock <ABSOLUTE_SHARED_LOCK_PATH>`。调用方用实际绝对路径替换占位符，并在私有运行配置中统一绑定；工具不内置宿主路径或 TMPDIR/hostname 默认值。缺参数抛 `ValueError('scale=True requires an explicit absolute scale_lock path')`，保存 `status=rejected`、`reason=missing-scale-lock`、`returncode=null`，不调用 Popen；相对或空路径抛相同错误。CLI 参数错误返回2。

运行器在创建子进程前以 `LOCK_EX | LOCK_NB` 获取指定锁。锁忙时保存 `status=busy`、`reason=scale-lock-held`、`returncode=null`，不调用 Popen，不启动等待中的规模任务。持锁者负责运行与收割子进程；正常退出、失败、取消或超时均对同组后代清理。清理后扫描 `/proc`，确认没有仍运行的同组成员并收割直接子进程，再关闭锁。不得删除或替换锁文件来规避竞争。

`returncode` 是直接子进程的退出码，`parent_reaped` 是其收割状态；`process_group_closed` 还要求实际组观测已确认没有运行成员。`process_group_cleanup` 分列信号发送、清理前后成员及其 Linux 状态。Z/X 是终止态；保留的后代僵尸不声称已收割。单纯发送 kill 不构成清理完成。清理观测最多等待2s；未确认时保存 `cleanup-unconfirmed`、抛错误并将锁与子进程句柄保留在运行器 `_unconfirmed_groups`，供调用方显式恢复，存活调用方不能自动开放规模槽。此工具只检查同一进程组，另起 session 的进程需要额外管理。

小型120s参考与拒绝检查不用 `--scale`，不占规模窗口。600s/16GiB为规模上限，不能通过未标 scale 的调用绕过窗口或预算。调用方不得自行改写已约定的锁路径。

自检使用独立临时锁与轻量 Python 控制进程，核对忙时无 spawn、子进程存活期间锁被持有，以及超时收割后解锁；这些检查不占真实规模锁，也不构成模型规模执行。
