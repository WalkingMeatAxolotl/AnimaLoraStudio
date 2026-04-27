"""project_jobs 的子进程入口。

每个 worker 都接受 `--job-id N`，由 supervisor 启动；写日志到
`studio_data/jobs/{id}.log`，退出码 0 = 成功。
"""
