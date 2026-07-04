"""Supervisor 端到端测试：用一个快速 sleep/exit 假 worker 替代 anima_train.py。

通过 cmd_builder 注入子进程命令，避免依赖真实训练栈。
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any

import pytest

from studio import db, secrets
from studio.supervisor import Supervisor


def _wait_for(predicate, timeout=5.0, interval=0.05):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return False


@pytest.fixture
def env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """初始化 db + 目录，并提供一个有效 config 文件。

    task-scoped 档案（tasks/<id>/run.log + monitor/state.json + samples/）
    会落到 STUDIO_DATA/tasks/，测试里 monkeypatch TASKS_DIR 到 tmp 隔离。
    """
    from studio.infrastructure import paths as _paths
    db_path = tmp_path / "studio.db"
    db.init_db(db_path)
    logs = tmp_path / "logs"
    configs = tmp_path / "configs"
    tasks = tmp_path / "tasks"
    logs.mkdir()
    configs.mkdir()
    monkeypatch.setattr(_paths, "TASKS_DIR", tasks)
    monkeypatch.setattr(_paths, "LOGS_DIR", logs)
    (configs / "fake.yaml").write_text("epochs: 1\n", encoding="utf-8")
    return {"db": db_path, "logs": logs, "configs": configs, "tasks": tasks}


def _events_collector():
    events: list[dict[str, Any]] = []
    def on_event(evt: dict[str, Any]) -> None:
        events.append(evt)
    return events, on_event


def test_pending_task_runs_to_completion(env) -> None:
    events, on_event = _events_collector()

    def fast_cmd(task, cfg):
        return [sys.executable, "-c", "import sys; sys.exit(0)"]

    sup = Supervisor(
        on_event=on_event, cmd_builder=fast_cmd,
        db_path=env["db"], logs_dir=env["logs"], configs_dir=env["configs"],
        poll_interval=0.05,
    )
    with db.connection_for(env["db"]) as conn:
        tid = db.create_task(conn, name="t", config_name="fake")

    sup.start()
    try:
        assert _wait_for(
            lambda: _task_status(env["db"], tid) == "done", timeout=10
        ), f"timeout waiting for done; status={_task_status(env['db'], tid)}"
    finally:
        sup.stop()

    statuses = [e["status"] for e in events if e["task_id"] == tid]
    assert "running" in statuses
    assert "done" in statuses


def test_default_cmd_builder_routes_by_task_type() -> None:
    """_default_cmd_builder 按 task_type 选择脚本（PR-9 commit 3）。"""
    from studio.paths import REPO_ROOT
    from studio.supervisor import _default_cmd_builder

    cfg = Path("/tmp/fake.json")

    cmd_train = _default_cmd_builder({"task_type": "train"}, cfg)
    assert str(REPO_ROOT / "runtime" / "anima_train.py") in cmd_train

    cmd_reg = _default_cmd_builder({"task_type": "reg_ai"}, cfg)
    assert str(REPO_ROOT / "runtime" / "anima_reg_ai.py") in cmd_reg

    cmd_gen = _default_cmd_builder({"task_type": "generate"}, cfg)
    assert str(REPO_ROOT / "runtime" / "anima_generate.py") in cmd_gen

    # 缺字段 / None / 未知 → 默认 train（兼容老 task）
    cmd_legacy = _default_cmd_builder({}, cfg)
    assert str(REPO_ROOT / "runtime" / "anima_train.py") in cmd_legacy
    cmd_none = _default_cmd_builder({"task_type": None}, cfg)
    assert str(REPO_ROOT / "runtime" / "anima_train.py") in cmd_none


def test_failed_task_marked_failed(env) -> None:
    events, on_event = _events_collector()

    def fail_cmd(task, cfg):
        return [sys.executable, "-c", "import sys; sys.exit(1)"]

    sup = Supervisor(
        on_event=on_event, cmd_builder=fail_cmd,
        db_path=env["db"], logs_dir=env["logs"], configs_dir=env["configs"],
        poll_interval=0.05,
    )
    with db.connection_for(env["db"]) as conn:
        tid = db.create_task(conn, name="t", config_name="fake")

    sup.start()
    try:
        assert _wait_for(
            lambda: _task_status(env["db"], tid) == "failed", timeout=10
        )
    finally:
        sup.stop()

    with db.connection_for(env["db"]) as conn:
        task = db.get_task(conn, tid)
    assert task["exit_code"] == 1
    assert "exit code 1" in (task["error_msg"] or "")


def test_missing_config_marks_failed(env) -> None:
    """config 文件不存在时，supervisor 应立即把任务标 failed。"""
    events, on_event = _events_collector()

    sup = Supervisor(
        on_event=on_event,
        cmd_builder=lambda *_: [sys.executable, "-c", "pass"],
        db_path=env["db"], logs_dir=env["logs"], configs_dir=env["configs"],
        poll_interval=0.05,
    )
    with db.connection_for(env["db"]) as conn:
        tid = db.create_task(conn, name="t", config_name="does_not_exist")

    sup.start()
    try:
        assert _wait_for(
            lambda: _task_status(env["db"], tid) == "failed", timeout=5
        )
    finally:
        sup.stop()

    with db.connection_for(env["db"]) as conn:
        task = db.get_task(conn, tid)
    assert "preset not found" in (task["error_msg"] or "")


def test_serial_execution(env) -> None:
    """两个任务排队，应先后串行执行。"""
    events, on_event = _events_collector()

    def slow_cmd(task, cfg):
        return [sys.executable, "-c", "import time; time.sleep(0.4)"]

    sup = Supervisor(
        on_event=on_event, cmd_builder=slow_cmd,
        db_path=env["db"], logs_dir=env["logs"], configs_dir=env["configs"],
        poll_interval=0.05,
    )
    with db.connection_for(env["db"]) as conn:
        a = db.create_task(conn, name="a", config_name="fake")
        b = db.create_task(conn, name="b", config_name="fake")

    sup.start()
    try:
        assert _wait_for(
            lambda: _task_status(env["db"], a) == "done"
                  and _task_status(env["db"], b) == "done",
            timeout=15,
        )
    finally:
        sup.stop()

    # a 的 finished_at 应早于 b 的 started_at
    with db.connection_for(env["db"]) as conn:
        ta = db.get_task(conn, a)
        tb = db.get_task(conn, b)
    assert ta["finished_at"] <= tb["started_at"] + 0.05


def test_cancel_pending(env) -> None:
    """pending 任务取消：直接标 canceled，不启动子进程。"""
    sup = Supervisor(
        cmd_builder=lambda *_: [sys.executable, "-c", "pass"],
        db_path=env["db"], logs_dir=env["logs"], configs_dir=env["configs"],
        poll_interval=10,  # 防止它真的拉起
    )
    with db.connection_for(env["db"]) as conn:
        tid = db.create_task(conn, name="t", config_name="fake")
    assert sup.cancel(tid) is True
    assert _task_status(env["db"], tid) == "canceled"


def test_cancel_running_returns_immediately(env) -> None:
    """cancel running task 必须**立刻返回**，不能阻塞 grace 期 (30s)。

    回归 Q6：原 `_terminate_current` 同步等 wait(grace)，让 web 请求挂 30s。
    现在通过 `_signal_terminate_async` 起后台 grace timer，cancel() 立即返。
    """
    events, on_event = _events_collector()

    # 长时间 sleep 的子进程，模拟训练
    sleep_cmd = lambda *_: [sys.executable, "-c", "import time; time.sleep(30)"]

    sup = Supervisor(
        on_event=on_event,
        cmd_builder=sleep_cmd,
        db_path=env["db"], logs_dir=env["logs"], configs_dir=env["configs"],
        poll_interval=0.05,
        # grace=3s 就足够验证「cancel 立即返回」的核心断言（仍 >>1s）；
        # Windows 下 Python sleep 不响应 CTRL_BREAK_EVENT，得等 grace 后
        # taskkill /T /F 才退，timeout 给 grace + 充足缓冲。
        terminate_grace=3.0,
    )
    with db.connection_for(env["db"]) as conn:
        tid = db.create_task(conn, name="t", config_name="fake")
    sup.start()
    try:
        # 等子进程跑起来
        assert _wait_for(
            lambda: _task_status(env["db"], tid) == "running", timeout=5
        )
        t0 = time.time()
        assert sup.cancel(tid) is True
        elapsed = time.time() - t0
        # cancel 必须远小于 grace（3s）：Windows 上 _send_terminate_signal 同步走
        # `taskkill /T /F` 子进程（~0.5-1.5s），不走 grace timer，所以阈值 2s
        # 留出 buffer 仍然能验证「不阻塞 grace 期」的核心断言。
        assert elapsed < 2.0, f"cancel blocked for {elapsed:.1f}s"
        # supervisor 主循环 poll 到进程退出后会把 status 改成 canceled。
        # grace 3s + 主循环 poll 0.05s + 容错 buffer
        assert _wait_for(
            lambda: _task_status(env["db"], tid) == "canceled", timeout=10
        )
    finally:
        sup.stop()


def test_orphan_running_marked_failed_on_start(env) -> None:
    """启动时清理 status='running' 但 pid 已死的任务。"""
    with db.connection_for(env["db"]) as conn:
        tid = db.create_task(conn, name="t", config_name="fake")
        db.update_task(conn, tid, status="running", pid=999999)

    sup = Supervisor(
        cmd_builder=lambda *_: [sys.executable, "-c", "pass"],
        db_path=env["db"], logs_dir=env["logs"], configs_dir=env["configs"],
        poll_interval=0.05,
    )
    sup.start()
    try:
        assert _wait_for(
            lambda: _task_status(env["db"], tid) == "failed", timeout=5
        )
    finally:
        sup.stop()

    with db.connection_for(env["db"]) as conn:
        task = db.get_task(conn, tid)
    assert "supervisor restart" in (task["error_msg"] or "")


def test_monitor_state_path_passed_to_cmd_and_db(env, monkeypatch) -> None:
    """PP6.1：spawn task 时把 --monitor-state-file 传给 cmd_builder + 写 monitor_state_path 到 db。"""
    captured: dict[str, Any] = {}

    def capturing_cmd(task, cfg):
        captured["cmd_msp"] = task.get("monitor_state_path")
        return [sys.executable, "-c", "import sys; sys.exit(0)"]

    sup = Supervisor(
        cmd_builder=capturing_cmd,
        db_path=env["db"], logs_dir=env["logs"], configs_dir=env["configs"],
        poll_interval=0.05,
    )
    with db.connection_for(env["db"]) as conn:
        tid = db.create_task(conn, name="t", config_name="fake")

    sup.start()
    try:
        assert _wait_for(
            lambda: _task_status(env["db"], tid) == "done", timeout=10
        )
    finally:
        sup.stop()

    # cmd_builder 收到的 task dict 含 monitor_state_path
    assert captured.get("cmd_msp"), "cmd_builder 没拿到 monitor_state_path"
    # task-scoped 档案：tasks/<id>/monitor/state.json
    msp_parts = Path(captured["cmd_msp"]).parts
    assert msp_parts[-4:] == ("tasks", str(tid), "monitor", "state.json")

    # db 也写入了
    with db.connection_for(env["db"]) as conn:
        row = db.get_task(conn, tid)
    assert row["monitor_state_path"] == captured["cmd_msp"]


def test_default_cmd_builder_includes_monitor_flag() -> None:
    """没传 cmd_builder 时，默认行为是把 --monitor-state-file 拼进去。"""
    from studio.supervisor import _default_cmd_builder
    cmd = _default_cmd_builder(
        {"id": 99, "config_name": "x",
         "monitor_state_path": "/tmp/x/state.json"},
        Path("/tmp/cfg.yaml"),
    )
    assert "--monitor-state-file" in cmd
    i = cmd.index("--monitor-state-file")
    assert cmd[i + 1] == "/tmp/x/state.json"


def test_popen_injects_wandb_env(env, tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = secrets.Secrets()
    cfg.wandb.enabled = True
    # 0.18 预设化：字段在当前选中的 preset 上
    wb = cfg.wandb.active
    wb.api_key = "wandb-key"
    wb.project = "anima"
    wb.entity = "team"
    wb.base_url = "https://wandb.example"
    wb.mode = "offline"
    wb.log_samples = False
    monkeypatch.setattr("studio.supervisor._secrets.load", lambda: cfg)
    captured: dict[str, Any] = {}

    class FakePopen:
        pid = 123

    def fake_popen(cmd, **kwargs):  # noqa: ANN001
        captured["cmd"] = cmd
        captured["env"] = kwargs["env"]
        return FakePopen()

    monkeypatch.setattr("studio.supervisor.subprocess.Popen", fake_popen)
    sup = Supervisor(
        db_path=env["db"], logs_dir=env["logs"], configs_dir=env["configs"],
    )
    log_path = tmp_path / "x.log"
    with log_path.open("wb") as fp:
        sup._popen([sys.executable, "-c", "pass"], fp)

    assert captured["env"]["WANDB_ENABLED"] == "1"
    assert captured["env"]["WANDB_API_KEY"] == "wandb-key"
    assert captured["env"]["WANDB_PROJECT"] == "anima"
    assert captured["env"]["WANDB_ENTITY"] == "team"
    assert captured["env"]["WANDB_BASE_URL"] == "https://wandb.example"
    assert captured["env"]["WANDB_MODE"] == "offline"
    assert captured["env"]["WANDB_LOG_SAMPLES"] == "0"
    assert captured["env"]["WANDB_UPLOAD_MODEL"] == "0"
    assert captured["env"]["WANDB_UPLOAD_MODEL_POLICY"] == "last"
    assert captured["env"]["WANDB_UPLOAD_STATE_MANUAL"] == "0"
    assert captured["env"]["WANDB_UPLOAD_STATE_MANUAL_POLICY"] == "last"
    assert captured["env"]["WANDB_UPLOAD_STATE_AUTO"] == "0"
    assert captured["env"]["WANDB_UPLOAD_STATE_AUTO_POLICY"] == "last"


def test_config_path_takes_priority(env, tmp_path) -> None:
    """PP6.3：task.config_path 设了就用它，不再读 _configs_dir。"""
    captured: dict[str, Any] = {}

    explicit_cfg = tmp_path / "private" / "config.yaml"
    explicit_cfg.parent.mkdir(parents=True)
    explicit_cfg.write_text("epochs: 1\n", encoding="utf-8")

    def capturing_cmd(task, cfg):
        captured["cfg"] = str(cfg)
        return [sys.executable, "-c", "import sys; sys.exit(0)"]

    sup = Supervisor(
        cmd_builder=capturing_cmd,
        db_path=env["db"], logs_dir=env["logs"], configs_dir=env["configs"],
        poll_interval=0.05,
    )
    with db.connection_for(env["db"]) as conn:
        tid = db.create_task(conn, name="t", config_name="ignored")
        db.update_task(conn, tid, config_path=str(explicit_cfg))

    sup.start()
    try:
        assert _wait_for(
            lambda: _task_status(env["db"], tid) == "done", timeout=10
        )
    finally:
        sup.stop()

    assert captured["cfg"] == str(explicit_cfg)


def test_finalize_version_writes_output_lora_path(env, tmp_path, monkeypatch) -> None:
    """PP6.3：训练 task 完成 → 推 version.output_lora_path + stage=done。"""
    from studio.services.projects import projects, versions

    monkeypatch.setattr(projects, "PROJECTS_DIR", tmp_path / "projects")

    # 建 project + version + 假 lora_final.safetensors
    with db.connection_for(env["db"]) as conn:
        p = projects.create_project(conn, title="P")
        v = versions.create_version(conn, project_id=p["id"], label="baseline")
    vdir = versions.version_dir(p["id"], p["slug"], "baseline")
    out_lora = vdir / "output" / f"{p['slug']}_baseline_final.safetensors"
    out_lora.parent.mkdir(parents=True, exist_ok=True)
    out_lora.write_bytes(b"fake-safetensors")

    # 用一个秒退 0 的假 cmd
    sup = Supervisor(
        cmd_builder=lambda *_: [sys.executable, "-c", "import sys; sys.exit(0)"],
        db_path=env["db"], logs_dir=env["logs"], configs_dir=env["configs"],
        poll_interval=0.05,
    )
    with db.connection_for(env["db"]) as conn:
        tid = db.create_task(conn, name="t", config_name="fake")
        db.update_task(
            conn, tid, project_id=p["id"], version_id=v["id"]
        )

    sup.start()
    try:
        assert _wait_for(
            lambda: _task_status(env["db"], tid) == "done", timeout=10
        )
    finally:
        sup.stop()

    # ADR-0007 PR-5: 老 stage 不再写；version.status='completed' + output_lora_path 回填
    with db.connection_for(env["db"]) as conn:
        v_after = versions.get_version(conn, v["id"])
    assert v_after["status"] == "completed"
    assert v_after["output_lora_path"] == str(out_lora)


def _task_status(dbfile: Path, tid: int) -> str:
    with db.connection_for(dbfile) as conn:
        task = db.get_task(conn, tid)
    return task["status"] if task else "missing"


# --- 0.17 item1：generate task 走 daemon → 落 run.log + emit task_log_appended ---

def _make_sup(env, on_event):
    return Supervisor(
        on_event=on_event, cmd_builder=lambda t, c: [],
        db_path=env["db"], logs_dir=env["logs"], configs_dir=env["configs"],
        poll_interval=0.05,
    )


def test_daemon_log_line_writes_run_log_and_emits_appended(env) -> None:
    """active generate task 期间的 daemon 日志落该 task run.log + emit task_log_appended。"""
    events, on_event = _events_collector()
    sup = _make_sup(env, on_event)
    tid = 777
    lp = env["tasks"] / str(tid) / "run.log"
    lp.parent.mkdir(parents=True, exist_ok=True)
    sup._daemon_active_task_id = tid
    sup._daemon_log_fp = open(lp, "ab")
    try:
        sup._on_daemon_log_line({"line": "loading model", "ts": 1, "seq": 1})
        sup._on_daemon_log_line({"line": "step 3/20", "ts": 2, "seq": 2})
    finally:
        sup._daemon_log_fp.close()

    content = lp.read_text(encoding="utf-8")
    assert "loading model" in content
    assert "step 3/20" in content
    appended = [
        e for e in events
        if e.get("type") == "task_log_appended" and e.get("task_id") == tid
    ]
    assert [e["text"] for e in appended] == ["loading model", "step 3/20"]
    # 日志抽屉用的 daemon_log_line 仍然照发
    assert any(e.get("type") == "daemon_log_line" for e in events)


def test_daemon_log_line_no_active_task_no_appended(env) -> None:
    """无 active daemon task（idle 噪声）时不 emit task_log_appended，只 daemon_log_line。"""
    events, on_event = _events_collector()
    sup = _make_sup(env, on_event)
    sup._on_daemon_log_line({"line": "idle noise", "ts": 1, "seq": 1})
    assert not any(e.get("type") == "task_log_appended" for e in events)
    assert any(e.get("type") == "daemon_log_line" for e in events)
