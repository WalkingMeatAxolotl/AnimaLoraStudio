"""诊断包（docs/design/logging-target-state.md §3.6）。

用户报 issue 时一个按钮导出 zip，开发者拿到就够，不用来回要文件：

    README.txt                  里面有什么、怎么生成的、脱敏规则
    env.json                    studio 版本 / Python / 平台 / 驱动 / CUDA / torch 状态
    task.json                   任务行（tasks 表，含 status / error_msg / 起止时间）
    task/run.log                该任务完整 run.log（原样，经脱敏）
    task/snapshot/config.yaml   训练配置快照（若有）
    task/monitor/state.json     训练指标快照（若有）
    studio.log                  webui 进程日志在任务起止时间窗内的片段（含轮转文件），
                                无 task 时取尾部 STUDIO_LOG_TAIL_BYTES

子进程不落 studio.log（D2）——所以诊断包是把两条线按时间窗拼在一起的地方。

脱敏：不含 secrets.json；run.log / studio.log 文本过一遍 REDACT_PATTERNS
（api_key / token / Authorization / hf_ 与 sk- 形态的 key）。不是加密级保证，
只是防止常见 key 顺手泄漏；README 里写明让用户发出前自己再过目。
"""
from __future__ import annotations

import datetime as _dt
import io
import json
import logging
import re
import time
import zipfile
from pathlib import Path
from typing import Any

from .. import __version__
from ..infrastructure import db
from ..infrastructure.logging import STUDIO_LOG_NAME
from ..infrastructure.paths import LOGS_DIR, task_dir, task_log_path

logger = logging.getLogger(__name__)

#: 任务起止时间窗两侧各放宽的秒数（supervisor 拉起前 / 收尾后的日志也要）
WINDOW_PAD_SECONDS = 60
#: 无 task 时 studio.log 取尾部多少字节
STUDIO_LOG_TAIL_BYTES = 5 * 1024 * 1024
#: studio.log 片段上限（防窗口太宽把 150MB 全塞进去）
STUDIO_LOG_SLICE_MAX_BYTES = 30 * 1024 * 1024

_TS_RE = re.compile(r'"ts":\s*"(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3})Z"')

REDACT_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    # key=value 形态（query string / env 打印 / yaml）
    (re.compile(r"(?i)((?:api[_-]?key|access[_-]?token|auth[_-]?token|secret|password|passwd|token)\s*[=:]\s*)([^\s&\"',;]+)"), r"\1***"),
    # Authorization: Bearer xxx / Basic xxx
    (re.compile(r"(?i)(authorization\s*:\s*\w+\s+)(\S+)"), r"\1***"),
    # HuggingFace / OpenAI 形态的 key
    (re.compile(r"\bhf_[A-Za-z0-9]{16,}\b"), "hf_***"),
    (re.compile(r"\bsk-[A-Za-z0-9_-]{16,}\b"), "sk-***"),
)


def redact(text: str) -> str:
    for pat, repl in REDACT_PATTERNS:
        text = pat.sub(repl, text)
    return text


def _env_info(extra: dict[str, Any] | None) -> dict[str, Any]:
    """机器 / 运行时事实；每一项单独兜底，一个探测失败不拖垮整包。

    `extra` 由 API 层传入（如 /api/env/summary 的结果）——services 不反向 import
    routers（4 层依赖方向）。"""
    import platform as platform_mod  # noqa: PLC0415

    out: dict[str, Any] = {
        "studio_version": __version__,
        "generated_at": _dt.datetime.now(tz=_dt.timezone.utc).isoformat(timespec="seconds"),
        "python_version": platform_mod.python_version(),
        "platform": platform_mod.platform(),
    }
    try:
        from .runtime import torch as torch_setup  # noqa: PLC0415

        out["torch"] = torch_setup.current_status()
    except Exception as e:  # noqa: BLE001
        out["torch_error"] = str(e)
    if extra:
        out.update(extra)
    return out


def _task_row(task_id: int) -> dict[str, Any] | None:
    with db.connection_for() as conn:
        return db.get_task(conn, task_id)


def _iter_studio_log_files() -> list[Path]:
    """studio.log + 轮转文件，按时间从旧到新（.N 越大越旧）。"""
    base = LOGS_DIR / STUDIO_LOG_NAME
    rotated = sorted(
        LOGS_DIR.glob(STUDIO_LOG_NAME + ".*"),
        key=lambda p: int(p.suffix[1:]) if p.suffix[1:].isdigit() else 0,
        reverse=True,
    )
    return [*rotated, base]


def _parse_ts(line: str) -> float | None:
    m = _TS_RE.search(line)
    if not m:
        return None
    try:
        return _dt.datetime.strptime(m.group(1), "%Y-%m-%dT%H:%M:%S.%f").replace(
            tzinfo=_dt.timezone.utc
        ).timestamp()
    except ValueError:
        return None


def studio_log_slice(start: float, end: float, *, max_bytes: int = STUDIO_LOG_SLICE_MAX_BYTES) -> str:
    """studio.log（含轮转）里 ts ∈ [start, end] 的行。按 ts 字段正则取时间，不
    json.loads 每行（150MB 量级的文件也要几秒内扫完）。"""
    chunks: list[str] = []
    size = 0
    for p in _iter_studio_log_files():
        if not p.exists():
            continue
        try:
            with open(p, "r", encoding="utf-8", errors="replace") as f:
                for line in f:
                    ts = _parse_ts(line)
                    if ts is None or ts < start or ts > end:
                        continue
                    chunks.append(line)
                    size += len(line)
                    if size >= max_bytes:
                        chunks.append(f"... truncated at {max_bytes} bytes ...\n")
                        return "".join(chunks)
        except OSError:
            logger.exception("read %s failed", p)
    return "".join(chunks)


def studio_log_tail(max_bytes: int = STUDIO_LOG_TAIL_BYTES) -> str:
    p = LOGS_DIR / STUDIO_LOG_NAME
    if not p.exists():
        return ""
    size = p.stat().st_size
    with open(p, "rb") as f:
        if size > max_bytes:
            f.seek(size - max_bytes)
            f.readline()
        return f.read().decode("utf-8", errors="replace")


def _readme(task_id: int | None, window: tuple[float, float] | None, members: list[str]) -> str:
    lines = [
        "AnimaLoraStudio 诊断包 / diagnostics bundle",
        f"studio version: {__version__}",
        f"generated: {_dt.datetime.now().isoformat(timespec='seconds')}",
        "",
    ]
    if task_id is not None:
        lines.append(f"task_id: {task_id}")
    if window:
        lines.append(
            "studio.log window: "
            f"{_dt.datetime.fromtimestamp(window[0]).isoformat(timespec='seconds')} ~ "
            f"{_dt.datetime.fromtimestamp(window[1]).isoformat(timespec='seconds')}"
            f"（任务起止各放宽 {WINDOW_PAD_SECONDS}s）"
        )
    lines += [
        "",
        "contents:",
        *[f"  {m}" for m in members],
        "",
        "脱敏 / redaction: 不含 secrets.json；run.log 与 studio.log 已过 api_key / token /",
        "Authorization / hf_* / sk-* 模式脱敏。发出前请自行过目一遍，路径中可能含用户名。",
    ]
    return "\n".join(lines) + "\n"


def build_bundle(task_id: int | None, *, extra_env: dict[str, Any] | None = None) -> tuple[bytes, str]:
    """生成 zip 字节与建议文件名。task_id 不存在抛 LookupError。"""
    task = None
    window: tuple[float, float] | None = None
    if task_id is not None:
        task = _task_row(task_id)
        if task is None:
            raise LookupError(f"task {task_id} not found")
        started = task.get("started_at") or task.get("created_at") or time.time()
        finished = task.get("finished_at") or time.time()
        window = (float(started) - WINDOW_PAD_SECONDS, float(finished) + WINDOW_PAD_SECONDS)

    buf = io.BytesIO()
    members: list[str] = []
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as z:
        def add(name: str, data: str | bytes) -> None:
            z.writestr(name, data)
            members.append(name)

        add("env.json", json.dumps(_env_info(extra_env), ensure_ascii=False, indent=2, default=str))
        if task is not None and task_id is not None:
            add("task.json", json.dumps(task, ensure_ascii=False, indent=2, default=str))
            lp = task_log_path(task_id)
            if not lp.exists():
                lp = LOGS_DIR / f"{task_id}.log"
            if lp.exists():
                add("task/run.log", redact(lp.read_bytes().decode("utf-8", errors="replace")))
            for rel in ("snapshot/config.yaml", "monitor/state.json"):
                p = task_dir(task_id) / rel
                if p.exists():
                    add(f"task/{rel}", redact(p.read_bytes().decode("utf-8", errors="replace")))
            assert window is not None
            add("studio.log", redact(studio_log_slice(*window)))
        else:
            add("studio.log", redact(studio_log_tail()))
        z.writestr("README.txt", _readme(task_id, window, members))

    stamp = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    name = f"anima-diag-task{task_id}-{stamp}.zip" if task_id is not None else f"anima-diag-{stamp}.zip"
    return buf.getvalue(), name
