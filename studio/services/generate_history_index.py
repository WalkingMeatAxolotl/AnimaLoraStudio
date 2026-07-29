"""落盘测试图历史的 SQLite 索引（sync-on-read 增量缓存）。

背景：`GET /api/generate/disk/history` 此前每次请求全量重扫
`studio_data/test/<date>/{single,xy}/` 下所有 PNG 的 `anima_params` 文本块。
#328 的手写 chunk reader 把单文件解析压到 ~0.1ms，但落盘图从当时的 356 张涨到
2000+ 张后，「每次进页面重扫 + 冷文件缓存 + 每文件 open」重新变成秒级；且
`limit=500` 截断让更老的历史根本列不出来。

方案：PNG 仍是唯一 canonical（#245 决策不变），这里只是**可重建的索引**：

  - 索引落在 `<studio_data>/.cache/disk-history-index.sqlite3`，独立文件、
    不进 studio.db —— 避免给热的主库（supervisor 高频写）添 sync 写竞争；
    删掉文件 = 全量重建，没有迁移负担（user_version 变化直接 DROP 重建）。
  - 每次 list 前先 sync：scandir 快照（只 stat 不 open）与索引 diff —— 新文件
    / stat 变了才重新解析 PNG；盘上消失的行删掉。2000 张的快照在暖缓存下
    ~10ms 级，解析成本只在首次建索引和真正有新图时发生。
  - staleness 判据：single = PNG 的 `mtime_ns:size`；xy = composite 的
    `mtime_ns:size` + **文件夹自身 mtime_ns**（NTFS/ext 下增删 cell 会碰目录
    mtime，覆盖手删 cell 的场景）。
  - 没有 `anima_params` 的 PNG 也记一行（params_json=NULL）做负缓存，否则
    每次 sync 都会重新 open 解析这些老图；列表查询会排除它们。

线程安全：模块级锁把 sync 串行化（多 tab 同时打开测试页时后到的等首个
sync 完直接吃现成索引）；连接 per-call 开关，SQLite 自身保证文件级一致性。
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import sqlite3
import threading
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional
from urllib.parse import quote

logger = logging.getLogger(__name__)

# 前端 params snapshot 的 schema 版本（v1→v2 迁移见 migrate_anima_params）
SCHEMA_VERSION = 2

# 目录 / 文件名约定（与 api/routers/generate.py 的落盘布局共用一套）
DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
XY_FOLDER_RE = re.compile(r"^xy plot (\d+)$")
XY_CELL_RE = re.compile(r"^cell x(\d+) y(\d+)\.png$")
XY_COMPOSITE_NAME = "xy plot.png"

_INDEX_DB_NAME = "disk-history-index.sqlite3"
_INDEX_SCHEMA_VERSION = 1
_SYNC_LOCK = threading.Lock()

_SCHEMA = """
CREATE TABLE IF NOT EXISTS entries (
    id             TEXT PRIMARY KEY,
    date           TEXT NOT NULL,
    mode           TEXT NOT NULL,
    name           TEXT NOT NULL,
    created_at     REAL NOT NULL,
    stat_key       TEXT NOT NULL,
    schema_version INTEGER NOT NULL,
    params_json    TEXT,
    xy_meta_json   TEXT
);
CREATE INDEX IF NOT EXISTS idx_entries_created ON entries(created_at DESC);
"""


# ---------------------------------------------------------------------------
# PNG anima_params 读取 + schema 迁移（从 api/routers/generate.py 原样搬来）
# ---------------------------------------------------------------------------

_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def _decode_png_text_chunk(ctype: bytes, data: bytes) -> str | None:
    """从 PNG 文本 chunk 取出 keyword==`anima_params` 的文本；非该 keyword /
    解析失败返 None。

    - tEXt：keyword\\0 + latin-1 明文
    - zTXt：keyword\\0 + 压缩方法(1 字节) + zlib 压缩流（latin-1）
    - iTXt：keyword\\0 + 压缩 flag(1) + 压缩方法(1) + 语言\\0 + 翻译 keyword\\0 + 文本(utf-8)

    解码用 PIL 读 PNG 文本块的同一套规则（tEXt/zTXt → latin-1，iTXt → utf-8），
    保证与旧 PIL 实现逐值一致。
    """
    keyword, sep, rest = data.partition(b"\x00")
    if not sep or keyword != b"anima_params":
        return None
    try:
        if ctype == b"tEXt":
            return rest.decode("latin-1")
        if ctype == b"zTXt":
            return zlib.decompress(rest[1:]).decode("latin-1") if rest else None
        if ctype == b"iTXt":
            if len(rest) < 2:
                return None
            comp_flag = rest[0]
            body = rest[2:]
            _, _, body = body.partition(b"\x00")  # 跳语言 tag
            _, _, body = body.partition(b"\x00")  # 跳翻译 keyword
            return (zlib.decompress(body) if comp_flag else body).decode("utf-8")
    except Exception:
        return None
    return None


def read_png_anima_params(path: Path) -> dict[str, Any] | None:
    """从 PNG `anima_params` tEXt / zTXt / iTXt 块解析 params；无 / 解析失败返 None。

    直接顺序扫 PNG chunk，读到第一个 IDAT（像素数据起始）前找 `anima_params`
    文本块即停。`anima_params` 由 `PngInfo.add_text(..., zip=True)` 写成 zTXt 且
    位于 IDAT 之前，必然在 header 区命中。

    原实现走 `PIL.Image.open` 只读 header（不 decode 像素），但实测 PIL open 每
    文件 ~30-40ms，disk-history 扫数百张落盘图要 10-15s（冷缓存可达 ~1min）。
    手写 chunk 扫描每文件 ~0.1ms（实测 356 张 15s → 0.04s，~350×），且对全部
    历史 PNG 与旧实现逐值一致。
    """
    try:
        with open(path, "rb") as f:
            if f.read(8) != _PNG_SIGNATURE:
                return None
            while True:
                head = f.read(8)
                if len(head) < 8:
                    return None
                length = int.from_bytes(head[:4], "big")
                ctype = head[4:8]
                if ctype == b"IDAT":
                    return None  # 到像素数据，header 区没有 anima_params
                if ctype in (b"tEXt", b"zTXt", b"iTXt"):
                    text = _decode_png_text_chunk(ctype, f.read(length))
                    f.read(4)  # CRC
                    if text is not None:
                        parsed = json.loads(text)
                        return parsed if isinstance(parsed, dict) else None
                else:
                    f.seek(length + 4, 1)  # 跳 chunk data + CRC（IHDR 等）
    except Exception:
        return None


def migrate_anima_params(meta: dict[str, Any]) -> dict[str, Any]:
    """v1 → v2 schema 迁移（决策 #18）。

    v1: `lora_configs[].path` 是绝对路径（旧 schema 直接存 path）
    v2: `loras[].name` basename + project_id/version_id；不存绝对路径

    迁移规则：v1 PNG → 把 `lora_configs[].path` 末段 basename 当 v2 `loras[].name`，
    保留 project_id/version_id/scale；旧 path 丢弃（隐私 + 跨机器死链）。
    """
    version = meta.get("schema_version", 1)
    if version >= 2:
        return meta
    if version == 1:
        legacy_loras = meta.pop("lora_configs", None)
        if isinstance(legacy_loras, list):
            new_loras: list[dict[str, Any]] = []
            for lc in legacy_loras:
                if not isinstance(lc, dict):
                    continue
                path = str(lc.get("path") or "")
                name = path.replace("\\", "/").rsplit("/", 1)[-1] if path else ""
                new_loras.append({
                    "name": name,
                    "scale": float(lc.get("scale", 1.0)),
                    "project_id": lc.get("project_id"),
                    "version_id": lc.get("version_id"),
                })
            meta["loras"] = new_loras
        meta["schema_version"] = 2
        return meta
    # 未知版本 → 当作 v2 透传（forward-compat）
    return meta


def disk_history_id(date_str: str, mode: str, filename: str) -> str:
    """前端 dedup / merge 用的稳定 id。

    用 sha1 短哈希替代直接拼 filename —— filename 含空格（决策 #6 "single image 1"）
    塞进 React key / data-testid / URL fragment 会踩坑。哈希 12 位足够全局唯一。
    """
    h = hashlib.sha1(f"{date_str}/{mode}/{filename}".encode("utf-8")).hexdigest()[:12]
    return f"disk:{h}"


def url_quote_filename(filename: str) -> str:
    """文件名内空格 / 中文等 URL encode（决策 #6 文件名带空格）。后端返 URL
    时直接 encode 好，前端拼接禁止。"""
    return quote(filename, safe="")


def build_xy_meta_from_folder(
    folder: Path, composite_params: dict[str, Any], date_str: str, folder_name: str,
) -> dict[str, Any] | None:
    """读 XY 文件夹下所有 cell 文件，按 composite 的 xy_draft 反查 xv/yv，
    拼成 disk-history entry 的 `xy_meta` 字段。

    决策：只读 composite 的 anima_params（一次 file open），cell 的 xi/yi
    从文件名 parse（regex），xv/yv 从 composite.xy_draft.x.raw/y.raw split
    后查表。**不**逐 cell 打开 anima_params —— 5×5 矩阵 1 次 open 而非 26 次。

    返回 None 表示 composite 缺 xy_draft（异常状态，前端兜底走 <img>）。
    """
    xy_draft = composite_params.get("xy_draft")
    if not isinstance(xy_draft, dict):
        return None
    x_axis_info = xy_draft.get("x")
    if not isinstance(x_axis_info, dict):
        return None
    x_raw = str(x_axis_info.get("raw", ""))
    x_values = [s.strip() for s in x_raw.split(",") if s.strip()]
    x_axis = x_axis_info.get("axis")

    y_axis_info = xy_draft.get("y") if xy_draft.get("y") else None
    y_values: list[str | None]
    y_axis: str | None
    if isinstance(y_axis_info, dict):
        y_raw = str(y_axis_info.get("raw", ""))
        y_values = [s.strip() for s in y_raw.split(",") if s.strip()]
        y_axis = y_axis_info.get("axis")
    else:
        y_values = [None]
        y_axis = None

    samples: list[dict[str, Any]] = []
    for cell_file in folder.glob("cell x*.png"):
        m = XY_CELL_RE.match(cell_file.name)
        if not m:
            continue
        xi = int(m.group(1))
        yi = int(m.group(2))
        xv: str | None = x_values[xi] if 0 <= xi < len(x_values) else None
        yv: str | None = y_values[yi] if 0 <= yi < len(y_values) else None
        enc_folder = url_quote_filename(folder_name)
        enc_file = url_quote_filename(cell_file.name)
        samples.append({
            "path": cell_file.name,
            "xy": {"xi": xi, "yi": yi, "xv": xv, "yv": yv},
            "image_url": f"/api/generate/disk/image/{date_str}/xy/{enc_folder}/{enc_file}",
        })
    samples.sort(key=lambda s: (s["xy"]["yi"], s["xy"]["xi"]))
    return {
        "x_axis": x_axis,
        "y_axis": y_axis,
        "x_values": x_values,
        "y_values": y_values,
        "samples": samples,
    }


# ---------------------------------------------------------------------------
# 索引：连接 / 磁盘快照 / sync
# ---------------------------------------------------------------------------


@dataclass
class _DiskItem:
    """磁盘快照里的一个 history 单位（single 一个 PNG / xy 一个文件夹）。"""
    entry_id: str
    date: str
    mode: str          # 'single' | 'xy'
    name: str          # single: 文件名；xy: 文件夹名
    stat_key: str      # staleness 判据（见模块 docstring）
    created_at: float  # single: PNG mtime；xy: composite mtime（与旧扫描口径一致）
    path: Path         # single: PNG 路径；xy: 文件夹路径


def index_db_path(root: Path) -> Path:
    """索引文件位置：`<root 的父目录>/.cache/disk-history-index.sqlite3`。

    root 生产环境是 `studio_data/test` → 索引在 `studio_data/.cache/` 下；
    测试里 root 是 tmp 目录 → 索引自动隔离在同一 tmp 下，fixture 零改动。
    """
    return root.parent / ".cache" / _INDEX_DB_NAME


def _connect(root: Path) -> sqlite3.Connection:
    path = index_db_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    ver = int(conn.execute("PRAGMA user_version").fetchone()[0])
    if ver != _INDEX_SCHEMA_VERSION:
        # 纯缓存：schema 变化直接重建，全量数据从 PNG 重新解析即可
        conn.executescript("DROP TABLE IF EXISTS entries;")
        conn.executescript(_SCHEMA)
        conn.execute(f"PRAGMA user_version = {_INDEX_SCHEMA_VERSION}")
        conn.commit()
    return conn


def _snapshot_disk(root: Path) -> dict[str, _DiskItem]:
    """scandir 快照 —— 只 stat 不 open 文件内容。返回 id → _DiskItem。"""
    out: dict[str, _DiskItem] = {}
    if not root.is_dir():
        return out
    for date_dir in root.iterdir():
        if not date_dir.is_dir() or not DATE_RE.match(date_dir.name):
            continue
        date_str = date_dir.name
        single_dir = date_dir / "single"
        if single_dir.is_dir():
            with os.scandir(single_dir) as it:
                for de in it:
                    name = de.name
                    if not name.lower().endswith(".png") or name.endswith(".tmp.png"):
                        continue
                    if not de.is_file():
                        continue
                    try:
                        st = de.stat()
                    except OSError:
                        continue
                    eid = disk_history_id(date_str, "single", name)
                    out[eid] = _DiskItem(
                        entry_id=eid, date=date_str, mode="single", name=name,
                        stat_key=f"{st.st_mtime_ns}:{st.st_size}",
                        created_at=float(st.st_mtime), path=Path(de.path),
                    )
        xy_dir = date_dir / "xy"
        if xy_dir.is_dir():
            for folder in xy_dir.iterdir():
                # legacy 平铺 `xy plot N.png` 文件不入 history（用户决策）
                if not folder.is_dir() or not XY_FOLDER_RE.match(folder.name):
                    continue
                composite = folder / XY_COMPOSITE_NAME
                try:
                    cst = composite.stat()
                    fst = folder.stat()
                except OSError:
                    continue  # 没 composite 的半成品文件夹跳过
                eid = disk_history_id(date_str, "xy", folder.name)
                out[eid] = _DiskItem(
                    entry_id=eid, date=date_str, mode="xy", name=folder.name,
                    # 文件夹 mtime 参与 staleness：增删 cell 会碰目录 mtime
                    stat_key=f"{cst.st_mtime_ns}:{cst.st_size}:{fst.st_mtime_ns}",
                    created_at=float(cst.st_mtime), path=folder,
                )
    return out


def _parse_item(item: _DiskItem) -> dict[str, Any]:
    """解析一个磁盘单位 → 索引行值。没有 anima_params 时 params_json=None
    （负缓存行：留着 stat_key 防每次 sync 重新 open，但不出现在列表里）。"""
    png = item.path if item.mode == "single" else item.path / XY_COMPOSITE_NAME
    params = read_png_anima_params(png)
    if params is None:
        return {"schema_version": SCHEMA_VERSION, "params_json": None, "xy_meta_json": None}
    params = migrate_anima_params(params)
    xy_meta_json: Optional[str] = None
    if item.mode == "xy":
        xy_meta = build_xy_meta_from_folder(item.path, params, item.date, item.name)
        if xy_meta is not None:
            xy_meta_json = json.dumps(xy_meta, ensure_ascii=False)
    return {
        "schema_version": int(params.get("schema_version", SCHEMA_VERSION)),
        "params_json": json.dumps(params, ensure_ascii=False),
        "xy_meta_json": xy_meta_json,
    }


def _row_to_entry(root: Path, row: sqlite3.Row) -> dict[str, Any]:
    """索引行 → API entry（shape 与旧全量扫描逐字段一致）。URL / path 现拼
    不入库，编码规则或 root 位置变了不用重建索引。"""
    date_str, mode, name = row["date"], row["mode"], row["name"]
    params = json.loads(row["params_json"])
    # migrate 幂等：schema_version >= 当前直接透传。这里再过一遍，未来加 v3
    # 迁移时老索引行不用重建也能出对的 shape。
    params = migrate_anima_params(params)
    entry: dict[str, Any] = {
        "id": row["id"],
        "date": date_str,
        "mode": mode,
        "path": str(root / date_str / mode / name),
        "created_at": float(row["created_at"]),
        "schema_version": int(row["schema_version"]),
        "params": params,
    }
    if mode == "single":
        enc = url_quote_filename(name)
        entry["filename"] = name
        entry["image_url"] = f"/api/generate/disk/image/{date_str}/single/{enc}"
        entry["thumb_url"] = f"/api/generate/disk/thumb/{date_str}/single/{enc}?w=128"
    else:
        enc_folder = url_quote_filename(name)
        enc_composite = url_quote_filename(XY_COMPOSITE_NAME)
        entry["folder"] = name
        entry["image_url"] = f"/api/generate/disk/image/{date_str}/xy/{enc_folder}/{enc_composite}"
        entry["thumb_url"] = f"/api/generate/disk/thumb/{date_str}/xy/{enc_folder}/{enc_composite}?w=128"
        entry["xy_meta"] = json.loads(row["xy_meta_json"]) if row["xy_meta_json"] else None
    return entry


def sync_and_list(root: Path, limit: int) -> list[dict[str, Any]]:
    """增量 sync 索引后按 created_at desc 返回前 limit 条 entry。

    首次调用（索引为空）等价于一次全量扫描解析；之后每次只付 scandir 快照
    + 变化文件的解析成本。
    """
    with _SYNC_LOCK:
        conn = _connect(root)
        try:
            disk = _snapshot_disk(root)
            known = {
                r["id"]: r["stat_key"]
                for r in conn.execute("SELECT id, stat_key FROM entries")
            }
            gone = [eid for eid in known if eid not in disk]
            if gone:
                conn.executemany(
                    "DELETE FROM entries WHERE id = ?", [(g,) for g in gone],
                )
            dirty = [
                item for eid, item in disk.items()
                if known.get(eid) != item.stat_key
            ]
            for item in dirty:
                parsed = _parse_item(item)
                conn.execute(
                    "INSERT OR REPLACE INTO entries "
                    "(id, date, mode, name, created_at, stat_key, schema_version, "
                    " params_json, xy_meta_json) VALUES (?,?,?,?,?,?,?,?,?)",
                    (
                        item.entry_id, item.date, item.mode, item.name,
                        item.created_at, item.stat_key, parsed["schema_version"],
                        parsed["params_json"], parsed["xy_meta_json"],
                    ),
                )
            if gone or dirty:
                conn.commit()
                logger.info(
                    "disk-history index synced: +%d updated, -%d removed, %d total",
                    len(dirty), len(gone), len(disk),
                )
            rows = conn.execute(
                "SELECT * FROM entries WHERE params_json IS NOT NULL "
                "ORDER BY created_at DESC, id LIMIT ?",
                (limit,),
            ).fetchall()
            return [_row_to_entry(root, r) for r in rows]
        finally:
            conn.close()


def remove_entry(root: Path, date_str: str, mode: str, name: str) -> None:
    """DELETE 端点删文件后同步剔行（不删也会在下次 sync 时被 diff 掉，
    这里只是让紧随其后的 list 立即一致）。失败静默 —— 索引是缓存。"""
    try:
        with _SYNC_LOCK:
            conn = _connect(root)
            try:
                conn.execute(
                    "DELETE FROM entries WHERE id = ?",
                    (disk_history_id(date_str, mode, name),),
                )
                conn.commit()
            finally:
                conn.close()
    except Exception:
        logger.warning("disk-history index remove failed", exc_info=True)
