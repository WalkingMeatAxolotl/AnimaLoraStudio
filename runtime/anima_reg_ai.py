#!/usr/bin/env python3
"""先验生成 — base 模型对每张训练图的 tag 反向出对照图作正则集。

设计来自 DreamBooth prior preservation：训练损失同时见到「LoRA 学到的样子」和
「base 模型本来的样子」，让 LoRA 只学差异、不污染 base 概念。

**不带 LoRA** —— 出现 LoRA 反而把要保留的 prior 给覆盖了。

用法：
    python runtime/anima_reg_ai.py --config reg_ai_config.json [--monitor-state-file state.json]

逻辑：
  1. 扫 train 目录所有图 + caption
  2. 每张图先把训练图同名 tag 文件复制为 reg 输出图的同名 tag 文件
  3. 从 reg 侧 tag 文件读取 tags，去除 excluded，按 Anima 空格 tag 规范拼 prompt
     并把 reg 侧 tag 文件重写为实际 prompt（JSON 保持标准 JSON 形态）
  4. 输出到 reg/{对应子文件夹}/{stem}_ai_{seed}.png（镜像 train 子目录结构）
  5. reg/meta.json 写 generation_method="ai_base", api_source=""
     （与 booru 拉取共用 reg_builder.RegMeta schema，不再撞名重写）

incremental=True：跳过 reg 子文件夹中已有以 train_stem 开头的图（重启续跑用）。
"""
from __future__ import annotations

import argparse
import json
import logging
import random
import shutil
import sys
import time
from collections import Counter
from pathlib import Path

import torch

# anima_train + train_monitor 都在 runtime/ 同目录，_THIS_DIR 即够。
_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parent
for _p in (_THIS_DIR, _REPO_ROOT):
    s = str(_p)
    if s not in sys.path:
        sys.path.insert(0, s)

import anima_train as _T  # noqa: E402

# 复用 reg_builder.RegMeta（PR-9 commit 2 加了 generation_method 字段）+ clear_reg_dir
# （booru full-mode build 入口同款实现，行为/语义跟 booru reg 路径绑定一致）。
from studio.services.reg.builder import (  # noqa: E402
    RegMeta,
    clear_reg_dir,
    read_meta,
    write_meta,
)
from studio.services.tagging.caption_format import (  # noqa: E402
    caption_json_to_tags,
    caption_json_to_text,
    normalize_caption_json,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("anima_reg_ai")

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Anima 先验生成（base 模型反向出 reg 集）")
    p.add_argument("--config", required=True)
    p.add_argument("--monitor-state-file", default="")
    return p.parse_args()


CAPTION_SUFFIXES = (".json", ".txt", ".caption")


def _tag_key(tag: str) -> str:
    """Canonical key for matching/excluding tags.

    Anima prompts should use spaces, not underscores.  We still treat spaces and
    underscores as equivalent for exclude matching so older UI/booru-style
    excluded tags continue to work.
    """
    return " ".join(str(tag or "").strip().lower().replace("_", " ").split())


def _dedupe_tags(tags: list[str]) -> list[str]:
    """去重但保留原始 tag 文本与顺序。"""
    out: list[str] = []
    seen: set[str] = set()
    for tag in tags:
        text = str(tag or "").strip()
        key = _tag_key(text)
        if not text or key in seen:
            continue
        seen.add(key)
        out.append(text)
    return out


def _prompt_tag(tag: str) -> str:
    """Normalize one tag for Anima text encoders: lowercase, spaces, no underscores."""
    return _tag_key(tag)


def _read_json_tags(json_path: Path) -> list[str]:
    data = json.loads(json_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        return []

    tags: list[str] = []
    meta = data.get("meta")
    if isinstance(meta, dict):
        trigger = meta.get("trigger")
        if isinstance(trigger, str) and trigger.strip():
            tags.append(trigger.strip())
    tags.extend(caption_json_to_tags(data))
    return _dedupe_tags(tags)


def _read_text_tags(caption_path: Path) -> list[str]:
    raw = caption_path.read_text(encoding="utf-8", errors="ignore").strip()
    if not raw:
        return []
    if "," in raw:
        return [t.strip() for t in raw.split(",") if t.strip()]
    return [t.strip() for t in raw.split() if t.strip()]


def _caption_candidates_for_image(img_path: Path) -> list[Path]:
    return [
        p for suffix in CAPTION_SUFFIXES
        if (p := img_path.with_suffix(suffix)).exists()
    ]


def _read_tags_from_caption(caption_path: Path) -> list[str]:
    if caption_path.suffix == ".json":
        return _read_json_tags(caption_path)
    return _read_text_tags(caption_path)


def _caption_path_for_image(img_path: Path) -> Path | None:
    """Return the first readable sidecar caption path, preferring JSON."""
    for p in _caption_candidates_for_image(img_path):
        try:
            _read_tags_from_caption(p)
            return p
        except Exception as e:
            logger.warning("caption 读取失败 %s: %s", p, e)
    return None


def _read_tags(img_path: Path) -> list[str]:
    """读图片旁边的 caption，返回 raw tag 列表（不归一化）。

    JSON caption 优先，TXT/CAPTION 作为回退；与训练数据集 / 标签编辑器保持
    同一套 JSON 语义，避免先验生成在 Step 4 选择 JSON 打标时拿不到 prompt。
    """
    caption_path = _caption_path_for_image(img_path)
    if caption_path is None:
        return []
    return _read_tags_from_caption(caption_path)


def _copy_caption_for_reg(train_img_path: Path, out_img_path: Path) -> Path | None:
    """Copy train sidecar caption to the generated reg image sidecar path."""
    src = _caption_path_for_image(train_img_path)
    if src is None:
        return None
    suffix = ".txt" if src.suffix == ".caption" else src.suffix
    dst = out_img_path.with_suffix(suffix)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)
    return dst


def _build_prompt_from_caption(caption_path: Path, excluded_tags: set[str]) -> str:
    """Build an Anima prior prompt from a reg-side caption file."""
    return ", ".join(_prompt_tags_from_caption(caption_path, excluded_tags))


def _prompt_tags_from_caption(caption_path: Path, excluded_tags: set[str]) -> list[str]:
    """Return normalized prompt tags from a caption file."""
    prompt_tags: list[str] = []
    seen: set[str] = set()
    for raw_tag in _read_tags_from_caption(caption_path):
        tag = _prompt_tag(raw_tag)
        if not tag or tag in excluded_tags or tag in seen:
            continue
        seen.add(tag)
        prompt_tags.append(tag)
    return prompt_tags


def _filter_normalized_caption(
    data: dict, excluded_tags: set[str]
) -> dict:
    """Drop excluded tags + meta.trigger from a normalized standard-shape caption.

    Reg 端不带 trigger：base prior 不认识 LoRA handle；同时避免 reg sidecar 被
    训练侧 caption_utils.load_and_build_caption 再次读到 trigger 注入。

    Scalar 字段（count/character/series/artist）按逗号拆开后逐 tag 过 excluded
    再 join 回，让 "1girl, 1boy" 这种合并值的单项 exclude 可以命中。
    """
    src_tags = data.get("tags") or {}

    def _keep_list(values: list[str]) -> list[str]:
        return [t for t in values if _tag_key(t) not in excluded_tags]

    def _keep_scalar(value: str) -> str:
        kept = [
            t.strip() for t in str(value or "").split(",")
            if t.strip() and _tag_key(t) not in excluded_tags
        ]
        return ", ".join(kept)

    meta = {
        k: v for k, v in (data.get("meta") or {}).items() if k != "trigger"
    }
    return {
        "meta": meta,
        "tags": {
            "quality": _keep_list(src_tags.get("quality") or []),
            "count": _keep_scalar(src_tags.get("count") or ""),
            "character": _keep_scalar(src_tags.get("character") or ""),
            "series": _keep_scalar(src_tags.get("series") or ""),
            "artist": _keep_scalar(src_tags.get("artist") or ""),
            "appearance": _keep_list(src_tags.get("appearance") or []),
            "tags": _keep_list(src_tags.get("tags") or []),
            "environment": _keep_list(src_tags.get("environment") or []),
            "nl": str(src_tags.get("nl") or "").strip(),
        },
    }


def _rewrite_json_caption_for_prompt(caption_path: Path, excluded_tags: set[str]) -> str:
    """Normalize → filter excluded + drop trigger → write back standard shape.

    Reg sidecar 是派生产物，统一写 caption_format.normalize_caption_json 输出的
    标准 shape；训练侧 caption_utils.load_and_build_caption 读 reg JSON 也走同一
    个 normalize 入口，不需要保留 user-side 原始 documented_full / simplified
    形态，反而消掉了 4 套 shape filter 的镜像维护成本。
    """
    raw = json.loads(caption_path.read_text(encoding="utf-8"))
    normalized = normalize_caption_json(raw if isinstance(raw, dict) else {})
    filtered = _filter_normalized_caption(normalized, excluded_tags)
    prompt = caption_json_to_text(filtered)
    caption_path.write_text(
        json.dumps(filtered, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return prompt


def _rewrite_caption_for_prompt(caption_path: Path, excluded_tags: set[str]) -> str:
    """Persist the reg sidecar caption that corresponds to the generated image."""
    if caption_path.suffix == ".json":
        return _rewrite_json_caption_for_prompt(caption_path, excluded_tags)

    prompt = _build_prompt_from_caption(caption_path, excluded_tags)
    caption_path.write_text(prompt, encoding="utf-8")
    return prompt


def _normalize(tag: str) -> str:
    return _tag_key(tag)


def _scan_train(train_dir: Path) -> list[dict]:
    """扫 train 目录，返回每张图的信息列表。

    元素: {"subfolder": str, "stem": str, "img": Path, "tags": list[str]}
    subfolder="" 表示 train 根目录。
    """
    entries: list[dict] = []
    train_dir = train_dir.resolve()

    def _scan(folder: Path, sub: str) -> None:
        for f in sorted(folder.iterdir()):
            if f.is_file() and f.suffix.lower() in IMAGE_EXTS:
                entries.append({
                    "subfolder": sub,
                    "stem": f.stem,
                    "img": f,
                    "tags": _read_tags(f),
                })
            elif f.is_dir():
                child = f.name if not sub else f"{sub}/{f.name}"
                _scan(f, child)

    _scan(train_dir, "")
    return entries


def _already_has_reg(reg_sub: Path, train_stem: str) -> bool:
    """incremental: reg 子目录里已有以 train_stem 开头的图就跳过。"""
    if not reg_sub.exists():
        return False
    for f in reg_sub.iterdir():
        if (
            f.is_file()
            and f.stem.startswith(train_stem)
            and f.suffix.lower() in IMAGE_EXTS
        ):
            return True
    return False


def _write_meta_final(
    reg_dir: Path,
    entries: list[dict],
    excluded_tags: set,
    incremental: bool,
    actual_count: int,
) -> None:
    """写 reg/meta.json 用 reg_builder.RegMeta（generation_method='ai_base'）。

    与 booru 拉取共享 schema；api_source 留空（先验生成无 booru 来源）。
    incremental_runs 在已有 meta 基础上 +1（与 PP5.1 booru 行为一致）。
    """
    prior = read_meta(reg_dir)
    runs = (prior.incremental_runs + 1) if (incremental and prior) else 0

    tag_dist: Counter = Counter()
    for e in entries:
        tag_dist.update(e["tags"])

    meta = RegMeta(
        generated_at=time.time(),
        based_on_version="",
        api_source="",  # 先验生成无 booru 来源
        target_count=len(entries),
        actual_count=actual_count + (prior.actual_count if (incremental and prior) else 0),
        source_tags=[],
        excluded_tags=sorted(excluded_tags),
        blacklist_tags=[],
        failed_tags=[],
        train_tag_distribution=dict(tag_dist.most_common(50)),
        auto_tagged=False,
        incremental_runs=runs,
        generation_method="ai_base",
    )
    write_meta(reg_dir, meta)


def main() -> None:
    args = parse_args()
    cfg_path = Path(args.config)
    if not cfg_path.exists():
        logger.error(f"配置文件不存在: {cfg_path}")
        sys.exit(1)

    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))

    train_dir = Path(cfg["train_dir"])
    reg_dir = Path(cfg["reg_dir"])
    excluded_tags: set = {_normalize(t) for t in cfg.get("excluded_tags", [])}
    negative_prompt: str = cfg.get("negative_prompt", "")
    width: int = int(cfg.get("width", 1024))
    height: int = int(cfg.get("height", 1024))
    steps: int = int(cfg.get("steps", 25))
    cfg_scale: float = float(cfg.get("cfg_scale", 4.0))
    sampler_name: str = cfg.get("sampler_name", "er_sde")
    scheduler: str = cfg.get("scheduler", "simple")
    base_seed: int = int(cfg.get("seed", 0))
    incremental: bool = bool(cfg.get("incremental", False))
    mixed_precision: str = cfg.get("mixed_precision", "bf16")
    backend: str = cfg.get("attention_backend", "flash_attn")
    use_flash = (backend == "flash_attn")
    use_xformers = (backend == "xformers")

    transformer_path: str = cfg["transformer_path"]
    vae_path: str = cfg["vae_path"]
    text_encoder_path: str = cfg["text_encoder_path"]
    t5_tokenizer_path: str = cfg.get("t5_tokenizer_path", "")

    # monitor (fallback 到 reg/.monitor_state.json，与 anima_generate 行为对齐)
    state_file = args.monitor_state_file or str(reg_dir / "monitor_state.json")
    _update_monitor = None
    try:
        from train_monitor import set_state_file, update_monitor
        set_state_file(state_file)
        update_monitor(config={"type": "reg_ai"})
        _update_monitor = update_monitor
    except Exception as e:
        logger.warning(f"monitor 初始化失败: {e}")

    if not train_dir.exists():
        logger.error(f"train 目录不存在: {train_dir}")
        sys.exit(1)

    entries = _scan_train(train_dir)
    if not entries:
        logger.error("train 目录没有任何图片")
        sys.exit(1)

    logger.info(f"train 共 {len(entries)} 张图")

    if incremental:
        to_generate = [
            e for e in entries
            if not _already_has_reg(
                reg_dir / e["subfolder"] if e["subfolder"] else reg_dir,
                e["stem"],
            )
        ]
        logger.info(f"incremental 模式：需生成 {len(to_generate)}/{len(entries)} 张")
    else:
        to_generate = entries

    if not to_generate:
        logger.info("所有图片已有对应正则图，无需生成")
        _write_meta_final(reg_dir, entries, excluded_tags, incremental, 0)
        return

    # 加载 base 模型（不带 LoRA）
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if mixed_precision == "bf16" else torch.float32

    repo_root = _T.find_diffusion_pipe_root()
    bases = [Path.cwd(), _THIS_DIR, repo_root]
    transformer_path = _T.resolve_path_best_effort(transformer_path, bases)
    vae_path = _T.resolve_path_best_effort(vae_path, bases)
    text_encoder_path = _T.resolve_path_best_effort(text_encoder_path, bases)
    if t5_tokenizer_path:
        t5_tokenizer_path = _T.resolve_path_best_effort(t5_tokenizer_path, bases)

    family = _T.resolve_family(cfg)  # D8'
    logger.info("加载 Transformer...")
    model = family.load_dit(
        transformer_path, device, dtype,
        attention_backend=("flash_attn" if use_flash else "none"), repo_root=repo_root,
        purpose="generate",
    )
    if use_xformers:
        _T.enable_xformers(model)

    logger.info("加载 VAE...")
    vae = family.load_vae(vae_path, device, dtype,
                          tiling=str(cfg.get("vae_tiling", "auto")))

    logger.info("加载文本编码器...")
    # 族 opaque 文本栈不拆包；ad-hoc prompt 关缓存（cached_varlen 族 TE 常驻）
    text_stack = family.load_text(
        text_encoder_path, device, dtype,
        t5_tokenizer_path=t5_tokenizer_path or None,
        purpose="generate",
        cache_enabled=False,
    )

    model.eval()

    if not incremental:
        logger.info("full 模式：清空旧 reg 内容")
        clear_reg_dir(reg_dir)

    # 生成循环
    total = len(to_generate)
    actual_count = 0

    for idx, entry in enumerate(to_generate):
        seed = (base_seed + idx) if base_seed != 0 else random.randint(0, 2**31 - 1)
        torch.manual_seed(seed)
        random.seed(seed)

        subfolder = entry["subfolder"]
        reg_sub = (reg_dir / subfolder) if subfolder else reg_dir
        reg_sub.mkdir(parents=True, exist_ok=True)

        out_name = f"{entry['stem']}_ai_{seed}.png"
        out_path = reg_sub / out_name
        caption_path = _copy_caption_for_reg(entry["img"], out_path)
        if caption_path is None:
            logger.warning(f"[{idx + 1}/{total}] {entry['img'].name} 无 tag 文件，跳过")
            continue

        try:
            prompt = _rewrite_caption_for_prompt(caption_path, excluded_tags)
        except Exception as e:
            logger.warning(f"[{idx + 1}/{total}] {caption_path.name} 读取失败，跳过: {e}")
            caption_path.unlink(missing_ok=True)
            continue

        if not prompt:
            logger.warning(f"[{idx + 1}/{total}] {entry['img'].name} 过滤后无 tag，跳过")
            caption_path.unlink(missing_ok=True)
            continue

        logger.info(f"[{idx + 1}/{total}] {entry['img'].name} -> {out_name}")
        logger.info(f"  caption: {caption_path.name}")
        logger.info(f"  prompt: {prompt[:80]}")

        try:
            img = family.sample_image(
                model, vae, text_stack,
                prompt,
                height=height,
                width=width,
                steps=steps,
                cfg_scale=cfg_scale,
                negative_prompt=negative_prompt,
                sampler_name=sampler_name,
                scheduler=scheduler,
                device=device,
                dtype=dtype,
            )
            img.save(out_path)
            actual_count += 1
            logger.info(f"  已保存: {out_path}")
            if _update_monitor:
                _update_monitor(sample_path=str(out_path), step=idx + 1)
        except Exception as e:
            logger.error(f"  生成失败: {e}")
            if not out_path.exists():
                caption_path.unlink(missing_ok=True)

    _write_meta_final(reg_dir, entries, excluded_tags, incremental, actual_count)
    logger.info(f"完成: {actual_count}/{total} 张")


if __name__ == "__main__":
    main()
