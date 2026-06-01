#!/usr/bin/env python3
"""先验生成 — base 模型对每张训练图的 tag 反向出对照图作正则集。

设计来自 DreamBooth prior preservation：训练损失同时见到「LoRA 学到的样子」和
「base 模型本来的样子」，让 LoRA 只学差异、不污染 base 概念。

**不带 LoRA** —— 出现 LoRA 反而把要保留的 prior 给覆盖了。

用法：
    python runtime/anima_reg_ai.py --config reg_ai_config.json [--monitor-state-file state.json]

逻辑：
  1. 扫 train 目录所有图 + caption
  2. 每张图 → tag 去除 excluded → ", " 拼成 prompt → 生成 1 张正则图
  3. 输出到 reg/{对应子文件夹}/{stem}_ai_{seed}.png（镜像 train 子目录结构）
  4. 同名 .txt 写入用过的 prompt
  5. reg/meta.json 写 generation_method="ai_base", api_source=""
     （与 booru 拉取共用 reg_builder.RegMeta schema，不再撞名重写）

incremental=True：跳过 reg 子文件夹中已有以 train_stem 开头的图（重启续跑用）。
"""
from __future__ import annotations

import argparse
import json
import logging
import random
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

# 复用 reg_builder.RegMeta（PR-9 commit 2 加了 generation_method 字段）
from studio.services.reg.builder import RegMeta, read_meta, write_meta  # noqa: E402

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


def _read_tags(img_path: Path) -> list[str]:
    """读图片旁边的 .txt caption，返回 raw tag 列表（不归一化）。"""
    txt = img_path.with_suffix(".txt")
    if not txt.exists():
        return []
    raw = txt.read_text(encoding="utf-8", errors="ignore").strip()
    return [t.strip() for t in raw.split(",") if t.strip()]


def _normalize(tag: str) -> str:
    return tag.lower().strip().replace(" ", "_")


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
    # 兼容老 cfg 的 xformers/flash_attn 双 bool（schema.RegAiConfig.attention_backend 默认 flash_attn）
    from studio.schema import migrate_legacy_attention
    cfg = migrate_legacy_attention(cfg)
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

    logger.info("加载 Transformer...")
    model = _T.load_anima_model(transformer_path, device, dtype, repo_root, flash_attn=use_flash)
    if use_xformers:
        _T.enable_xformers(model)

    logger.info("加载 VAE...")
    vae = _T.load_vae(vae_path, device, dtype, repo_root)

    logger.info("加载文本编码器...")
    qwen_model, qwen_tok, t5_tok = _T.load_text_encoders(
        text_encoder_path, t5_tokenizer_path or None, device, dtype,
    )

    model.eval()

    # 生成循环
    total = len(to_generate)
    actual_count = 0

    for idx, entry in enumerate(to_generate):
        tags = [_normalize(t) for t in entry["tags"]]
        tags = [t for t in tags if t and t not in excluded_tags]

        if not tags:
            logger.warning(f"[{idx + 1}/{total}] {entry['img'].name} 过滤后无 tag，跳过")
            continue

        prompt = ", ".join(tags)
        seed = (base_seed + idx) if base_seed != 0 else random.randint(0, 2**31 - 1)
        torch.manual_seed(seed)
        random.seed(seed)

        subfolder = entry["subfolder"]
        reg_sub = (reg_dir / subfolder) if subfolder else reg_dir
        reg_sub.mkdir(parents=True, exist_ok=True)

        out_name = f"{entry['stem']}_ai_{seed}.png"
        out_path = reg_sub / out_name

        logger.info(f"[{idx + 1}/{total}] {entry['img'].name} -> {out_name}")
        logger.info(f"  prompt: {prompt[:80]}")

        try:
            img = _T.sample_image(
                model, vae, qwen_model, qwen_tok, t5_tok,
                prompt=prompt,
                height=height,
                width=width,
                steps=steps,
                cfg_scale=cfg_scale,
                negative_prompt=negative_prompt or None,
                sampler_name=sampler_name,
                scheduler=scheduler,
                device=device,
                dtype=dtype,
            )
            img.save(out_path)
            out_path.with_suffix(".txt").write_text(prompt, encoding="utf-8")
            actual_count += 1
            logger.info(f"  已保存: {out_path}")
            if _update_monitor:
                _update_monitor(sample_path=str(out_path), step=idx + 1)
        except Exception as e:
            logger.error(f"  生成失败: {e}")

    _write_meta_final(reg_dir, entries, excluded_tags, incremental, actual_count)
    logger.info(f"完成: {actual_count}/{total} 张")


if __name__ == "__main__":
    main()
