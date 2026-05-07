#!/usr/bin/env python3
"""AI 正则图生成脚本 — 遍历 train 每张图的 tag，逐图生成对应的正则图。

用法：
    python anima_reg_ai.py --config reg_ai_config.json [--monitor-state-file state.json]

逻辑：
  1. 扫描 train_dir 所有图片及其 caption（.txt）
  2. 对每张图：tag 去除 excluded_tags → 剩余 tag 作正向提示词 → 生成 1 张正则图
  3. 正则图保存到 reg_dir/{对应子文件夹}（镜像 train 结构）
  4. 写入 reg/meta.json（api_source 记为 "ai_generated"）

incremental=true 时跳过 reg 子文件夹中已有对应文件名前缀的图（补足模式）。
"""
from __future__ import annotations

import argparse
import json
import logging
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import torch

import anima_train as _T

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("anima_reg_ai")

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif"}


# ---------------------------------------------------------------------------
# meta (镜像 reg_builder.RegMeta，避免导入 studio 包)
# ---------------------------------------------------------------------------


@dataclass
class RegMeta:
    generated_at: float
    based_on_version: str
    api_source: str
    target_count: int
    actual_count: int
    source_tags: list
    excluded_tags: list
    blacklist_tags: list
    failed_tags: list
    train_tag_distribution: dict
    auto_tagged: bool
    incremental_runs: int = 0
    postprocessed_at: Optional[float] = None
    postprocess_clusters: Optional[int] = None
    postprocess_method: Optional[str] = None
    postprocess_max_crop_ratio: Optional[float] = None


def _write_meta(reg_dir: Path, meta: RegMeta) -> None:
    p = reg_dir / "meta.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(asdict(meta), ensure_ascii=False, indent=2), encoding="utf-8")


def _read_meta(reg_dir: Path) -> Optional[RegMeta]:
    p = reg_dir / "meta.json"
    if not p.exists():
        return None
    try:
        return RegMeta(**json.loads(p.read_text(encoding="utf-8")))
    except Exception:
        return None


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _read_tags(img_path: Path) -> list[str]:
    """读图片旁边的 .txt caption，返回标准化 tag 列表。"""
    txt = img_path.with_suffix(".txt")
    if not txt.exists():
        return []
    raw = txt.read_text(encoding="utf-8", errors="ignore").strip()
    return [t.strip() for t in raw.split(",") if t.strip()]


def _normalize(tag: str) -> str:
    return tag.lower().strip().replace(" ", "_")


def _scan_train(train_dir: Path) -> list[dict]:
    """扫 train 目录，返回每张图的信息列表。

    返回元素：{"subfolder": str, "stem": str, "img": Path, "tags": list[str]}
    subfolder="" 表示在 train 根目录。
    """
    entries = []
    train_dir = train_dir.resolve()

    def _scan(folder: Path, subfolder: str) -> None:
        for f in sorted(folder.iterdir()):
            if f.is_file() and f.suffix.lower() in IMAGE_EXTS:
                entries.append({
                    "subfolder": subfolder,
                    "stem": f.stem,
                    "img": f,
                    "tags": _read_tags(f),
                })
            elif f.is_dir():
                sub = f.name if not subfolder else f"{subfolder}/{f.name}"
                _scan(f, sub)

    _scan(train_dir, "")
    return entries


def _already_has_reg(reg_sub: Path, train_stem: str) -> bool:
    """incremental 模式：reg 子文件夹里已有以 train_stem 开头的图则跳过。"""
    if not reg_sub.exists():
        return False
    for f in reg_sub.iterdir():
        if f.is_file() and f.stem.startswith(train_stem) and f.suffix.lower() in IMAGE_EXTS:
            return True
    return False


def _load_loras(model, lora_configs: list[dict], device: str, dtype) -> None:
    """加载并合并多个 LoRA（与 anima_generate.py 逻辑一致）。"""
    from safetensors import safe_open

    valid = [c for c in lora_configs if c.get("path") and Path(c["path"]).exists()]
    if not valid:
        return

    merged: dict = {}
    first_meta: dict = {}
    for cfg in valid:
        path, scale = cfg["path"], float(cfg.get("scale", 1.0))
        try:
            with safe_open(path, framework="pt", device="cpu") as f:
                meta = json.loads((f.metadata() or {}).get("ss_network_args", "{}"))
                if not first_meta:
                    first_meta = meta
                for k in f.keys():
                    t = f.get_tensor(k).to(device=device, dtype=dtype) * scale
                    merged[k] = merged[k] + t if k in merged else t
        except Exception as e:
            logger.warning(f"加载 LoRA 失败 {path}: {e}")

    if not merged:
        return

    algo = first_meta.get("algo", "lokr")
    factor = int(first_meta.get("factor", 8))
    from utils.lycoris_adapter import AnimaLycorisAdapter
    injector = AnimaLycorisAdapter(algo=algo, rank=32, alpha=32.0, factor=factor)
    injector.inject(model)
    injector.load_state_dict(merged, strict=False)
    logger.info(f"已加载 {len(valid)} 个 LoRA")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Anima AI 正则图生成")
    p.add_argument("--config", required=True)
    p.add_argument("--monitor-state-file", default="")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg_path = Path(args.config)
    if not cfg_path.exists():
        logger.error(f"配置文件不存在: {cfg_path}")
        sys.exit(1)

    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))

    train_dir = Path(cfg["train_dir"])
    reg_dir = Path(cfg["reg_dir"])
    excluded_tags: set[str] = {_normalize(t) for t in cfg.get("excluded_tags", [])}
    negative_prompt: str = cfg.get("negative_prompt", "")
    width: int = int(cfg.get("width", 1024))
    height: int = int(cfg.get("height", 1024))
    steps: int = int(cfg.get("steps", 25))
    cfg_scale: float = float(cfg.get("cfg_scale", 4.0))
    sampler_name: str = cfg.get("sampler_name", "er_sde")
    scheduler: str = cfg.get("scheduler", "simple")
    base_seed: int = int(cfg.get("seed", 0))
    lora_configs: list[dict] = cfg.get("lora_configs", [])
    incremental: bool = bool(cfg.get("incremental", False))
    mixed_precision: str = cfg.get("mixed_precision", "bf16")
    xformers: bool = bool(cfg.get("xformers", False))

    transformer_path: str = cfg["transformer_path"]
    vae_path: str = cfg["vae_path"]
    text_encoder_path: str = cfg["text_encoder_path"]
    t5_tokenizer_path: str = cfg.get("t5_tokenizer_path", "")

    # monitor
    state_file = args.monitor_state_file
    _update_monitor = None
    try:
        from train_monitor import set_state_file, update_monitor
        set_state_file(state_file)
        update_monitor(config={"type": "reg_ai"})
        _update_monitor = update_monitor
    except Exception as e:
        logger.warning(f"monitor 初始化失败: {e}")

    # 扫 train
    if not train_dir.exists():
        logger.error(f"train 目录不存在: {train_dir}")
        sys.exit(1)

    entries = _scan_train(train_dir)
    if not entries:
        logger.error("train 目录没有任何图片")
        sys.exit(1)

    logger.info(f"train 共 {len(entries)} 张图")

    # incremental 过滤
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

    # 加载模型
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if mixed_precision == "bf16" else torch.float32

    repo_root = _T.find_diffusion_pipe_root()
    script_dir = Path(__file__).resolve().parent
    bases = [Path.cwd(), script_dir, repo_root]
    transformer_path = _T.resolve_path_best_effort(transformer_path, bases)
    vae_path = _T.resolve_path_best_effort(vae_path, bases)
    text_encoder_path = _T.resolve_path_best_effort(text_encoder_path, bases)
    if t5_tokenizer_path:
        t5_tokenizer_path = _T.resolve_path_best_effort(t5_tokenizer_path, bases)

    logger.info("加载 Transformer...")
    model = _T.load_anima_model(transformer_path, device, dtype, repo_root)
    if xformers:
        _T.enable_xformers(model)

    logger.info("加载 VAE...")
    vae = _T.load_vae(vae_path, device, dtype, repo_root)

    logger.info("加载文本编码器...")
    qwen_model, qwen_tok, t5_tok = _T.load_text_encoders(
        text_encoder_path, t5_tokenizer_path or None, device, dtype
    )

    _load_loras(model, lora_configs, device, dtype)
    model.eval()

    # 生成
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

        logger.info(f"[{idx + 1}/{total}] {entry['img'].name} → {out_name}")
        logger.info(f"  prompt: {prompt[:80]}...")

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
            actual_count += 1
            logger.info(f"  已保存: {out_path}")

            if _update_monitor:
                _update_monitor(sample_path=str(out_path), step=idx + 1)

        except Exception as e:
            logger.error(f"  生成失败: {e}")

    _write_meta_final(reg_dir, entries, excluded_tags, incremental, actual_count)
    logger.info(f"完成：{actual_count}/{total} 张")


def _write_meta_final(
    reg_dir: Path,
    entries: list[dict],
    excluded_tags: set[str],
    incremental: bool,
    actual_count: int,
) -> None:
    prior = _read_meta(reg_dir)
    runs = (prior.incremental_runs + 1) if (incremental and prior) else 0

    from collections import Counter
    tag_dist: Counter = Counter()
    for e in entries:
        tag_dist.update(e["tags"])

    meta = RegMeta(
        generated_at=time.time(),
        based_on_version="",
        api_source="ai_generated",
        target_count=len(entries),
        actual_count=actual_count + (prior.actual_count if incremental and prior else 0),
        source_tags=[],
        excluded_tags=sorted(excluded_tags),
        blacklist_tags=[],
        failed_tags=[],
        train_tag_distribution=dict(tag_dist.most_common(50)),
        auto_tagged=False,
        incremental_runs=runs,
    )
    _write_meta(reg_dir, meta)


if __name__ == "__main__":
    main()
