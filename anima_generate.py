#!/usr/bin/env python3
"""独立图片生成脚本 —— 复用 anima_train.sample_image() 推理链路。

用法：
    python anima_generate.py --config generate_config.json [--monitor-state-file state.json]

JSON 配置字段见 studio.schema.GenerateConfig。
"""
from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from pathlib import Path

import torch

# 复用 anima_train 的模型加载/采样函数（无副作用，main() 在 __name__=='__main__' 里）
import anima_train as _T

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("anima_generate")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Anima 独立图片生成")
    p.add_argument("--config", required=True, help="JSON 配置文件路径")
    p.add_argument("--monitor-state-file", default="", help="进度状态文件路径")
    return p.parse_args()


def _read_lora_meta(lora_path: str) -> dict:
    """从 safetensors metadata 读取 LoRA 训练参数。"""
    from safetensors import safe_open
    try:
        with safe_open(lora_path, framework="pt", device="cpu") as f:
            meta = f.metadata() or {}
        return json.loads(meta.get("ss_network_args", "{}"))
    except Exception:
        return {}


def load_loras(model, lora_configs: list[dict], device: str, dtype) -> None:
    """加载并合并多个 LoRA（按 scale 加权叠加权重）。"""
    valid = [c for c in lora_configs if c.get("path") and Path(c["path"]).exists()]
    if not valid:
        return

    from safetensors import safe_open

    merged: dict = {}
    first_meta: dict = {}
    for lora_cfg in valid:
        path = lora_cfg["path"]
        scale = float(lora_cfg.get("scale", 1.0))
        meta = _read_lora_meta(path)
        if not first_meta:
            first_meta = meta
        with safe_open(path, framework="pt", device="cpu") as f:
            for k in f.keys():
                t = f.get_tensor(k).to(device=device, dtype=dtype) * scale
                merged[k] = merged[k] + t if k in merged else t

    if not merged:
        return

    algo = first_meta.get("algo", "lokr")
    factor = int(first_meta.get("factor", 8))
    from utils.lycoris_adapter import AnimaLycorisAdapter
    injector = AnimaLycorisAdapter(algo=algo, rank=32, alpha=32.0, factor=factor)
    injector.inject(model)
    injector.load_state_dict(merged, strict=False)
    logger.info(f"已加载 {len(valid)} 个 LoRA")


def main() -> None:
    args = parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.exists():
        logger.error(f"配置文件不存在: {cfg_path}")
        sys.exit(1)

    with open(cfg_path, encoding="utf-8") as f:
        cfg = json.load(f)

    output_dir = Path(cfg.get("output_dir", "./generate_output"))
    sample_subdir: str = cfg.get("sample_subdir", "samples")
    sample_dir = output_dir / sample_subdir
    sample_dir.mkdir(parents=True, exist_ok=True)

    prompts: list[str] = cfg.get("prompts") or ["newest, safe, 1girl, masterpiece, best quality"]
    negative_prompt: str = cfg.get("negative_prompt", "")
    width: int = int(cfg.get("width", 1024))
    height: int = int(cfg.get("height", 1024))
    steps: int = int(cfg.get("steps", 25))
    cfg_scale: float = float(cfg.get("cfg_scale", 4.0))
    sampler_name: str = cfg.get("sampler_name", "er_sde")
    scheduler: str = cfg.get("scheduler", "simple")
    count: int = max(1, int(cfg.get("count", 1)))
    base_seed: int = int(cfg.get("seed", 0))
    lora_configs: list[dict] = cfg.get("lora_configs", [])
    mixed_precision: str = cfg.get("mixed_precision", "bf16")
    xformers: bool = bool(cfg.get("xformers", False))

    transformer_path: str = cfg["transformer_path"]
    vae_path: str = cfg["vae_path"]
    text_encoder_path: str = cfg["text_encoder_path"]
    t5_tokenizer_path: str = cfg.get("t5_tokenizer_path", "")

    # monitor state
    state_file = args.monitor_state_file or str(output_dir / "monitor_state.json")
    _update_monitor = None
    try:
        from train_monitor import set_state_file, update_monitor
        set_state_file(state_file)
        update_monitor(config={
            "type": "generate",
            "prompts": len(prompts),
            "count": count,
            "steps": steps,
            "cfg_scale": cfg_scale,
        })
        _update_monitor = update_monitor
    except Exception as e:
        logger.warning(f"monitor 初始化失败: {e}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if mixed_precision == "bf16" else torch.float32

    # 路径解析（相对路径相对于 repo root）
    repo_root = _T.find_diffusion_pipe_root()
    script_dir = Path(__file__).resolve().parent
    bases = [Path.cwd(), script_dir, repo_root]
    transformer_path = _T.resolve_path_best_effort(transformer_path, bases)
    vae_path = _T.resolve_path_best_effort(vae_path, bases)
    text_encoder_path = _T.resolve_path_best_effort(text_encoder_path, bases)
    if t5_tokenizer_path:
        t5_tokenizer_path = _T.resolve_path_best_effort(t5_tokenizer_path, bases)

    # 加载模型
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

    # 可选 LoRA（支持多个）
    load_loras(model, lora_configs, device, dtype)

    model.eval()

    # 生成循环
    total = count * len(prompts)
    logger.info(f"开始生成：{len(prompts)} 个 prompt × {count} 次 = {total} 张")

    img_idx = 0
    for pi, prompt in enumerate(prompts):
        for ci in range(count):
            seed = (base_seed + img_idx) if base_seed != 0 else random.randint(0, 2**31 - 1)
            torch.manual_seed(seed)
            random.seed(seed)

            logger.info(f"[{img_idx + 1}/{total}] seed={seed}  prompt={prompt[:60]}...")
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
                fname = f"gen_{img_idx:04d}_p{pi}_c{ci}_s{seed}.png"
                out_path = sample_dir / fname
                img.save(out_path)
                logger.info(f"已保存: {out_path}")
                if _update_monitor:
                    _update_monitor(sample_path=str(out_path), step=img_idx + 1)
            except Exception as e:
                logger.error(f"生成失败 [{img_idx + 1}/{total}]: {e}")

            img_idx += 1

    logger.info("生成完成")


if __name__ == "__main__":
    main()
