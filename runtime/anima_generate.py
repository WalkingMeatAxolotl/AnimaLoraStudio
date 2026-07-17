#!/usr/bin/env python3
"""测试出图 — 独立运行推理（CLI 用法，不再被 Studio server 调）。

用法：
    python runtime/anima_generate.py --config generate_config.json [--monitor-state-file state.json]

JSON 配置字段见 studio.schema.GenerateConfig。

历史 / 当前位置：
  - 早期 server 通过 supervisor spawn 这个脚本作为 generate task 的 worker；
    每次出图都要 30-60s 重 load 模型。
  - PR Phase 2（commit 9+）改成常驻 inference_daemon（runtime/anima_daemon.py）+
    模型跨 task 复用 + 图不落盘走内存 cache。Server 不再 spawn 这个脚本。
  - 本文件保留作 CLI 用法：用户在命令行直跑出图，写盘到 cfg.output_dir
    （用户指定路径，是真实持久化）。

关键实现：
  - 多 LoRA 加载走 studio.services.inference_core.apply_loras —— 每份 LoRA 独立
    inject 一份 AnimaLycorisAdapter，rank/alpha 从 ss_network_args 读，用
    multiplier=scale 控制贡献权重（修 PR #17 硬编码 rank=32 + LoKr 子矩阵
    直加的出错图问题）。
  - 进度通过 train_monitor 推 SSE，前端按 sample_path 拉单图显示。
"""
from __future__ import annotations

import argparse
import json
import logging
import random
import sys
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

from studio.domain.comfy_parity import force_comfy_parity_runtime_config  # noqa: E402
from studio.services.inference.core import LoRASpec, apply_loras  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("anima_generate")


def _torch_dtype_from_precision(value: str | None) -> torch.dtype:
    normalized = str(value or "fp32").lower().strip()
    if normalized in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if normalized in {"fp16", "float16", "half"}:
        return torch.float16
    return torch.float32


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Anima 测试出图")
    p.add_argument("--config", required=True, help="JSON 配置文件路径")
    p.add_argument("--monitor-state-file", default="", help="进度状态文件路径")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.exists():
        logger.error(f"配置文件不存在: {cfg_path}")
        sys.exit(1)

    with open(cfg_path, encoding="utf-8") as f:
        cfg = json.load(f)
    cfg = force_comfy_parity_runtime_config(
        cfg,
        force_exact_ksampler_backend=False,
    )

    output_dir = Path(cfg.get("output_dir", "./generate_output"))
    output_dir.mkdir(parents=True, exist_ok=True)

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
    distilled: bool = bool(cfg.get("distilled", False))
    lora_configs: list[dict] = cfg.get("lora_configs", [])
    mixed_precision: str = cfg.get("mixed_precision", "bf16")
    vae_precision: str = cfg.get("vae_precision", mixed_precision)
    vram_policy: str = str(cfg.get("vram_policy") or "auto")
    text_encoder_backend: str = cfg.get("text_encoder_backend", "hf")
    t5_tokenizer_backend: str = cfg.get("t5_tokenizer_backend", "slow")
    backend: str = cfg.get("attention_backend", "none")
    use_flash = (backend == "flash_attn")
    use_xformers = (backend == "xformers")

    transformer_path: str = cfg["transformer_path"]
    vae_path: str = cfg["vae_path"]
    text_encoder_path: str = cfg["text_encoder_path"]
    t5_tokenizer_path: str = cfg.get("t5_tokenizer_path", "")

    # monitor
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
    dtype = _torch_dtype_from_precision(mixed_precision)
    vae_dtype = _torch_dtype_from_precision(vae_precision)

    # 路径解析
    repo_root = _T.find_diffusion_pipe_root()
    bases = [Path.cwd(), _THIS_DIR, repo_root]
    transformer_path = _T.resolve_path_best_effort(transformer_path, bases)
    vae_path = _T.resolve_path_best_effort(vae_path, bases)
    text_encoder_path = _T.resolve_path_best_effort(text_encoder_path, bases)
    if t5_tokenizer_path:
        t5_tokenizer_path = _T.resolve_path_best_effort(t5_tokenizer_path, bases)

    family = _T.resolve_family(cfg)  # D8'：旁路调用方经 family 派发
    logger.info("加载 Transformer...")
    model = family.load_dit(
        transformer_path, device, dtype,
        attention_backend=("flash_attn" if use_flash else "none"), repo_root=repo_root,
        purpose="generate",
    )
    if use_xformers and not _T.enable_xformers(model):
        raise RuntimeError(
            "Exact ComfyUI KSampler parity is guaranteed only with xformers, "
            "but xformers could not be enabled"
        )

    logger.info("加载 VAE...")
    vae = family.load_vae(vae_path, device, vae_dtype,
                          tiling=str(cfg.get("vae_tiling", "auto")))

    logger.info("加载文本编码器...")
    # 族 opaque 文本栈不拆包；ad-hoc prompt 关缓存（cached_varlen 族 TE 常驻）
    text_stack = family.load_text(
        text_encoder_path, device, dtype,
        t5_tokenizer_path=t5_tokenizer_path or None,
        comfy_qwen=text_encoder_backend == "comfy_qwen3",
        t5_fast=t5_tokenizer_backend == "fast",
        purpose="generate",
        cache_enabled=False,
        te_quantize=str(cfg.get("te_precision") or "fp16") == "fp8",
    )

    # 多 LoRA：每份独立 inject + multiplier=scale。adapters 必须保持引用，否则
    # 被 GC 后 forward hook 失效（lycoris 通过 closure 持有 network）。
    specs = [
        LoRASpec(path=str(lc.get("path", "")), scale=float(lc.get("scale", 1.0)))
        for lc in lora_configs
    ]
    _adapters = apply_loras(model, specs, device, torch.float32,  # noqa: F841 — 保持引用
                            family_id=family.spec.family_id)

    model.eval()

    # krea2 任务级 prompt 预编码（daemon _precache_prompts_and_offload 同款）：
    # prompts 集合封闭（XY 时 schema 保证单条），预编码后卸 TE，采样期零占用。
    # anima 文本栈（tuple）无此 API 自然跳过；performance 档不卸。
    precache = getattr(text_stack, "precache_online_prompts", None)
    if callable(precache):
        try:
            encoded = precache([*[str(p) for p in prompts], negative_prompt])
        except Exception:
            logger.exception("prompt 预编码失败；退回逐图惰性编码")
        else:
            if vram_policy != "performance":
                offload = getattr(text_stack, "offload_model", None)
                if callable(offload):
                    offload()
            if encoded:
                logger.info("krea2 预编码 %d 条 prompt；采样期 TE 不占显存", encoded)

    # XY 矩阵分支（schema 已校验：xy_matrix 设值时 prompts 单条 + count=1）
    xy_matrix = cfg.get("xy_matrix")
    if xy_matrix is not None:
        _run_xy_matrix(
            xy_matrix=xy_matrix,
            base_specs=specs,
            adapters=_adapters,
            prompt=prompts[0],
            negative_prompt=negative_prompt,
            base_seed=base_seed,
            base_steps=steps,
            base_cfg_scale=cfg_scale,
            base_sampler=sampler_name,
            scheduler=scheduler,
            distilled=distilled,
            height=height,
            width=width,
            family=family, model=model, vae=vae, text=text_stack,
            device=device, dtype=dtype,
            output_dir=output_dir,
            update_monitor=_update_monitor,
            vram_policy=vram_policy,
        )
        logger.info("XY 矩阵生成完成")
        return

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
                    distilled=distilled,
                    device=device,
                    dtype=dtype,
                    seed=seed,
                    vram_policy=vram_policy,
                )
                fname = f"gen_{img_idx:04d}_p{pi}_c{ci}_s{seed}.png"
                out_path = output_dir / fname
                img.save(out_path)
                logger.info(f"已保存: {out_path}")
                if _update_monitor:
                    _update_monitor(sample_path=str(out_path), step=img_idx + 1)
            except Exception as e:
                logger.error(f"生成失败 [{img_idx + 1}/{total}]: {e}")

            img_idx += 1

    logger.info("生成完成")


# ---------------------------------------------------------------------------
# XY 矩阵实现 —— 单 task 内循环全图，省去 N 次 model load 摊销成本
# ---------------------------------------------------------------------------


def _set_lora_multiplier(adapter, scale: float) -> None:
    """In-place 改一份 adapter 的 multiplier，不需要 re-inject。

    与 inference_core.apply_loras 内的设值路径一致：network.multiplier 是
    forward 内取的全局倍率；per-lora.multiplier 兜底（lycoris 不同版本取值
    路径有差异）。
    """
    if adapter.network is None:
        return
    adapter.network.multiplier = float(scale)
    for lora in getattr(adapter.network, "loras", []):
        if hasattr(lora, "multiplier"):
            lora.multiplier = float(scale)


def _apply_axis(
    axis: dict,
    value,
    *,
    cur_steps: int, cur_cfg_scale: float, cur_seed: int, cur_sampler: str,
    base_specs, adapters,
) -> tuple[int, float, int, str]:
    """对 axis_type 派生的字段做更新；lora_scale 直接 mutate 所有 adapter。

    lora_ckpt 不在这里 —— 它要 reinject，由 _run_xy_matrix 单独走
    apply_loras 重新加载路径。

    返回 (steps, cfg_scale, seed, sampler) 4 元组（不变量直接透传）。
    """
    axis_type = axis["axis"]
    if axis_type == "steps":
        cur_steps = int(value)
    elif axis_type == "cfg_scale":
        cur_cfg_scale = float(value)
    elif axis_type == "seed":
        cur_seed = int(value)
    elif axis_type == "sampler_name":
        cur_sampler = str(value)
    elif axis_type == "lora_scale":
        # 全局轴：所有 LoRA 的 multiplier 都设成同一个 cell 值
        for ad in adapters:
            _set_lora_multiplier(ad, float(value))
    return cur_steps, cur_cfg_scale, cur_seed, cur_sampler


def _run_xy_matrix(
    *,
    xy_matrix: dict,
    base_specs: list,
    adapters: list,
    prompt: str,
    negative_prompt: str,
    base_seed: int,
    base_steps: int,
    base_cfg_scale: float,
    base_sampler: str,
    scheduler: str,
    height: int,
    width: int,
    family, model, vae, text,
    device: str, dtype,
    output_dir,
    update_monitor,
    distilled: bool = False,
    vram_policy: str = "auto",
) -> None:
    """循环 (yi, xi) 出 N×M 张图。

    设计：
      - 每个 cell 从 base_* 派生本次参数（防上次 cell 修改泄漏到下次）；
        lora_scale 通过 mutate adapter.multiplier 实现，每次 cell 进入前必须
        把所有 LoRA 的 multiplier 重置回 base_specs[i].scale。
      - 文件名 `xy_x{xi:02d}_y{yi:02d}_s{seed}.png`，前端按 (yi, xi) 排 grid。
      - update_monitor 推 sample_path + xy 元数据；前端拿 xy={xi,yi,xv,yv}
        渲染 cell 标签 + 排序。
      - base_seed=0 → 随机一次后所有 cell 共享（XY 仅看轴效应）；axis=seed
        时按 cell 值覆盖。
    """
    x_spec = xy_matrix["x"]
    y_spec = xy_matrix.get("y")
    x_values = x_spec["values"]
    y_values = y_spec["values"] if y_spec else [None]

    # lora_ckpt 轴需要 cell 间 detach + reinject LoRA，CLI runner 不接入
    # CACHE.apply_loras 那套（daemon 才有），所以这里直接拒绝。生产路径走
    # runtime/anima_daemon.py:_run_xy 已支持。
    if x_spec.get("axis") == "lora_ckpt" or (y_spec and y_spec.get("axis") == "lora_ckpt"):
        raise NotImplementedError(
            "lora_ckpt 轴需走 daemon path (runtime/anima_daemon.py)；"
            "CLI runner anima_generate.py 不支持热切换 LoRA 文件"
        )

    # fp8 底模的 LoRA 是 merge 进权重的（无常驻 network），lora_scale 轴
    # 需要逐格 detach + 重 merge——daemon 的 CACHE.apply_loras 路径已支持
    # （_cell_lora_configs）；CLI runner 无缓存管理，与 lora_ckpt 轴同款
    # 不接入。
    if x_spec.get("axis") == "lora_scale" or (y_spec and y_spec.get("axis") == "lora_scale"):
        from training.families.krea2.quant_fp8 import model_has_fp8_layers

        if model_has_fp8_layers(model):
            raise NotImplementedError(
                "fp8 底模的 LoRA 强度轴需逐格重新合并，走 daemon path "
                "(runtime/anima_daemon.py)；CLI runner 请改用 bf16 底模。"
            )

    if base_seed == 0:
        base_seed = random.randint(0, 2**31 - 1)
        logger.info(f"XY 共享种子（cfg.seed=0 随机化）: {base_seed}")

    base_scales = [float(s.scale) for s in base_specs]
    total = len(x_values) * len(y_values)
    logger.info(f"开始 XY 生成：{len(x_values)}×{len(y_values)} = {total} 张")

    img_idx = 0
    for yi, yv in enumerate(y_values):
        for xi, xv in enumerate(x_values):
            # 重置每个 LoRA 到 base scale，避免上次 cell 的 lora_scale 改动遗留
            for i, s in enumerate(base_scales):
                if i < len(adapters):
                    _set_lora_multiplier(adapters[i], s)

            cur_steps = base_steps
            cur_cfg_scale = base_cfg_scale
            cur_seed = base_seed
            cur_sampler = base_sampler

            cur_steps, cur_cfg_scale, cur_seed, cur_sampler = _apply_axis(
                x_spec, xv,
                cur_steps=cur_steps, cur_cfg_scale=cur_cfg_scale,
                cur_seed=cur_seed, cur_sampler=cur_sampler,
                base_specs=base_specs, adapters=adapters,
            )
            if y_spec is not None and yv is not None:
                cur_steps, cur_cfg_scale, cur_seed, cur_sampler = _apply_axis(
                    y_spec, yv,
                    cur_steps=cur_steps, cur_cfg_scale=cur_cfg_scale,
                    cur_seed=cur_seed, cur_sampler=cur_sampler,
                    base_specs=base_specs, adapters=adapters,
                )

            torch.manual_seed(cur_seed)
            random.seed(cur_seed)

            logger.info(
                f"XY [{xi},{yi}] x={xv} y={yv} "
                f"steps={cur_steps} cfg={cur_cfg_scale} seed={cur_seed} sampler={cur_sampler}"
            )
            try:
                img = family.sample_image(
                    model, vae, text,
                    prompt,
                    height=height,
                    width=width,
                    steps=cur_steps,
                    cfg_scale=cur_cfg_scale,
                    negative_prompt=negative_prompt,
                    sampler_name=cur_sampler,
                    scheduler=scheduler,
                    distilled=distilled,
                    device=device,
                    dtype=dtype,
                    seed=cur_seed,
                    vram_policy=vram_policy,
                )
                fname = f"xy_x{xi:02d}_y{yi:02d}_s{cur_seed}.png"
                out_path = output_dir / fname
                img.save(out_path)
                logger.info(f"已保存: {out_path}")
                if update_monitor:
                    update_monitor(
                        sample_path=str(out_path),
                        step=img_idx + 1,
                        xy={"xi": xi, "yi": yi, "xv": xv, "yv": yv},
                    )
            except Exception as e:
                logger.error(f"XY [{xi},{yi}] 失败: {e}")

            img_idx += 1


if __name__ == "__main__":
    main()
