"""共享 dependency helpers（PR-6 起从 server.py 抽出）。

跨 router 共用的 helper：拿 Supervisor 实例、版本 / 项目记录验证等。
未来会改 FastAPI `Depends(...)` 形式，本次先维持「直接调函数」风格保
零行为变更。
"""
from __future__ import annotations

from typing import Optional

from fastapi import HTTPException

from ..supervisor import Supervisor


def _supervisor() -> Supervisor:
    """从 app.state 取 Supervisor。lifespan startup 还没跑完时返 503。

    本 helper 内做 late import 避免 `api/app.py ↔ api/routers/* ↔ api/deps.py`
    三方循环——routers 在 app.py include 时还在初始化，此时 import api.app
    虽然拿得到 `app`（`app = FastAPI(...)` 已执行）但循环关系不健康。
    """
    from .app import app
    sup: Optional[Supervisor] = getattr(app.state, "supervisor", None)
    if sup is None:
        raise HTTPException(503, "supervisor not running")
    return sup


def _resolve_anima_model_paths() -> dict[str, str]:
    """解析 base 模型默认路径（先验生成 / 测试出图共用）。

    与 version_config 的 model 字段对齐。用户用别的 base 模型时，
    在 Settings → 模型 里改 selected_anima 影响这里的 anima 主权重路径。
    """
    from ..services.models import models_root
    root = models_root()
    return {
        "transformer_path": str(root / "diffusion_models" / "anima-base-v1.0.safetensors"),
        "vae_path": str(root / "vae" / "qwen_image_vae.safetensors"),
        "text_encoder_path": str(root / "text_encoders"),
        "t5_tokenizer_path": str(root / "t5_tokenizer"),
    }
