# 项目结构

整仓顶层目录布局。Studio 内部模块结构见 [`studio/README.md`](../../studio/README.md)，跨步骤架构总览见 [`studio-pipeline.md`](studio-pipeline.md)。

```
AnimaLoraStudio/
├── runtime/                       # 训练/推理运行时核心（多模型族；独立进程，Studio 通过 subprocess 拉起，也可单独 CLI 跑）
│   ├── anima_train.py             # 训练入口（两族共用）
│   ├── training/                  # 训练栈子包：context / phases / text cache / loop / sample_runner / sysmem（显存/内存编排）
│   │   ├── families/              # 模型族 registry：spec / protocol / latent_spaces + anima/ + krea2/（loader / 文本栈 / 采样 / preset 按族自治）
│   │   ├── adapters/              # plugin: lokr / loha / lora / ortho / tlora
│   │   ├── optimizers/            # plugin: adamw / automagic / came / lion / prodigy / prodigy_plus_schedulefree / soap / soap_sf
│   │   ├── schedulers/            # plugin: cosine / cosine_with_restart / cosine_with_warmup / none
│   │   ├── inference_samplers/    # plugin: er_sde / dpmpp_3m_sde 等
│   │   ├── timestep_samplers/     # plugin: baseline / infonoise / krea2_shift
│   │   └── phases/                # bootstrap / models / dataset / text_cache / optimizer / resume / finalize
│   ├── anima_generate.py          # 出图：单图 / XY 矩阵（按族派发）
│   ├── anima_daemon.py            # 推理 daemon：常驻 GPU 加载 LoRA 和底模（按族派发）
│   ├── anima_reg_ai.py            # AI 先验生成：无 LoRA 直接用底模出 reg 集
│   └── train_monitor.py           # 训练状态写入器
├── studio/                        # AnimaStudio Web 工作台（FastAPI + React）— 4 层架构（ADR 0008）
│   ├── api/                       # HTTP 表面：FastAPI app + router + schemas + deps + exception_handlers
│   ├── services/                  # 业务服务 11 子包：tagging / booru / reg / inference / models /
│   │                              #   preprocess / projects / dataset / presets / runtime / data_io
│   ├── domain/                    # pydantic 模型：TrainingConfig / LoRA / XY / Generate / RegAi + migrations
│   ├── infrastructure/            # 路径 / 数据库 / event bus / secrets / 日志 / argparse 桥接 / migrations
│   ├── supervisor/                # 任务调度守护线程
│   ├── workers/                   # 后台子进程入口（download / tag / reg_build / preprocess）
│   ├── server.py                  # 兼容 shim，re-export `app` / `main`（真实入口在 api/app.py / api/main.py）
│   └── web/                       # React + Vite 前端
├── tools/                         # 用户 CLI / 启动期 setup helper（见 tools/README.md）
├── utils/                         # anima_train 共享 utility（model loader / optimizer / lycoris_adapter / ...）
├── modeling/                      # 模型架构定义（tracked）：按族组织的结构定义
│   ├── anima/                     # Anima 族（anima_modeling.py + cosmos_predict2_modeling.py，基于 ComfyUI / vendored diffusion-pipe 子集）
│   ├── krea2/                     # Krea 2 族（krea2_modeling.py，single-stream MMDiT，基于 ComfyUI / musubi-tuner）
│   └── wan/vae2_1.py              # Wan2.1 VAE 实现（跨族共享）
├── docs/                          # user-guide / architecture / adr / design / todo / announcements（见 docs/README.md）
└── models/                        # 下载的权重 / tokenizer 数据落点（gitignored、按需创建）
    ├── diffusion_models/          # 用户下载的主模型（Anima；Krea 2 Raw / Turbo 的 bf16 / fp8 单文件）
    ├── vae/                       # 用户下载的 VAE 权重（两族共享）
    ├── text_encoders/             # 文本编码器 + tokenizer（Anima 的 Qwen3-0.6B；Krea 2 的 Qwen3-VL-4B bf16 目录 / fp8 单文件目录）
    ├── t5_tokenizer/              # T5 tokenizer 文件（下载）
    ├── wd14/                      # WD14 ONNX 模型（HF 自动下载）
    └── taeflux/                   # TAEFlux 中间步预览权重
```

**依赖方向单向**：`modeling → utils → runtime → studio → tools`，不要反向 import。（`models/` 是纯数据目录，不含代码、不在依赖链里。）

## 运行时数据（gitignored）

- `studio_data/` — SQLite + 用户 preset
- `studio_data/tasks/{id}/` — 每个训练 task 的 config snapshot + monitor state + 采样图 + run.log（删 version 不丢历史）
- `studio_data/projects/{id}-{slug}/versions/{label}/output/` — 训练产物 LoRA
- `studio_data/projects/{id}-{slug}/versions/{label}/reg/` — 正则集（多 task 复用）
- `models/diffusion_models/`、`models/vae/`、`models/wd14/` — 大权重文件
