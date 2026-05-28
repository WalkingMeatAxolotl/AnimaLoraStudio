"""Studio 业务服务 — 0.11.0 起按主题分 11 子包（详 ADR-0008）。

每个子模块都是「纯函数 + dataclass」风格，由 worker 或服务端调用。
不依赖 db；db 操作在调用侧完成。

## 11 子包对照

| 子包 | 内容 |
|---|---|
| `tagging/`     | wd14 / cltagger / llm_tagger / joycaption / caption_format / caption_snapshot / onnx_tagger_base / tagger (factory) |
| `booru/`       | gelbooru / danbooru HTTP API (api.py) + 连接池 + token bucket 限速 (pool.py) + 项目级图片下载 (downloader.py) |
| `reg/`         | 正则数据集构建主流程 (builder.py 742 行) + 纯分析/评分函数 (analysis.py 424 行) + 后处理聚类裁剪 (postprocess.py) |
| `inference/`   | LoRA 元数据 + apply (inference_core.py) + 长驻 daemon (inference_daemon.py) + 测试出图 cache (generate_cache.py) + 超分 (upscaler.py) |
| `models/`      | 模型下载器 4 文件拆分：catalog / paths / sources / downloader（PR-3.8）|
| `preprocess/`  | preprocess 主流程 (core.py) + duplicate finder (duplicates.py) + manifest (manifest.py) |
| `projects/`    | 项目 CRUD (projects.py) + version (versions.py) + phase 状态机 (versions_phase.py) + curation (curation.py) + project_jobs (project_jobs.py) |
| `dataset/`     | dataset 扫描 + browse + thumb_cache + task_snapshot + presets_io |
| `presets/`     | 预设 fork / save-as flow（io.py + __init__.py 提供 fork_preset_for_version / save_version_config_as_preset）|
| `runtime/`     | 运行时 install 类：onnxruntime_setup / torch_setup / flash_attention_setup / xformers_setup / pending_install / updater |
| `data_io/`     | train.zip / bundle.zip 导入导出 (train_io.py) |

## shim 兼容（PR-3 起）

为保 `from studio.services.X import Y` 老路径兼容，老平铺位置留 sys.modules
别名 shim：`studio.services.wd14_tagger` → `studio.services.tagging.wd14_tagger`
等。0.11.1+ 会按子包逐批删 shim（详 0.11.0 ADR-0008 follow-ups）。
"""
