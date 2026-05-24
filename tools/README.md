# tools/

仓库根的一次性脚本、bootstrap helper、诊断和 release / migration 工具集。所有脚本均在 **仓库根目录** 跑（`python tools/xxx.py ...`），多数需要先激活 venv。

> 不属于这里：训练 / 推理代码（在 `runtime/`）、Studio 服务（在 `studio/`）、生产时调用的子进程（在 `studio/services/`）。

---

## Bootstrap helpers（被 studio.bat / studio.sh 调用，stdlib only）

### `check_requirements_changed.py`
启动期检测 `requirements.txt` 内容 hash 是否变了，决定是否补装新依赖。用 content hash 不用 mtime，避免 `git checkout` 后误判 stale。

```
python tools/check_requirements_changed.py             # 输出 stale / current / missing
python tools/check_requirements_changed.py --update-marker   # 同步成功后写新 hash
```

### `select_torch_index.py`
venv **首装**时按 `nvidia-smi` 检测的驱动版本输出对应的 PyTorch wheel index URL，让 caller 用对应的 CUDA wheel 装 torch（而不是 PyPI 默认 CPU 版）。

```
python tools/select_torch_index.py     # 检测到 → 输出 URL；否则静默 exit 0
```

驱动 → cu wheel 映射跟 `studio/services/torch_setup.py:_DRIVER_TO_BEST_CU` 双向同步。

---

## 模型 / 环境 setup

### `download_models.py`
下载 Anima 训练所需的全部模型 + tokenizer（CLI 薄壳，逻辑在 `studio.services.model_downloader`，跟 Studio 设置页 UI 共用）。

```
python tools/download_models.py
python tools/download_models.py --variant preview3-base
python tools/download_models.py --no-mirror              # 不走 ModelScope 镜像
python tools/download_models.py --skip-main --skip-vae
python tools/download_models.py --output /data/anima
```

### `install_flash_attn.py`
flash_attn prebuild wheel 安装 CLI（跟 Settings UI 共享 `studio.services.flash_attention_setup` 的 wheel 选择逻辑）。

```
python tools/install_flash_attn.py            # 自动选最优 wheel
python tools/install_flash_attn.py --url URL  # 手动指定
python tools/install_flash_attn.py --dry-run  # 只列环境 + 候选不真装
python tools/install_flash_attn.py --force    # 已装也重装
```

退出码：0 成功 / 1 安装失败 / 2 环境不支持。

### `validate_local_models.py`
离线验证本地模型可正常加载（设 `HF_HUB_OFFLINE=1` + `TRANSFORMERS_OFFLINE=1`，分别测 T5 tokenizer / Qwen tokenizer + model）。默认从 `tools/models/` 找权重。

```
python tools/validate_local_models.py
```

---

## 诊断 / benchmark

### `diagnose_onnx_gpu.py`
WD14 打标遇到 "CUDA EP 静默降级到 CPU" 警告时跑这个定位根因。在 studio 同一个 venv 里执行，把 stdout 全文贴到 PR / issue。

```
python tools/diagnose_onnx_gpu.py
```

### `bench_wd14.py`
WD14 打标性能诊断：分阶段计时（preprocess / session.run / postprocess）+ EP / preload / 模型 / 线程数自检。同时回答：CPU vs GPU 实测吞吐差几倍。

```
python tools/bench_wd14.py [<图目录>] [--n 10] [--model <hf_id>]
```

不传图目录时默认扫 `studio_data/projects/*/raw_*` 找最近一批图取前 N 张。日志同时打 stdout 与 `bench_wd14.log`。

### `bench_gelbooru.py`
三步定位 gelbooru 下载速度瓶颈：网络 / 上游限速 / Studio 代码。

```
python tools/bench_gelbooru.py
```

凭证自动从 `studio_data/secrets.json` 读；缺失时回退环境变量 `GELBOORU_USER_ID` / `GELBOORU_API_KEY`。日志同时打 stdout 与 `bench_gelbooru.log`。

---

## Release / schema 维护

### `bump_version.py`
`release_notes.yaml` 校验 + 版本号同步 + `CHANGELOG.md` 派生。详见 `docs/release-notes-spec.md`。**不创建 entries**，那是 agent 改 yaml 的事。

```
python tools/bump_version.py validate          # schema 校验整个 yaml
python tools/bump_version.py bump              # 读 yaml top version，同步 3 处版本文件
python tools/bump_version.py bump --version 0.6.1
python tools/bump_version.py render-changelog  # 只重写 CHANGELOG.md，不动版本号
python tools/bump_version.py verify-versions   # __init__.py / package.json / package-lock.json drift 检查 (CI 用)
```

---

## 一次性 migration

### `preset_toml_to_yaml.py`
救活 **2026-05-21 ~ 2026-05-24** 期间前端 `downloadCurrentPreset` bug 导出的"假 yaml 真 toml"预设文件。新版前端已改走 server `/api/presets/{name}/download` 端点直发原 yaml，本工具仅用于回收历史下载。

```
python tools/preset_toml_to_yaml.py broken.yaml          # 输出 broken.fixed.yaml
python tools/preset_toml_to_yaml.py --in-place x.yaml    # 覆盖 + 备份 .toml-bak
python tools/preset_toml_to_yaml.py --output o.yaml x.yaml
python tools/preset_toml_to_yaml.py --no-validate x.yaml # 跳 schema 校验
```

处理 3 类破损：多行 `{...}` → inline table、空值 key → drop、TOML → tomllib 解析 + TrainingConfig 校验 → yaml 落盘。py 3.11+ 走 stdlib `tomllib`，3.10 及更早需 `pip install tomli`。

---

## `spike/`

ADR 验证用的临时脚本（验证完会随 cleanup PR 删）。当前内容是 ADR 0006 暂停/恢复的信号链路 spike，独立 README 见 [`spike/README.md`](spike/README.md)。
