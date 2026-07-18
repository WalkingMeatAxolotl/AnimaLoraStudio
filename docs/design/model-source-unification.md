# 模型来源管理统一 —— 候选列表 + 双添加入口

> 状态：设计定稿（2026-07-18，D1-D6 全部拍板）。
> 本文回答：**预处理放大器自定义下载 / WD14 候选编辑 / CLTagger 自定义 repo / 训练本地模型 / 评估指标自定义 repo 这五处同质功能，如何统一成一个心智模型。**

---

## 1. 现状统计：五处"模型来源管理"

五处功能的本质相同——**一个候选列表（内置预设 + 用户添加）+ 一个当前选中**，差异只在候选从哪来。现状是 4 种添加入口、3 种删除语义、2 条保存链路：

| | 添加入口 UI | 用户填什么 | 持久化 | 下载端点 | 删除语义 |
|---|---|---|---|---|---|
| WD14 候选（`modelCards.tsx` WD14ModelCard） | 折叠「候选编辑」→ ModelIdsEditor chip 列表 | HF repo id | `secrets.wd14.model_ids[]` | 通用 `/api/models/download` | × 移出候选 + 🗑 删文件，两层并存 |
| CLTagger 自定义 repo（CLTaggerModelCard） | 折叠「自定义 repo」→ 单文本框改值 | HF repo id | `secrets.cltagger.model_id`（单值） | 通用端点，variant=版本 label | 🗑 删文件；旧 repo 无入口回退 |
| Eval 指标 ×3（EvalMetricModelCard） | 折叠「自定义 repo」→ 单文本框改值 | HF repo id | `secrets.eval_metrics.*_model_name`（单值） | 通用端点 | 🗑 删文件；改值即丢旧值 |
| 放大器自定义下载（UpscalerSection） | **常驻独立卡片**：source + repo + filename 表单 | repo id + 文件名 | **无**——落盘后纯扫盘发现 | **独立端点** `/api/upscalers/download_custom` | 🗑 删文件；custom 不能重下 |
| 训练本地模型（ModelsSection） | 列表底部按钮 → PathPicker | 本地 .safetensors 路径 | `secrets.models.custom.{family}[]` | **独立端点** `/api/models/{family}/custom`（仅登记） | 🗑 **只注销引用不删文件** |

已统一的部分（保持）：列表行骨架（radio + accent 高亮 + ModelStatusBadge + DownloadButton）、ModelGroupCard 外壳、SourceSelect、`catalog.downloads` + SSE 状态流。

## 2. 统一行为规范

每处 = **候选列表 + 底部两个添加入口**（「+ 添加下载」「+ 添加本地文件」）。候选三类，动作矩阵：

| 候选类型 | 来源 | 未下载 | 已下载 | 移除（移出列表，不动磁盘） |
|---|---|---|---|---|
| 内置 preset | 代码写死 | [下载] | [🗑 删除文件] | 无（内置不可移除，保护默认） |
| 用户添加 · 下载型 | repo id（+ 单文件资产需 filename） | [下载] + [× 移除] | [🗑 删除文件] + [× 移除] | 有 |
| 用户添加 · 本地文件 | PathPicker 登记绝对路径 | —（缺失时显示「文件缺失」） | — | **只有 [× 移除]**，永不删用户文件 |

- **选中值 = 预设标识或绝对路径**。现有先例：`models.selected[family]` 已是 variant key 或 custom path 二选一；推广到五处（transformers `from_pretrained` 等加载器天然吃本地路径）。
- **移除当前选中的候选 → 回退默认**（训练模型现状逻辑，推广）。
- **删除文件走 confirm 且文案写明大小**；移除不 confirm（无损操作）。
- 放大器扫盘保留为兜底：`upscalers/` 目录内扫出的未登记文件显示为特殊候选，动作只有 [🗑 删除文件]（「移除」对扫盘项无意义，下次扫又出现）。

### 校验（D3：从简，运行时报错兜底）

| | 添加下载 | 添加本地文件 |
|---|---|---|
| 规则 | repo id 非空 + `owner/name` 正则；filename 后缀白名单（`.pth` / `.safetensors` 等） | 路径存在 + 后缀白名单；目录型资产查关键文件（wd14 查 onnx+csv，eval clip/dino 查 config.json） |
| 不做 | 网络探测（repo 是否存在等下载时报错） | 权重内容校验（加载时报错） |

## 3. 数据模型（D5：完全统一，两条纪律）

新增单一字段吃下全部候选：

```
secrets.model_sources: dict[domain, list[SourceCandidate]]

domain ∈ { wd14, cltagger, eval_clip, eval_dino, eval_ccip, upscaler, anima, krea2, ... }

SourceCandidate:
  kind: "download" | "local"
  repo: str = ""            # download 型：HF/MS repo id
  filename: str = ""        # download 型单文件资产（upscaler / 主模型）
  path: str = ""            # local 型：绝对路径（文件或目录）
  extra: dict[str, str]     # 域特有键（cltagger: model_path / tag_mapping_path）
```

**两条兼容纪律**（把完全统一的风险压到最低的关键）：

1. **当前选中值字段一律不动**：`wd14.model_id`、`cltagger.model_id/model_path/tag_mapping_path`、`eval_metrics.*_model_name`、`models.selected`、`models.selected_upscaler` 保持原名原语义。运行时消费方（tagger worker / eval jobs / 训练路径解析 / 放大器加载 / CLI / runtime 子进程）**只读这些字段，零改动**（2026-07-18 逐点核对：候选列表的唯一读者是 `catalog.py` 与 UI）。
2. **被迁移的旧字段用 computed_field 保留写盘**（先例：`selected_anima` / `custom_anima_paths`）：`wd14.model_ids`、`models.custom` 迁入 `model_sources` 后继续以 computed 键 dump 回 secrets.json，回滚版本可读。

注意：PUT merge 时 merge base 必须剥掉 computed 兼容键（先例与坑记录见 `infrastructure/secrets.py` merge 注释），否则过期 legacy 值覆盖新写入。WD14 的 `model_id ∈ model_ids` 不变量 validator 升级为 Secrets 级（跨 `model_sources` 与选中值字段）。

## 4. 兼容性风险评估（老用户）

| 层 | 风险 | 依据 |
|---|---|---|
| 历史训练 config | **零** | 4 个模型路径是创建 version 时的绝对路径快照，「已存在 version 不动」，不经过 secrets |
| 运行时（训练 / 打标 / 评估 / 放大 / generate daemon） | **零** | 纪律 1：只读的选中值字段不动 |
| 已下载磁盘文件 | **零** | models_root 目录布局不动 |
| 读老写新迁移 | 低 | before-validator + 单测；库内已有 4 个同款先例（`_migrate_legacy_schema` / `_migrate_legacy_model_fields` / `_accept_legacy_prompt` / WD14 不变量） |
| 版本回滚 | 低 | 纪律 2：选中值 + wd14 候选 + 训练本地模型经 computed 键无损回读；**唯一丢失面**=新结构独有数据（eval 候选列表 / upscaler 持久化候选 / cltagger 用户候选）在旧 UI 暂不可见——文件仍在盘上，升级回来即恢复 |
| 老前端 dist 缓存窗口 | 低 | 老前端 PUT 旧键 → before-validator 迁移兜住；与现有硬刷新已知问题同口径 |

**结论：完全统一可行，风险小。** 前提是严格执行两条纪律。

## 5. 五处落地映射

| | 下载型添加 | 本地型添加 | 迁移 |
|---|---|---|---|
| WD14 | 现状 `model_ids[]` 语义 → `model_sources["wd14"]`；「候选编辑」编辑器退役 | 新增：选目录（查 model.onnx + selected_tags.csv） | 非默认 model_ids 项 → download 候选；`model_ids` computed 键保留 |
| Eval ×3 | 单值升级为候选列表 | 新增：选目录 | 旧单值 ≠ 默认时生成一条 download 候选；选中值字段不动 |
| 放大器 | 「自定义下载」独立卡退役 → 底部入口（repo + filename）；**候选持久化**（修复删后不能重下）；独立 source 下拉删除，跟全局 `download_sources` | 新增：选 .pth/.safetensors 登记路径 | 扫盘 custom 不迁移（继续扫盘兜底）；新下载起记录候选 |
| 训练主模型 | **新增**：repo + filename（第三方微调主模型） | 现状 `custom[]` → `model_sources[family]` local 候选；行内 🗑 改 ×（语义修正） | `custom` computed 键保留 |
| CLTagger | 用户候选 = (repo, model_path, tag_mapping_path)，添加时默认继承当前双文件相对路径；**「自定义 repo」镜像覆盖退役（D4）** | 新增：选 model 文件 + tag_mapping 文件（extra 双路径） | `model_id` 为非官方值时生成一条 download 候选；**选中值不动**（纪律 1，用户 fork 继续生效），内置 preset 恒指官方 repo |

## 6. API 与前端

- **端点收敛**：`POST /api/model-sources/{domain}` 添加候选（服务端做 §2 校验）、`DELETE /api/model-sources/{domain}` 移除；下载仍走通用 `POST /api/models/download`。`/api/upscalers/download_custom` 与 `/api/models/{family}/custom` 在对应刀退役。
- **catalog 统一 shape**：五处 variants 统一为 `{ id, label, kind: preset|download|local|scanned, exists, size, files?, removable, deletable, is_current }`，能力位由后端拼好，前端不再各自判断。
- **前端单组件**：WD14ModelCard / EvalMetricModelCard / CLTaggerModelCard / UpscalerSection 列表段 / ModelsSection 列表段合并为一个泛化候选卡组件；per-domain 只剩配置（标题、添加表单字段、help 文案）。ModelIdsEditor 退役。打标页双 surface 继续复用同一组件（候选写入仍走该页 `commitSecrets` 链路，不引入第三条链路）。

## 7. PR 切分

每刀合完独立可用，无临时脚手架：

1. **PR-1（wd14 + eval + 统一地基）**：`model_sources` schema + 迁移 + Secrets 级不变量；catalog 统一 shape；`/api/model-sources` 端点；前端泛化候选卡组件首发，wd14 / eval ×3 切换（ModelIdsEditor 退役）。
2. **PR-2（放大器 + 训练主模型）**：放大器候选持久化 + 表单收进添加入口 + 独立 source 下拉退役 + `download_custom` 端点退役；训练主模型加下载型添加 + `{family}/custom` 端点合并 + 🗑→× 语义修正。
3. **PR-3（CLTagger + 收尾）**：CLTagger 候选化 + 镜像覆盖退役 + 迁移；全部旧入口 / 死代码清理。

## 8. 决策记录

| # | 决策 | 内容 |
|---|---|---|
| D1 | 统一心智模型 | 五处全部 = 候选列表 + 两种添加（下载 / 本地文件），全部支持两种 |
| D2 | 动作语义 | 下载型 = 移除 + 删除两个独立动作；本地型 = 只移除；内置 = 只删除（不可移除，保护默认） |
| D3 | 校验从简 | 格式 / 存在性校验；不做网络探测，运行时报错兜底 |
| D4 | CLTagger 镜像覆盖退役 | fork 场景改为添加完整候选行；内置 preset 始终指官方 repo |
| D5 | secrets 完全统一 | 单一 `model_sources` 结构；两条兼容纪律（选中值字段不动 + 旧字段 computed 键保留写盘） |
| D6 | 扫盘兜底保留 | upscalers/ 目录扫盘项作为 `scanned` 类候选，只有删除动作 |
