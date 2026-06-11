# 首次启动引导 Modal（First-Run Onboarding）

**创建于** 2026-06-06
**状态** 🟡 设计已对齐，开始实现

---

## 背景

Settings 里有很多东西必须先下载/配置才能用（底模、VAE、Qwen3、T5、WD14、ONNX Runtime、训练加速等）。新人第一次打开 app 不知道这些东西的存在，只能在训练 / 打标时不断撞墙，再回到 Settings 里翻配置。

我们需要一个**首次启动引导 modal**，在选完语言后立即弹出，用 checklist 形式列出所有"基础能力"，每条可独立勾选安装，状态实时反馈，装完统一重启。

---

## 设计 Spec

### 触发与生命周期

| 时机 | 行为 |
|---|---|
| 首次启动 | `FirstRunLangModal` 选完语言关闭后立即弹 Onboarding modal |
| Skip / 关掉 | 写 `localStorage.studio.onboarding.done = "true"`，下次不自动弹 |
| 重新查看 | Settings 顶部加 **"重新运行首次引导"** 按钮（清 localStorage + 打开 modal） |
| 已装好的项 | 进 modal 时自动显绿勾，不需要重装 |

### 形态：Checklist（不是 Stepper）

- 一屏 modal 不分页（5 条目预估 ~600px 高，1080p / 14" MacBook 都能放下）
- 顶部全局下载源 + 中部 3 主条目 + 底部"高级"折叠区 2 条目 + sticky bar
- 复选框模式：默认勾推荐项 → 底部"一键安装选中(N)"按钮一次性触发

### Modal 布局

```
┌─ Onboarding ────────────────────────────────────────────┐
│  欢迎使用 AnimaLoraStudio                                │
│  下面是建议安装的基础能力，可以勾选后一键安装。            │
│                                                          │
│  下载源:  ◉ ModelScope   ○ HuggingFace                  │
│           (已为你按语言推荐，可改)                        │
│  ─────────────────────────────────────────────           │
│  ☑ 底模套件             ⚪ 未安装                        │
│    Anima + VAE + Qwen3 + T5,训练所需基础模型              │
│                                                          │
│  ☑ 打标功能             ⚪ 未安装                        │
│    WD14 + ONNX Runtime,自动给训练集打标签                │
│                                                          │
│  ▾ 高级选项                                              │
│    ☐ 训练加速                                            │
│       ◉ Flash Attention  ○ Xformers  ○ 不装             │
│    ☐ 图像增强 (Upscaler 4x-AnimeSharp)                  │
│  ─────────────────────────────────────────────           │
│  跳过引导                          [一键安装选中(2)]  →   │
└──────────────────────────────────────────────────────────┘
```

### 状态机

每条目五态：

| 状态 | 徽章 | 说明 |
|---|---|---|
| `idle` | ⚪ 未安装 | 默认，未触发安装 |
| `queued` | ⏳ 排队中 | 一键装中，等待轮到 |
| `installing` | 🔄 安装中 | 正在下载/安装，展开可看 log tail |
| `done` | ✓ 已安装 | catalog 显示全部组件 exists |
| `failed` | 🔴 失败 | 显 Retry 按钮，log 手动展开 |

**多组件聚合规则**（底模套件、打标功能）：
- 底层组件全 `exists` → `done`
- 任意一个 `running` → `installing`（显进度 `X/N`）
- 任意一个 `failed` → `failed`（其他不受影响继续装）
- 其余 → `idle`

**安装中态布局示例：**

```
┌─ 正在安装: 底模套件 (2/4)  ·  总进度 1/2 ────────────────┐
│  ☑ 底模套件             🔄 安装中  [▾ 查看日志]          │
│  ☑ 打标功能             ⏳ 排队中                         │
└──────────────────────────────────────────────────────────┘
```

**完成态：**

```
│  ✓ 全部安装完成,需要重启服务才能生效                       │
│  [稍后手动重启]              [现在重启服务]               │
```

如果没有需要重启的项（只装了底模套件），直接显示"完成 [关闭]"。

### 下载源

- 顶部全局选一次，按 i18n 当前语言推断默认：
  - `zh` → ModelScope
  - 其余 → HuggingFace
- 此默认值**也下沉到 Settings**：用户跳过 onboarding 直接进 Settings 时，如果 `secrets.download_source` 为空，按 i18n 显示默认值，不显示空白下拉

### 失败处理

- 状态变红 + 单独 Retry 按钮
- log tail 折叠在"展开详情"里，用户主动展开
- **不中断其他项的安装**（队列继续）

### 重启

- **全部装完后统一一次性提示**
- 复用 `POST /api/system/restart` + `pollHealthThenReload()`
- 中途不打断别的下载

### 进度反馈

- **不做 % 进度条**（HF/ModelScope/PyPI 没有可靠统一 %）
- 顶部 sticky："正在安装: 底模套件 (2/4 文件) · 总进度 1/2"
- 条目"展开"显 log tail（来自 `catalog.downloads[key].log_tail`）

### 训练加速（折叠区单选）

```
☐ 训练加速 (可选)
   ◉ Flash Attention  推荐,速度最快
   ○ Xformers         备选,flash-attn 装不上时用
   ○ 不装             CPU/旧显卡也能跑,只是慢
```

只展开一次、装一个、装完打勾就完事。

---

## 条目清单

| 条目 | 聚合内容 | 默认 | 区域 |
|---|---|---|---|
| 下载源 | HF / ModelScope 二选一 | 按 i18n 推断 | 顶部 |
| 底模套件 | Anima 主模型 + VAE + Qwen3 + T5 | ☑ 勾选 | 主区 |
| 打标功能 | WD14 + ONNX Runtime | ☑ 勾选 | 主区 |
| 训练加速 | Flash-Attn / Xformers / 不装（单选） | ☐ 不勾 | 折叠 |
| 图像增强 | Upscaler 4x-AnimeSharp | ☐ 不勾 | 折叠 |

---

## 不在 scope（继续走 Settings 配）

- 数据集凭证（Gelbooru / Danbooru token）
- LLM Tagger 预设
- CLTagger（WD14 一个够新人用，CLTagger 是进阶选项）
- PyTorch 重装（CUDA 版本切换；setup 时已装）
- HF / ModelScope token（不强制，无 token 也能下载公开模型；进阶可去 Settings）

---

## 代码改动清单

### 前端

| 文件 | 改动 |
|---|---|
| `web/src/components/FirstRunOnboardingModal.tsx` | 新建主组件 |
| `web/src/components/FirstRunLangModal.tsx` | 同层，关掉 lang modal 后触发 onboarding（或在 App 层挂） |
| `web/src/App.tsx`（或同层） | 挂载 onboarding modal + localStorage 控制 |
| `web/src/pages/tools/Settings.tsx` | (1) 顶部加"重新运行首次引导"按钮 (2) `download_source` 读取加 i18n 兜底 |
| `web/src/i18n/locales/zh.json` + `en.json` | onboarding 全部文案 |
| `web/src/storage.ts`（或同等） | `localStorage` helper：`studio.onboarding.done` |

### 后端

**不需要改后端接口。** 全部复用：
- `POST /api/models/download` — 触发下载
- `GET /api/models/catalog` — 查状态（含 `downloads.log_tail`）
- SSE `model_download_changed` — 实时状态推送
- `POST /api/system/restart` — 重启
- `secrets.download_source` — 已存在

---

## 实施计划

1. 写 design 文档（本文件）
2. 实现 `FirstRunOnboardingModal.tsx`（主组件 + 状态机）
3. App 层挂载（lang modal close → 弹 onboarding）
4. Settings 加"重新运行首次引导"按钮
5. Settings `download_source` 加 i18n 兜底
6. i18n 文案（zh + en）
7. 本地手测（首次启动模拟：清 localStorage 重开）
8. commit + push + PR → dev（squash 合入）

---

## 验证清单（手测）

- [ ] 清 localStorage 后启动：lang modal → 选完语言 → onboarding 自动弹
- [ ] 中文环境默认源 = ModelScope；英文默认 = HuggingFace
- [ ] 修改下载源，关闭重开 Settings：源持久化正确
- [ ] 跳过 onboarding 后，Settings 顶部按钮可重新触发
- [ ] 跳过 onboarding 直接进 Settings：下载源默认按 i18n 显示，不是空白
- [ ] 一键装：底模套件 + 打标功能并行/串行 OK，状态实时
- [ ] 装失败：变红 + Retry，其他项继续
- [ ] 全部装完：统一重启提示；点重启 → 服务回来后 modal 关闭
- [ ] 中途关 modal：后台继续装，重开 modal 显示进度
