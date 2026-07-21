# version config 所有权模型（初始化一次，之后用户主权）

## 1. 背景：issue #458 复盘

用户在训练配置页设了 `output_name`，保存后 YAML 预览正常，但一点「开始训练」就变回
`{slug}_{label}`。与中英文无关。

直接原因在 `enqueue_version_training`：为同步全局模型路径，它把整份 config 读出再整份
写回，且用了 `force_project_overrides=True` —— 这个 flag 会用
`project_specific_overrides()` 重置全部 `PROJECT_SPECIFIC_FIELDS`，其中就有
`output_name`。同一行还会把 `resume_lora` / `resume_state` 清成 `None`，即
**0.20.0 起接续训练在入队瞬间被静默清空**，只是没人报。

引入自 `5e986f6f`（PR #439，随 v0.20.0 发布）。

复盘的结论不是「flag 传错了」，而是两个结构性问题：

1. `PROJECT_SPECIFIC_FIELDS` 是按「来源不是用户表单」这个表面特征聚起来的桶，里面混着
   权威来源完全不同的字段，却用一个布尔 flag 控制 —— 在任何时机点上只能全对或全错。
2. 消费期（入队）出现了写盘。只要消费期写 config，所见即所得就有破口。

## 2. 现状全景

### 2.1 `project_specific_overrides` 的四个调用点

| 调用点 | 时机 | 性质 |
| --- | --- | --- |
| `presets/__init__.py` `fork_preset_for_version` | 换预设 | 创建期 |
| `services/projects/versions.py` 复制 version | 复制 | 创建期 |
| `services/data_io/train_io.py` bundle 导入 | 导入 | 创建期 |
| `api/routers/projects/training.py` `enqueue_version_training` | **入队** | **消费期** ← 异类 |

前三个是构造一份还不存在的 config，覆盖不违反任何契约。第四个发生在用户已经看过、
编辑过、YAML 预览确认过之后。

### 2.2 消费期那次写盘为什么会被加进来

trainer 是独立子进程，只吃 `--config <path>`，直接读磁盘 yaml 原文，不经 studio 的任何
读取面（`supervisor/cmd_builder.py` 全部 override 通道只有 `--monitor-state-file` 和
`--resume-state`）。所以任何「跟随外部状态」的字段，不落盘就不生效。

而以下字段的真值确实不住在 config 里：

- 4 个模型路径（`auto_sync_paths=ON` 时跟随全局 selected）
- `reg_data_dir`（真值是 `reg/meta.json` 存不存在）
- `trigger_word`（真值是 `versions.trigger_word` 表列，Tagging 页写入）

后两个此前从未在消费期同步过，PR #439 的 `force=True` 顺带把它们也刷新了 —— 意外修好了
「先 fork 后打标 / 后生成 reg 集则 config 过期」这个老 bug。这是它没被 review 挡下的原因：
行为上确实变好了，代价是 `output_name` / `resume_*` 被一起砸掉。

### 2.3 WYSIWYG 破口不止一处

4 个模型路径在读取面有 `apply_global_path_overlay`，页面显示的就是入队后会落盘的值，
所见即所得成立。但 `trigger_word` 和 `reg_data_dir` **没有读取面 overlay** —— 今天同样是
「页面显示旧值、训练实际用刷新后的值」，只是刷新方向恰好是用户想要的，所以没人报。

即：需要在消费期刷新的字段，读取面和写入面必须用同一个 overlay，否则必然产生破口。

## 3. 新模型：所有权三阶段

**config.yaml 从「部分字段是缓存」变成「全部字段都是用户真值」。**

| 阶段 | 谁能写 | 具体 |
| --- | --- | --- |
| 初始化 | 系统写一次 | fork 预设 / 复制 version / bundle 导入 |
| 编辑 | 只有用户 | `PUT /config` |
| 消费 | **零写** | 入队 / spawn 纯读 |

这不是新设计，而是回到原设计。`services/models/families/anima.py`
`default_paths_for_new_version` 的 docstring 一直写着：

> 用户在 settings 切了 selected_anima → 之后新建的 version 自动用新选择；
> **已存在 version 的 yaml 不动（重现性）**

PR #439 是对它的偏离。本次改造是回到原意，并补上原设计缺的那一环 —— 用户此前没有称手的
方式改这四个字段（输入框 disabled），所以才会有人想用「自动跟随」去补偿。

## 4. 逐字段决策

| 字段 | 真值住在 | 初始化 | 用户 PUT | 消费期 |
| --- | --- | --- | --- | --- |
| `data_dir` / `output_dir` | 项目结构（派生） | 填 | 用户可改 | 不动 |
| `reg_data_dir` | 项目结构（**改为无条件填**） | 填 | 用户可改 | 不动 |
| `output_name` | 用户（默认值只是初值） | 填初值 | 用户可改 | 不动 |
| `resume_lora` / `resume_state` | 用户 | 填空 | 用户可改 | 不动 |
| 4 个模型路径 | 用户（初值按 `auto_sync_paths` 取全局） | 填初值 | 用户可改 | 不动 |
| `trigger_word` | —— | **字段退役** | —— | —— |

### 4.1 `reg_data_dir` 改为无条件填

不再判断 `reg/meta.json` 是否存在，初始化时直接填 `{vdir}/reg`。

runtime 对空目录 / 不存在目录完全容错（`runtime/training/phases/dataset.py`：路径不存在
只 warning + skip，目录空则 log「正则数据集为空，已跳过」）。所以填一个当前还没内容的
路径无副作用，而后来生成的 reg 集会自动生效。

这一改是删掉消费期刷新的前提：**当前的「条件填充」正是消费期刷新存在的理由之一。**

### 4.2 4 个模型路径：解锁 + dropdown + 恢复默认

- **初始化**：按 `auto_sync_paths` 决定是否用全局设置覆盖预设里的值（与当前行为一致）。
- **不再 disable**：输入框对用户开放，可手填、可浏览。
- **新增 dropdown**：从模型设置里已添加的模型直接选，选中即填入绝对路径。
  数据源用现成的 `/api/models/catalog`（已含各族 variant、本地注册的 custom 项、`exists`
  状态）。注意候选集按字段分，不是一份通用列表：`transformer_path` 的候选是 family
  variants + custom 底模，`vae_path` / `text_encoder_path` / `t5_tokenizer_path` 基本是
  固定资产。`model_family` 切换时候选集必须跟着换（krea2 的 config 塞 anima 路径 = 把
  config 改坏，fork 那边已按 family 取，前端同理）。
- **「恢复默认」小链接**：放在字段标签右侧（原「自动 · 全局设置」徽标位置），值与全局
  当前设置一致时不渲染；不一致时出现，点击填回全局值。走 labelExtra 小链接范式，不用
  按钮（避免撑高容器）。

这条链接是「不再自动跟随」的补偿：用户换了全局底模后，已有 version 不会被偷改，但页面
会明示"这里和全局设置不一样"，一键即可对齐。所见即所得与便利同时成立。

**`auto_sync_paths` 设置语义退化**：它现在管四件事（fork 时覆盖 / 读取面 overlay / 入队
刷新 / 导出预设时清回全局值），改造后只剩第一和第四件。名字与文案需要改成「新建版本时
使用全局模型设置」一类表述，否则用户仍以为在持续同步。

### 4.3 `trigger_word` 字段退役

`config.trigger_word` 编码了一个错误假设 —— **「每个 LoRA 恰好有一个 trigger」**。实际上
有的 LoRA 没有 trigger，有的有多个（多画风 style、角色的不同服装），有的用户用标签编辑
而非打标页完成。

该字段今天唯一的实际作用是 runtime `phases/bootstrap.py`
`_prepend_trigger_to_sample_prompts` 把它 prepend 到每条 sample_prompt。而 sample_prompt
本就是用户手写字段：想让采样图带 trigger 自己写进去即可，多 trigger 场景还能每条 prompt
写不同的。自动注入反而制造了「页面上的 prompt ≠ 实际用的 prompt」这一破口。

决定：

- **删除** `config.trigger_word` 字段与 runtime 注入逻辑。
- **保留** `versions.trigger_word` 表列，降级为「上次用打标页 prepend 的词」这一操作记忆：
  Tagging 页回填输入框、TagEdit badge、Overview tag 云高亮 —— 这三处展示的是「这批
  caption 里哪个词是 trigger」，属于打标操作的产物，不是训练配置。它不再外溢到 yaml。

打标页的触发词由此回到它本来的定位：**一个把词 prepend 到 caption 的语法糖。**

**不做迁移**（已拍板）：老 config 里设过 `trigger_word` 的用户，升级后采样图不再自动带
trigger，需要自己写进 sample prompt。用 release note 说明。为一个正在退役的错误假设写
迁移代码不划算。

## 5. 删除清单

改造后可以整段删掉：

- `enqueue_version_training` 里的 config round-trip（端点回到只剩校验 + 建行 + 发事件）
- `version_config.apply_global_path_overlay`（读取面 overlay）
- `write_version_config` 的 `force_project_overrides` 布尔 flag —— 三个创建期调用点合并到
  一个 `initialize_project_fields` 语义
- `config.trigger_word` 字段 + runtime `_prepend_trigger_to_sample_prompts`
- 4 个模型路径输入框的 disabled 逻辑与「自动 · 全局设置」徽标

顺带修掉：`supervisor/finalizer.py` 硬编码 `f"{slug}_{label}"` 找训练产物 —— 改读 config
的 `output_name`（读失败回退到旧拼法）。不修的话，自定义名训练出的
`{自定义名}_final.safetensors` 会认不到，`output_lora_path` 写不进版本记录。

## 6. PR 切分

按「中间态有没有坏行为」切，不按模块切。

**刀 1 — 所有权收口（后端 + 前端解锁）**

后端：入队删写盘、`reg_data_dir` 无条件填、`trigger_word` 字段退役、`force_project_overrides`
→ `initialize_project_fields`、删读取面 overlay、finalizer 读 config `output_name`。
前端：4 个模型路径取消 disable、「恢复默认」小链接、`auto_sync_paths` 文案改。
外加 release note（trigger 行为变更）。

这一刀关掉 #458，且自洽可发版。

之所以不能再切细：若只删消费期写盘而不解锁输入框，用户换了全局底模后既不自动跟随、又
改不动字段，只能靠换预设重建 config —— 这个中间态有真实受众（任何换过全局模型的用户），
不能留。同理 `trigger_word` 若不在同一刀退役，中间态会回归「打标页设了 trigger 但采样图
不带」的老 bug。

**刀 2 — 模型 dropdown**

输入框加下拉选择，从模型设置里已添加的模型直接选。纯 UX 增强，独立可发。刀 1 之后字段
已可手填 + 浏览，用户有出路，所以这一刀晚到不构成坏中间态。

## 7. 决策记录

- **D1** config.yaml 全部字段视为用户真值；系统只在初始化写一次，消费期零写。
- **D2** `reg_data_dir` 初始化时无条件填路径，不再判断 `reg/meta.json`。
- **D3** 4 个模型路径初值仍按 `auto_sync_paths` 取全局；字段解锁可编辑；不再持续跟随全局。
- **D4** 「恢复默认」小链接放字段标签右侧，与全局一致时不渲染。（用户拍板）
- **D5** `config.trigger_word` 字段退役，runtime 注入逻辑删除；`versions.trigger_word`
  表列保留为打标操作记忆。
- **D6** trigger 行为变更**不做迁移**，用 release note 说明。（用户拍板）
- **D7** 按中间态坏行为切 PR：刀 1 含前端解锁与 trigger 退役，刀 2 只做 dropdown。

## 8. Open Questions

- `auto_sync_paths` 语义退化后，这个开关是否还值得保留？留着的话它只影响「新建 version 的
  初值」和「导出预设时是否清回全局值」，可能不如直接固定行为。待刀 1 实施时再判。
- 「恢复默认」链接的判定口径：只比 4 个模型路径，还是所有有派生默认值的字段（`data_dir` /
  `output_dir` / `output_name` 等）都给？后者一致性更好但触及面更大。倾向先只做模型路径。
