# PP5 — 正则集生成

**状态**：✅ 已完成
**前置依赖**：PP4
**外部脚本来源**：`C:\Users\Mei\Desktop\SD\danbooru\dev\regex_dataset_builder.py`（**source of truth**）

## 目标

- 把 `regex_dataset_builder.py` **完整库化**为 `studio/services/reg_builder.py`，逻辑一字不改
- 基于 train 的 tag 分布，从 booru 拉一组「相似但不同」的图作正则集
- 落到 `versions/{label}/reg/1_general/{train-子文件夹镜像}/`，附 `meta.json`
- 前端 ⑤ 正则集页：参数 + 进度 + meta + 结果预览
- auto_tag 拉完后内联跑 WD14 给 reg 集打标（默认勾选）

**不在 PP5 范围**：
- ❌ 分辨率 K-means 聚类后处理（按裁剪比例统一分辨率）→ 移到 **PP5.5**
- ❌ 「补足」增量再跑 → 移到 **PP5.1**
- ❌ Danbooru Gold/Platinum 账户类型 UI 暴露（脚本逻辑保留，secrets 已支持）

## 关键约定

- **每个 version 一份 reg**（PP1 即决定）
- 触发条件：train 必须有图（端点 400 校验）；建议先打过标但非强制
- 选图算法：贪心 + 标签数递减搜索
  1. 扫 train 的 tag 分布（Counter）
  2. 标签权重 = 出现次数 / 总图数
  3. 每批前算 missing = target_weight − current_weight，按差排序
  4. 用 missing top N 个 tag 搜索 booru：N = 10 → 5 → 3 → 2 → 1（capped at max_search_tags）
  5. 每个 N 最多尝试 3 个不同 offset
  6. 单 tag 搜索失败的 → 加入 `failed_tags`，后续不再尝试
  7. 找到候选但本批未下载（候选都在 source / 都不符合）→ 标 `invalid_tag_combinations`
  8. 候选评分：tag_score（负 MSE，sigmoid 0.1）+ resolution_score(aspect 0.6 + width/height 0.4) × 0.1（resolution 仅作 tie-breaker）
  9. skip_similar：候选只取偶数索引（去掉相邻相似图）
  10. 连续 5 次未找到匹配 → 退出；max_rounds = 50
- **80% 达成率算成功**（与源脚本一致）
- 自动黑名单：把 `based_on_version`（version label）加入临时 blacklist，防同人画师互撞
- 跨版本去重：`collect_source_image_ids` 把 train 文件 stem 作为 ID 集合，下载时 skip
- 默认 reg 写到 `1_general/`（repeat=1 与 train 的 5_concept 形成对比）
- 子文件夹镜像 train：train/`5_concept` → reg/1_general/`5_concept`

## 后端

### A. 共用层 `studio/services/booru_api.py`

`downloader.py` 与 `reg_builder.py` 都用这套 HTTP 原语：
- `search_posts(api_source, tags_query, ...)` — 通用 booru 搜索
- `post_fields(post, api_source)` — 抽 (id, file_url, file_ext, tags_str)
- `post_dimensions(post, api_source)` / `post_tag_list(post, api_source)`
- `download_image(url, save_path, ...)` — 下载 + PNG 转换 + alpha 处理 + 完整性校验

`downloader.py` 已重构为引用本模块，单 tag 批量下载逻辑不变。

### B. `studio/services/reg_builder.py`

```python
@dataclass
class RegBuildOptions:
    train_dir: Path
    output_dir: Path
    api_source: str = "gelbooru"
    user_id: str = ""
    api_key: str = ""
    username: str = ""
    target_count: Optional[int] = None       # None = train 总数
    max_search_tags: int = 20                # gelbooru 默认 20，danbooru 免费 2 / gold 6 / platinum 12
    batch_size: int = 5
    excluded_tags: list[str] = []            # UI 勾选
    blacklist_tags: list[str] = []           # 全局
    skip_similar: bool = True
    aspect_ratio_filter_enabled: bool = False
    min_aspect_ratio: float = 0.5
    max_aspect_ratio: float = 2.0
    save_tags: bool = False                  # auto_tag 走 WD14 替代
    convert_to_png: bool = True
    remove_alpha_channel: bool = False
    auto_tag: bool = True
    based_on_version: str = ""               # 仅用于 meta + 自动黑名单

@dataclass
class RegMeta:
    generated_at: float
    based_on_version: str
    api_source: str
    target_count: int
    actual_count: int
    source_tags: list[str]            # 实际用过的搜索 tag
    excluded_tags: list[str]
    blacklist_tags: list[str]
    failed_tags: list[str]
    train_tag_distribution: dict[str, int]   # top 50
    auto_tagged: bool

def analyze_dataset_structure(dataset_path) -> dict
def collect_source_image_ids(source_path) -> set[str]
def calculate_missing_tags(target_weights, current_weights, blacklist_tags, failed_tags)
def calculate_tag_similarity / resolution_similarity / find_best_match / check_aspect_ratio
def build(opts, *, on_progress, cancel_event) -> RegMeta
def write_meta / read_meta / update_meta_auto_tagged
def preview_train_tag_distribution(train_dir, top=20)
```

### C. `studio/workers/reg_build_worker.py`

`python -m studio.workers.reg_build_worker --job-id N`，模式同 `download_worker` / `tag_worker`：

```python
def run(job_id: int) -> int:
    # 1. 读 job + version + secrets
    # 2. opts = RegBuildOptions(...)
    # 3. meta = reg_builder.build(opts, on_progress=progress, cancel_event=...)
    # 4. if opts.auto_tag and meta.actual_count > 0:
    #        ok = _run_auto_tag(reg_dir, progress)   # 内联 WD14，不开子进程
    #        reg_builder.update_meta_auto_tagged(reg_dir, ok)
    # 5. return 0 if meta.actual_count > 0 else 1
```

`project_jobs.VALID_KINDS` 加 `"reg_build"`（已加），supervisor 自动按 `kind="reg_build"` 路由到 `studio.workers.reg_build_worker`，无需改 `supervisor.py`。

### D. 端点（4 个）

```python
class RegBuildRequest(BaseModel):
    target_count: Optional[int] = None
    excluded_tags: list[str] = []
    auto_tag: bool = True
    api_source: str = "gelbooru"

@app.post("/api/projects/{pid}/versions/{vid}/reg/build")    # 启 job，推 stage=regularizing
@app.get("/api/projects/{pid}/versions/{vid}/reg")           # meta + image_count + files
@app.delete("/api/projects/{pid}/versions/{vid}/reg")        # rmtree(reg/1_general)
@app.get("/api/projects/{pid}/versions/{vid}/reg/preview-tags?top=N")  # 仅扫 train，不真生成
```

### E. VersionStats 新增

`stats_for_version` 返回值新增：
- `reg_image_count: int`（递归扫 reg/）
- `reg_meta_exists: bool`（reg/1_general/meta.json 存在性）

Stepper ⑤ 派生：`reg_meta_exists && reg_image_count > 0` → done。

## 前端

### `pages/project/steps/Regularization.tsx`

```
┌──────────────────────────────────────────────────────┐
│ ⑤ 正则集 — 基于 train tag 分布拉相似图              │
│ ─ 状态条 ─                                           │
│   reg 集存在: 95 张 · target 95/100 · gelbooru ·    │
│   auto-tag: ✓ · 5 分钟前  [清空]                     │
├──────────────────────────────────────────────────────┤
│ ─ 参数 ─                                             │
│   来源 [gelbooru▾] | 目标数量 [    ] | ☑ 自动 WD14  │
│   排除 train top tag (用户勾选):                    │
│     [+ 1girl ×88] [✕ character_x ×89] [+ solo ×60]  │
│   [开始生成]                                          │
├──────────────────────────────────────────────────────┤
│ ─ JobProgress ─（live SSE 日志 + 取消按钮）         │
├──────────────────────────────────────────────────────┤
│ ─ 结果预览 ─（reg/1_general/ 缩略图，复用 ImageGrid）│
└──────────────────────────────────────────────────────┘
```

### API client（`api/client.ts`）

```typescript
export interface RegMeta { ... }
export interface RegStatus { exists; meta; image_count; files }
export interface RegTagCount { tag; count }
export interface RegBuildRequest { target_count?; excluded_tags?; auto_tag?; api_source? }

api.getRegStatus(pid, vid)
api.previewRegTags(pid, vid, top)
api.startRegBuild(pid, vid, body)
api.deleteReg(pid, vid)
```

## 测试

| 文件 | 范围 |
|---|---|
| `tests/test_reg_builder.py` | 18 case：tag 标准化、analyze_dataset_structure、collect_source_image_ids、calculate_missing_tags、calculate_tag_similarity、find_best_match、build 主流程（写 meta/落图/skip source/blacklist/auto-blacklist version 名/cancel）、preview_train_tag_distribution、meta roundtrip |
| `tests/test_reg_endpoints.py` | 10 case：preview-tags / GET / POST(create job + 推 stage / 拒空 train / 拒非法 api_source / 拒负 target) / DELETE |
| `tests/test_reg_build_worker.py` | 3 case：worker 跑通 + meta 写盘 + auto_tag 改写 / unknown job / 关 auto_tag |
| `studio/web/src/components/ProjectStepper.test.tsx` | +1 case：reg done 派生 |

## 手测剧本

1. baseline version → 已有 train (5 张以上) + 全部已打标 → 进 ⑤ 正则集
2. 看到 reg 状态：「不存在」；train tag 分布 30 个（按频率排）
3. 勾选 1-2 个独占 tag（角色名等）排除
4. target_count = 5；勾选自动 WD14；点「开始生成」
5. JobProgress 实时滚日志：
   ```
   [start] version=baseline api=gelbooru target=5 auto_tag=True
   [reg] api=gelbooru train=...
   ===== 子文件夹 5_concept =====
     最缺失: a(缺0.50), b(缺0.40), ...
     用 5 tag 搜索: ['a','b','c','d','e']
     候选 100 张
     [1/5] ✓ 5001 score=-0.034 matched=['a','b']
     ...
   [reg-done] actual=5/5
   [auto-tag] 启动 WD14，5 张图
   [auto-tag] 5/5
   [auto-tag] done 5/5 (errors=0)
   ```
6. 完成 → 状态条更新到「reg 集存在: 5 张 · auto-tag ✓」
7. 结果预览出现 5 张缩略图 + 每张 .txt
8. Stepper ⑤ 出现 ✓
9. 切到 v2 → reg 状态独立（不存在）
10. 回 baseline → 点「清空」→ reg 消失，stepper ⑤ 回 active

## 风险 / 已知行为

| 项 | 行为 |
|---|---|
| 单 tag 搜索失败 | 加入 failed_tags，meta.failed_tags 体现，状态条显示「N 失败 tag」 |
| target 不足 80% | meta 仍写盘；状态条 actual/target 反映；用户可点「清空」重跑（PP5.1 加「补足」） |
| 用户勾排除 tag 太多 | 不强制校验最少剩 3 个 — 让 booru 自己处理「无结果」分支 |
| Auto-tag 失败 | catch 在 worker 内，meta.auto_tagged=false，reg 集本体保留 |
| 跨 train 撞图 | `collect_source_image_ids` 抓文件 stem，下载时 skip |
| skip_similar 偶数采样 | 默认开启，与源脚本一致 |
| K-means 后处理 | 移到 PP5.5；当前 reg 集分辨率不统一，靠训练侧 bucketing 处理 |
