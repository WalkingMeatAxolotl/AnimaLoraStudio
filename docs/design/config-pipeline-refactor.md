# 训练 config 管线重构(YAML 格式不变)

状态:讨论中。已定决策见「决策记录」;未定项见「Open Questions」。
触发:2026-07-17 krea2 训练 `shuffle_caption` 拒训 bug(见下)暴露的结构性同步成本。

## 1. 背景:触发 bug 复盘

fork anima 版本 → FamilySwitchDialog 切到 krea2 → 落盘裁剪把 `shuffle_caption` 键裁掉(show_when 为假,PR #366 语义)→ studio 侧 pydantic 读回时 `FAMILY_CONFIG_DEFAULTS` overlay 补 False,一切正常;但 trainer 走 argparse bridge 加载,缺键落回 schema 裸默认 True(anima 语义),第三层能力防线拒训。

根因:**同一份 yaml 有两套「缺失键默认值」语义**——pydantic 路径(validator/overlay 全生效)与 argparse 路径(全不生效)。这是下述 A 类痛点的一个实例。

## 2. 现状全景

单一权威源 `TrainingConfig`(studio/domain/training.py,169 字段 / 17 validator)派生三个面:

```
TrainingConfig (pydantic, 单一权威源)
 ├─ model_json_schema() ──→ GET /api/schema ──→ 前端 SchemaForm 泛型渲染
 ├─ build_parser()      ──→ trainer CLI argparse(argparse_bridge.py)
 └─ _tolerant_validate  ──→ 保存/preset 导入时校验 + 修复
```

写路径:前端完整 config → PUT → `_tolerant_validate`(17 validator + 迁移)→
`prune_inactive_fields`(裁 show_when 假 + hidden 默认值)→ yaml 落盘。

读路径(studio):yaml → pydantic(迁移 + 族 overlay + 校验)→ model_dump 补全默认 → 前端。

读路径(trainer):yaml → `apply_yaml_config`(手工重跑 2 个迁移)→
`merge_yaml_into_namespace`(「值==默认值」近似 CLI 显式性,**绕过全部 validator**)→
argparse Namespace(训练进程的实际真相)→ 第三层能力校验。

validator 构成:归一化 1 + 族 overlay 1 + 迁移 2 + 族门控 3 + 互斥 8 + 区间 2。

## 3. 痛点分类(改一个东西要同步多处的结构性原因)

| # | 痛点 | 实例 |
|---|---|---|
| A | **argparse Namespace 是第二个运行时真相**:merge 绕过 pydantic,每个 before-validator 都要在 runtime bootstrap 手工补调 | 迁移函数双调(bootstrap.py:98);族 overlay 漏调 → shuffle_caption bug;互斥/区间校验 runtime 侧不跑 |
| B | **跨语言双实现**:show_when 求值器 + prune 前后端逐字镜像,靠注释纪律 + 平行测试 | `eval_show_when`↔`evalShowWhen`;`prune_inactive_fields`↔`pruneInactiveConfig`;`_js_str` 手工复刻 JS `String()` |
| C | **跨进程三份矩阵**:FAMILY_CAPABILITIES / FAMILY_CONFIG_DEFAULTS / FAMILY_SAMPLING 在 studio/domain 与 runtime SPECS 各一份 | 靠 test_model_family_gating.py 单点锁死;加族/加能力位两边写 |
| D | **元数据表达不了的逻辑逃逸成硬编码且已漂移** | SchemaForm `isAutoLrOptimizer` 缺 soap_sf(与后端 validator 名单不一致);disable_when 的「reset 到 disable_value」无后端对应物,tolerant 修复语义是 reset 到 default(learning_rate:disable_value=1.0 ≠ default=1e-4,两侧修复结果不同) |
| E | **一条互斥规则写三处**:UI 灰显(disable_when 元数据)+ 后端 fail-fast(8 个同构 validator)+ 容错修复(InfoNoise 专用垫片) | 加一对互斥 = 三处开工 |

## 4. 业界对照

| 参考 | 对应痛点 | 核心做法 |
|---|---|---|
| jsonargparse / LightningCLI | A | yaml+CLI 合并后实例化类型对象、校验一次、运行时读对象;CLI 显式性用 `argparse.SUPPRESS` 精确探测 |
| Kubernetes CRD + CEL | E | 校验规则声明进 schema(`x-kubernetes-validations`),规则=随 schema 传输的数据而非散落代码 |
| VS Code settings when-clause | B/D | 统一表达式语言 + 单一求值器;设置 UI 全泛型渲染,零字段级硬编码 |
| Hydra / OmegaConf | C | 族默认 = base + overlay 声明式 compose |

本项目 `cap_gate()`「作者写时展开」已经是 K8s/VS Code 的思想,R2 是把它推广到互斥规则。

## 5. 改造方案

### R1. 收口运行时真相:trainer 过 TrainingConfig(根治 A)【已拍板,见 D1】

`apply_yaml_config` 改为:

```
yaml → TrainingConfig(**yaml)   # 全部迁移/overlay/互斥/区间校验单点生效
     → model_dump()
     → 覆盖 CLI 显式项(SUPPRESS parser 产出的 sparse namespace)
     → 再 validate 一次(CLI 覆盖也可能造出非法组合)
     → 填 argparse.Namespace(下游零改动)
```

- CLI 显式性从「值==默认值」近似改为 `argparse.SUPPRESS`(parse 结果只含用户真正传的键),语义变精确。
- runtime 三层能力防线退役为一层(pydantic 校验已含);bootstrap 手工迁移调用退役。
- 行为变化:裸 CLI / 手写 yaml 的非法组合从「静默跑」变 fail-fast 拒训(与 Studio 写盘校验对齐)。**已确认接受。**
- 依赖方向不变:runtime 本来就 import studio.schema。
- yaml 格式零改动。

### R2. 互斥/联动规则声明化(根治 D/E)【形态见 §6(v2 简化版),待拍板】

### R3. 族矩阵单源(根治 C)【方向已认可,并入刀 1】

capabilities / config_defaults / sampling 是纯数据;runtime→studio import 方向已存在,
让 runtime SPECS 直接引用 `studio/domain/common.py` 的数据(loader/build 代码逻辑留 runtime),
双份变单份,test_model_family_gating.py 的镜像锁死测试退役。

### R4. YAML 预览改后端渲染 + 删前端镜像(缓解 B)【已拍板 D3;实现修订见下】

现状:保存链路 = onChange → 600ms debounce → PUT 自动保存(改后**不变**)。
预览抽屉独立于保存:前端把当前表单 state 实时过 `pruneInactiveConfig` + `configToYaml`
两个镜像函数本地渲染 → 零滞后,但"与落盘一致"靠镜像纪律声称。

**实现修订(刀 2 实施时)**:原方案「PUT 响应带回执」覆盖不了 Presets 页
(该页预览的是编辑中未保存的 preset,没有 PUT 可搭);改为统一预览端点
`POST /api/schema/preview-yaml` —— tolerant + 裁剪 + safe_dump 与落盘走
**同一个** `render_config_yaml` 序列化出口(io.py),纯计算不落盘。
ConfigYamlPanel 300ms debounce 调用,两页共用。达成同一目标:
- 删 `pruneInactiveConfig` + `configToYaml` 两镜像及其测试,裁剪/序列化语义唯一实现在后端;
- 预览显示的是「点保存后文件会是的样子」,物理一致;
- `evalShowWhen`(表单实时可见性)保留在前端,是跨语言双实现的最后一处。

### R5. 杂项

- 前端 `isAutoLrOptimizer` 硬编码退役(随 R2 元数据派生,刀 2 已做);
  automagic lr 改写保留但并入 R6 确认弹窗(它是「切换瞬间的建议性一次改写」,
  pin 语义表达不了,advisory 写值与规则 takeover 走同一个确认清单)。
- Generate/RegAi 配置重复的路径/sampler Literal 抽公共基类 —— **移出刀 2**,
  独立小 PR(与训练 config 管线无耦合,控制刀 2 规模)。

### R6. 规则触发的「有损改值」走结构化确认弹窗(FamilySwitchDialog 泛化)【待拍板】

现状不一致:model_family 切换有 modal 展示变更清单(from→to,用户确认才应用);
其他联动(选 ppsf/automagic → learning_rate 被改写、disable_when takeover)是**静默改值 + 文本说明**。

改后:规则声明化(R2)使「用户这次改动会连带改哪些字段」可计算——gate 字段变化时求值规则表,
得到受影响字段清单(from → to + reason),复用 FamilySwitchDialog 的结构化 modal 范式:
确认才应用,取消则回滚本次 gate 改动。FamilySwitchDialog 本身成为该机制的一个特例
(族切换规则的确认弹窗),两套 UI 归一。

**触发粒度(防弹窗疲劳)**:只在「有损」时弹——受影响字段当前值 ≠ 将写入值才列入清单,
清单为空(受影响字段本来就在钉值上,如 loss_weighting 本就是 none)则静默应用,零打扰。
与 switch_family 的 changes 清单只列真实变化字段同一语义。

## 6. R2 规则声明(v2 简化版):升级现有 disable_when 为单源

### 6.0 业界参考与取舍

| 参考 | 形态 | 取舍 |
|---|---|---|
| **JSON Schema `if/then`**(draft-07+,标准) | `if:{properties:{infonoise_enabled:{const:true}}} then:{properties:{loss_weighting:{const:"none"}}}`;ajv/RJSF 生态直接消费 | 与本提案语义完全同构(pin=`then..const`,forbid=`then..not.const`)。**不直接采用**:项目已有 show_when 微文法 + 双端求值器,引入 if/then 会造成两套条件文法并存;pydantic 产出 JSON Schema 但不消费 if/then 做校验,后端仍要自写求值器,标准化收益落空 |
| **Terraform provider schema** | `ConflictsWith/RequiredWith/ExactlyOneOf` 声明**挂在字段上**,大规模生态验证 | **采用其组织方式**:规则挂在字段元数据上,不建平行规则表 |
| **Kubernetes CRD + CEL** | 校验表达式随 schema 传输 | 思想同源(规则=数据);CEL 太重不引入 |

结论:**不引入新表、不引入新文法、不引入新概念**——现有 `disable_when`/`disable_value`/`disable_hint`
本来就是 pin 规则的完整声明(条件/钉值/理由),只是今天**只有前端消费**。
R2 v2 = 把这组既有元数据升级为「单源、双端强制」,并补一个对称的 option 级键承载 forbid。

### 6.1 机制(零新概念)

1. **pin 规则** = 字段上的 `disable_when` + `disable_value` + `disable_hint`(全部已存在):
   - 前端:灰显 + takeover 写值(现状不变,但硬编码分支退役,全走元数据);
   - **后端(新增消费)**:通用 model_validator——disable_when 命中且值 ≠ disable_value → raise,
     文案 = 字段 + 钉值 + disable_hint;8 个手写互斥 validator 退役;
   - **tolerant 修复(新增消费)**:违反时写 disable_value(修复语义与前端 takeover 对齐,
     消除现状 reset-to-default 的不一致,如 learning_rate 1.0 vs 1e-4)。
2. **forbid 规则** = 新元数据键 `option_disable_when: {值: when表达式}`(`option_show_when` 的姊妹):
   - 前端:下拉中该选项**灰显 + hint**(D4 已拍板灰显不隐藏);
   - 后端:同一 validator 消费,命中且等于禁值 → raise / tolerant 修复为 schema 默认。
3. **fix=gate 策略**(InfoNoise 垫片泛化):极小全局集合 `TOLERANT_FIX_GATE_FIRST = {"infonoise_enabled"}`
   ——tolerant 修复时,违反规则的 when 表达式含集合内开关则优先关开关(保住用户在目标字段的投入),
   其余默认修目标字段。不做 per-rule 配置,YAGNI。

### 6.2 现有 11 组规则逻辑 → 字段元数据(最终形态)

写在各字段 `_meta(...)` 里(多 gate 钉同字段用 `||` 合并,现文法已支持):

| 目标字段 | 新增/升级的元数据 | 替代的手写代码 |
|---|---|---|
| `lr_scheduler` | `disable_when="optimizer_type==automagic\|\|…prodigy\|\|…prodigy_plus_schedulefree\|\|…soap_sf", disable_value="none"`(已有,升级为双端强制) | `_validate_prodigy_scheduler` + 前端 `isAutoLrOptimizer` takeover 硬编码 |
| `loss_weighting` | `disable_when="infonoise_enabled==true\|\|leap_enabled==true", disable_value="none"` | infonoise/leap × loss_weighting 两个 validator |
| `timestep_schedule_shift` | `disable_when="infonoise_enabled==true", disable_value=1.0` | infonoise × schedule_shift validator |
| `noise_enhancement_type` | `disable_when="infonoise_enabled==true", disable_value="none"` | infonoise × noise_enhancement validator |
| `loss_type` | `option_disable_when={"huber": "infonoise_enabled==true\|\|leap_enabled==true"}` | infonoise/leap × huber 两个 validator |
| `infonoise_enabled` | `disable_when="leap_enabled==true\|\|navit_packing==true", disable_value=False` | leap/navit × infonoise |
| `leap_enabled` | `disable_when="navit_packing==true", disable_value=False` | navit × leap |
| `sra_enabled` | `disable_when="navit_packing==true", disable_value=False` | navit × sra |
| `lora_type` | `option_disable_when={"tlora": "navit_packing==true"}` | navit × tlora |
| `cache_latents` | `disable_when="navit_packing==true", disable_value=True` | navit → cache_latents |
| `navit_native_resolution` | `disable_when="navit_packing==false", disable_value=False` | navit_native_resolution 依赖校验 |
| `attention_backend` | 已有 `disable_when="navit_packing==true", disable_value="xformers"`,升级为双端强制 | `_coerce_navit_attention_backend` before-validator |

另:InfoNoise 专用 tolerant 垫片(presets/io.py)→ 由 `TOLERANT_FIX_GATE_FIRST` 通用化后删除。

### 6.3 每条声明派生出什么(一处声明,五面生效)

1. **后端校验**:通用 validator 消费 disable_when/option_disable_when(手写互斥 validator 退役)。
2. **前端 UI**:灰显/takeover/hint——现有 SchemaForm 消费逻辑不变,新增 option 级灰显一处。
3. **tolerant 修复**:写 disable_value(与前端 takeover 语义对齐)/ gate-first 集合。
4. **R6 确认弹窗**:有损改值清单直接从同一份元数据求值生成。
5. **runtime**:R1 后走同一 TrainingConfig,自动继承。

### 6.4 保留手写的 validator(规则声明刻意不覆盖)

| validator | 原因 |
|---|---|
| `_validate_detail_inv_t_range` / `_validate_sra_decay_range` / `_validate_infonoise_n_min_le_b` | 字段间区间比较(min≤max),3 个且语义各异,模板化收益低 |
| navit_token_budget > 0 | 「必须显式设置非零」不是钉值/禁值 |
| `_normalize_resolution` / 迁移 2 个 / 族 overlay / sampler grandfather coerce | 归一化/迁移类,本就不是规则 |

判据:**bool/enum 门控 → 钉值/禁值**进表;涉及数值关系或改写逻辑的保留手写。

## 7. PR 切分(合并后的两刀)

| 刀 | 内容 | 状态 |
|---|---|---|
| **刀 1(后端收口)** | R1 + R3 + shuffle_caption bug 正式修复 + 回归测试(krea2 裁剪产物过 trainer 加载路径 → 零 violation);三层防线/双调迁移/镜像矩阵测试退役 | **已合(PR #431)** |
| **刀 2(声明化 + 前端)** | R2 v2(disable_when 双端强制)+ R4 预览端点 + R6 确认弹窗 + 审计补录(#1/#6 规则、#3 runtime 分派、#2 warning) | 已实施待合 |
| 遗留(独立小 PR) | Generate/RegAi 公共基类(R5);审计 #7 数值复核;golden vectors 锁 evalShowWhen(跨语言双实现仅剩这一处,优先级降低) | 待排期 |

## 8. 决策记录

- **D1(2026-07-17)**:R1 采纳,含行为严格化——裸 CLI/手写 yaml 的非法组合从静默跑变 fail-fast,与 Studio 写盘校验对齐。
- **D2(2026-07-17)**:PR 切分从五刀合并为两刀(§7)。
- **D3(2026-07-17)**:R4 采纳——预览改后端落盘回执,删 pruneInactiveConfig/configToYaml 前端镜像。
- **D4(2026-07-17)**:forbid 规则前端形态 = 灰显选项(带 hint),不隐藏。
- **D5(2026-07-17)**:R2 v2 采纳——不建规则表,升级现有 disable_when/disable_value/disable_hint 为双端强制,新增 option_disable_when 承载禁值;v1(独立 RULES 表)废弃。
- **D6(2026-07-17)**:R6 采纳——规则触发的有损改值走结构化确认弹窗,「有损才弹、无损静默」;FamilySwitchDialog 并入同一机制。
- **D7(2026-07-17)**:审计候选处置按 §10.1 表定案——#1/#6 补规则声明、#3 修 runtime 分派错位、#2 加 warning、#4 文档说明、#5 保留、#7 先数值复核再定;随刀 2 落地。

## 9. Open Questions

(当前无——D1-D7 已覆盖全部;#7 数值复核结果若确认漂移,补一条互斥即可,不阻塞。)

## 10. 互斥覆盖审计结果(2026-07-17,runtime 全量对照)

方法:深读 runtime/training/ 各机制实现里的 guard/skip/覆盖逻辑,对照配置层
(11 个 validator + disable_when 元数据 + FAMILY_CAPABILITIES)。

### 10.1 遗漏候选(待逐条拍板)

| # | 组合 | runtime 证据 | 静默后果 | 处置建议 |
|---|---|---|---|---|
| 1 | **leap + sra** | loop.py:448 leap 步整段跳过 SRA;SRA MLP 仍建仍进优化器。leap_ratio=1.0 时 SRA 100% 零生效 | **高**:用户以为开了,实际不跑 | **补规则**:`sra_enabled` 加 `disable_when="leap_enabled==true\|\|navit_packing==true"`(与 navit 侧对称),leap validator 期加 sra 互斥 |
| 2 | tlora + batch_size>1 | ortho/lycoris_adapter 取 batch 均值 t 生成单一 rank mask 广播,per-sample 机制退化为批均值 | 中:行为偏移非失效 | **warning 不拦**:bootstrap 期打印;强拦会误伤小 batch 用户 |
| 3 | noise_offset>0 且 pyramid_noise_iters>0 | runtime 从不读 noise_enhancement_type,直接读两个原始值,各自独立生效可叠加;互斥只是 UI 显隐+prune 假象 | 中:裸 CLI/脏 yaml 双倍低频扰动 | **修错位**:runtime 改按 noise_enhancement_type 分派(根治「type 被 runtime 无视」);R1 后 pydantic 校验兜底 |
| 4 | krea2 + infonoise(timestep_sampling 改离 krea2_shift 后) | timestep_samplers/__init__.py:42 只 fail-fast「仍是 krea2_shift」子集;改 logit_normal 再开 infonoise → krea2 静默失去校准 shift | 中:需用户显式两步操作 | **半拦**:krea2 的 timestep_sampling 纳入族白名单警告(同 sampler 白名单模式),或文档说明;倾向后者(A1「同代码不限制」先例) |
| 5 | krea2 + timestep_shift_resolution_aware=true | loop.py:178 runtime 已 fail-fast raise | 低:崩但不静默 | **保留**(失败点偏晚但响) |
| 6 | automagic v2 + grad_clip>0 | automagic.py:52 仅 warning,fused backward 下裁剪静默失效;grad_clip 默认 1.0 极易撞 | 中:默认值即触发,仅一行 log | **补规则**:`grad_clip_max_norm` 加 `disable_when="optimizer_type==automagic&&automagic_variant==v2", disable_value=0`(grad_accum/fp16 已 fail-fast 保留) |
| 7 | infonoise + resolution_aware shift(多分辨率) | infonoise CDF 建在 shifted-t 上、采样值又被 shift 一次,refresh 可能沿复合方向漂移;单分辨率恒等无影响 | 低-中,**未数值验证** | **先复核**:确认漂移则按 schedule_shift 同款互斥;有界则文档说明 |

### 10.2 已核实真兼容、不需拦(防重复怀疑)

masked_loss×infonoise(loop.py:414 显式排除 mask 区不污染 CDF)、masked_loss×reg/loss_weighting、
reg×infonoise(仅 main 集进 record)、reg×leap/navit(loss_weight 均生效)、
loss_weighting/huber×navit(逐图 t 恰好匹配 per-sample SNR 语义,与 leap 不同,现状不拦正确)、
noise_enhancement×leap、timestep 系×leap(已由 alt_description 披露 leap 恒用 U(0,1))、
krea2×huber/loss_weighting/noise_enhancement/各 optimizer、sra×reg、eval×训练机制(正交)。

### 10.3 附带发现(非配置问题,已单独开任务)

navit_packing 路径 NameError 回归:loop.py:309 引用从未定义的 `t5_attn`
(文本编码下沉 family 时 encode_text_for_batch 只 return cross,navit 分支未同步),
navit_packing=True 首个 batch 即崩。已核实(grep 全文件唯一出现)。独立修复,不进本重构。
