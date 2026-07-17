"""disable_when 规则引擎 —— 字段联动声明的双端强制(刀 2 / R2 v2)。

设计(docs/design/config-pipeline-refactor.md §6,D5):不建平行规则表,字段上
既有的 `disable_when` + `disable_value` + `disable_hint` 元数据就是一条完整的
「钉值」规则声明(条件 / 钉值 / 理由),`option_disable_when` 是它的 option 级
姊妹(「禁值」:条件成立时该枚举值不可选)。本模块把这份声明升级为单源多面:

1. 后端校验 —— TrainingConfig._enforce_disable_rules 调 disable_rule_violations,
   违反即 raise(替代历史上按对手写的 8 个互斥 validator);
2. tolerant 修复 —— _tolerant_validate 调 apply_disable_rule_fixes,修复语义与
   前端 takeover 对齐(写 disable_value,而非笼统回 schema default);
3. 前端 UI —— SchemaForm 消费同一份元数据做灰显 + takeover(既有行为);
4. R6 确认弹窗 —— 有损改值清单从同一份声明求值生成。

强制判据:disable_when 强制的是「违反时 runtime 静默失效或语义错误」的硬规则。
UI 引导性的软钉值(如 learning_rate 对 Prodigy 是真实生效的缩放因子,钉 1.0
只是推荐)列入 ADVISORY_DISABLE_FIELDS,保持 UI-only,不参与校验与修复。

eval_show_when / _js_str 从 config_prune 下沉至此(本模块是零依赖叶子,
training.py 的 validator 要 import 本模块,而 config_prune import training,
放那边会成环);config_prune 继续 re-export,既有消费方不受影响。
"""
from typing import Any, Mapping

from pydantic import BaseModel

_MISSING = object()

#: disable_when 保持 UI-only 的字段(软钉值)。判据见模块 docstring;
#: 加字段前先确认「违反它 runtime 是否真的静默失效」——不是的进这里。
ADVISORY_DISABLE_FIELDS: frozenset = frozenset({"learning_rate"})

#: tolerant 修复的 gate-first 集合:规则违反且 when 表达式含集合内开关时,
#: 优先关掉开关本身(保住用户在目标字段上的投入),其余默认修目标字段。
#: 历史上 presets/io.py 的 InfoNoise 专用垫片(「优先关 InfoNoise 保住
#: loss_weighting 等配置」)的泛化,该垫片已由本机制替代。
TOLERANT_FIX_GATE_FIRST: frozenset = frozenset({"infonoise_enabled"})


def _js_str(value: Any) -> str:
    """JS `String(v)` 的等价物 —— show_when 比较按前端的字符串化语义。

    差异点:JS 的 true/false 小写;整数值的 float 不带小数点
    (String(1.0) === "1",而 Python str(1.0) == "1.0",training.py 里
    `timestep_schedule_shift!=1` 依赖这一点)。
    """
    if value is True:
        return "true"
    if value is False:
        return "false"
    if value is None:
        return "null"
    if value is _MISSING:
        return "undefined"
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def eval_show_when(expr: str | None, values: Mapping[str, Any]) -> bool:
    """schema.ts evalShowWhen 的逐字镜像:`||` 分支 any、`&&` 合取 all、
    `==`/`!=` 字符串比较;空表达式与解析不出的表达式都返回 True(failsafe)。"""
    if not expr:
        return True
    branches = [p.strip() for p in expr.split("||") if p.strip()]
    if len(branches) > 1:
        return any(eval_show_when(b, values) for b in branches)
    ands = [p.strip() for p in expr.split("&&") if p.strip()]
    if len(ands) > 1:
        return all(eval_show_when(c, values) for c in ands)
    eq = expr.split("==")
    if len(eq) == 2:
        return _js_str(values.get(eq[0].strip(), _MISSING)) == eq[1].strip()
    ne = expr.split("!=")
    if len(ne) == 2:
        return _js_str(values.get(ne[0].strip(), _MISSING)) != ne[1].strip()
    return True


def _expr_fields(expr: str) -> list[str]:
    """when 表达式里出现的字段名(比较式左侧),按出现顺序去重。"""
    out: list[str] = []
    for branch in expr.split("||"):
        for clause in branch.split("&&"):
            for op in ("==", "!="):
                parts = clause.split(op)
                if len(parts) == 2:
                    name = parts[0].strip()
                    if name and name not in out:
                        out.append(name)
                    break
    return out


def _schema_default(field) -> Any:
    return field.get_default(call_default_factory=True)


def iter_pin_rules(model_cls: type[BaseModel]):
    """遍历强制钉值规则:yield (field, when_expr, pin_value, hint)。

    pin_value = disable_value(声明了则修复/校验语义与前端 takeover 一致),
    否则 schema default(前端 takeover 的同款回退)。
    """
    for name, field in model_cls.model_fields.items():
        extra = field.json_schema_extra
        if not isinstance(extra, dict):
            continue
        expr = extra.get("disable_when")
        if not isinstance(expr, str) or not expr or name in ADVISORY_DISABLE_FIELDS:
            continue
        pin = extra.get("disable_value", _MISSING)
        if pin is _MISSING:
            pin = _schema_default(field)
        yield name, expr, pin, str(extra.get("disable_hint") or "")


def iter_forbid_rules(model_cls: type[BaseModel]):
    """遍历禁值规则:yield (field, forbidden_value_str, when_expr, hint)。

    来源 = 字段元数据 option_disable_when: {枚举值: when 表达式}。
    """
    for name, field in model_cls.model_fields.items():
        extra = field.json_schema_extra
        if not isinstance(extra, dict):
            continue
        gates = extra.get("option_disable_when")
        if not isinstance(gates, dict):
            continue
        for value, expr in gates.items():
            if isinstance(expr, str) and expr:
                yield name, str(value), expr, str(extra.get("disable_hint") or "")


def apply_pin_setdefaults(
    data: dict[str, Any], model_cls: type[BaseModel]
) -> dict[str, Any]:
    """构造期 setdefault ——「缺省跟随钉值,显式违反才报错」。

    pin 规则 when 为真且目标字段 **未显式提供** 时,落到钉值而非 schema 默认
    (与前端 takeover / FAMILY_CONFIG_DEFAULTS overlay 的 setdefault 同构):
    `TrainingConfig(navit_packing=True)` 的 attention_backend 应自动落 xformers,
    而不是拿着 schema 默认 flash_attn 去撞 after 校验。显式提供且违反的值
    不在此改写 —— 由 _enforce_disable_rules fail-fast,用户显式配置绝不静默改。

    when 求值视图 = data 缺键补 schema default(条件字段缺失按默认求值)。
    """
    view: dict[str, Any] | None = None
    out = data
    for name, expr, pin, _hint in iter_pin_rules(model_cls):
        if name in data:
            continue
        if view is None:  # lazy:多数构造没有缺失的 pin 字段
            view = {
                k: (data[k] if k in data else _schema_default(f))
                for k, f in model_cls.model_fields.items()
            }
        if eval_show_when(expr, view):
            if out is data:
                out = dict(data)
            out[name] = pin
            view[name] = pin
    return out


def disable_rule_violations(
    values: Mapping[str, Any], model_cls: type[BaseModel]
) -> list[dict[str, Any]]:
    """返回违反清单,每项 {field, expected, actual, hint, kind}。

    kind = "pin"(须等于 expected)| "forbid"(不可等于 actual,expected 为
    修复回退值 = schema default)。对完整 values 求值(与前端对完整表单 state
    求值一致)。
    """
    out: list[dict[str, Any]] = []
    for name, expr, pin, hint in iter_pin_rules(model_cls):
        if eval_show_when(expr, values):
            actual = values.get(name, _MISSING)
            if actual is not _MISSING and _js_str(actual) != _js_str(pin):
                out.append({
                    "field": name, "expected": pin, "actual": actual,
                    "hint": hint, "kind": "pin",
                })
    for name, forbidden, expr, hint in iter_forbid_rules(model_cls):
        actual = values.get(name, _MISSING)
        if actual is not _MISSING and _js_str(actual) == forbidden and eval_show_when(expr, values):
            out.append({
                "field": name,
                "expected": _schema_default(model_cls.model_fields[name]),
                "actual": actual, "hint": hint, "kind": "forbid",
            })
    return out


def apply_disable_rule_fixes(
    data: dict[str, Any], model_cls: type[BaseModel]
) -> tuple[dict[str, Any], list[str]]:
    """tolerant 修复:按规则把违反字段修到合法值,返回 (新 dict, 修过的字段名)。

    每轮只修一处然后重算 —— 双向对称锁下一个冲突会产生两条违反(A 的规则钉 A、
    B 的规则按 gate 关 A),同轮全修会把两边都改掉、过度修复(如 huber+InfoNoise
    冲突把 huber 也抹了);单步 + 重算后,关掉 gate 的那一步会让对侧违反自然消失。

    选步策略:任何违反的 when 表达式含 TOLERANT_FIX_GATE_FIRST 开关且该开关
    当前为真 → 先关开关(牺牲开关、保住用户在目标字段上的投入,历史 InfoNoise
    垫片语义的泛化);否则钉第一条违反的目标字段(pin 写 disable_value /
    forbid 写 schema default)。

    注意:对 **完整** dict 求值;缺键字段不参与判定(pydantic 构造时的族
    overlay / schema 默认不会造出违反 —— 默认组合恒合法,由测试锁)。
    """
    out = dict(data)
    fixed: list[str] = []
    for _ in range(20):  # 上限 > 规则总数;正常几步内收敛
        violations = disable_rule_violations(out, model_cls)
        if not violations:
            break
        target = value = None
        for v in violations:
            gates = [
                g for g in _expr_fields_of_violation(v, model_cls)
                if g in TOLERANT_FIX_GATE_FIRST and bool(out.get(g))
            ]
            if gates:
                target, value = gates[0], False
                break
        if target is None:
            v = violations[0]
            target, value = v["field"], v["expected"]
        out[target] = value
        if target not in fixed:
            fixed.append(target)
    return out, fixed


def _expr_fields_of_violation(violation: dict[str, Any], model_cls: type[BaseModel]) -> list[str]:
    """取该违反对应规则的 when 表达式字段(gate 候选)。"""
    name = violation["field"]
    extra = model_cls.model_fields[name].json_schema_extra
    if not isinstance(extra, dict):
        return []
    if violation["kind"] == "pin":
        expr = extra.get("disable_when") or ""
    else:
        expr = (extra.get("option_disable_when") or {}).get(_js_str(violation["actual"]), "")
    return _expr_fields(expr) if isinstance(expr, str) else []
