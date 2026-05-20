"""Spike-PPSF: 验证 ProdigyPlusScheduleFree save→resume 后 p.data 语义是否正确。

ADR 0006 Addendum 1 第 7 条决策需要确定 PPSF resume fix 走方案 X / Y / Z 哪一种。
本脚本黑盒对比 4 个 candidate fix 的 loss / p.data 轨迹，跟 ground truth（不重启
继续训练）对照，找出最接近的方案。

跑法：
    python tools/spike/ppsf_resume.py > tools/spike/ppsf_resume.log 2>&1

不依赖：
- GPU（toy 16×16 MLP CPU 即可）
- 项目模型 / dataset / supervisor
- 真训练 state（fake state，几 KB）

依赖：
- prodigyplus（必须，pip install prodigy-plus-schedule-free）
- runtime/utils.optimizer_utils.optimizer_eval_mode（轻量 context manager）

输出：
- 每 phase 关键数值
- 末尾 SPIKE REPORT：4 方案 vs ground truth 的 loss / p.data 偏差排序
- 退出码 0 = 至少 1 个方案完美对齐 ground truth；非 0 = 全部方案都偏

ADR 0006 Addendum 1 候选方案 — 名称跟脚本 phase 对齐：
- 方案 0  (dev 现状)：load → 直接跑（不调任何 train/eval）
- 方案 A：load → optimizer.train() once
- 方案 B：load → optimizer.eval(); optimizer.train()
- 方案 D：save 不在 eval mode → state.pt 内 lora_state_dict 存 y（save-side fix）
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

# ── Windows console (cp932/cp936/gbk) 默认 codec 编不了中文 / em dash，
#    stdout 先 reconfigure 成 utf-8 + errors=replace 防 UnicodeEncodeError 崩进程
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

# ── 同时双写到 utf-8 文件，不靠 PowerShell `>` 重定向（PS 5.1 默认 UTF-16 LE）
_LOG_PATH = Path(__file__).with_suffix(".log")
_LOG_FILE = open(_LOG_PATH, "w", encoding="utf-8")


class _Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, s):
        for st in self.streams:
            try:
                st.write(s)
                st.flush()
            except Exception:
                pass

    def flush(self):
        for st in self.streams:
            try:
                st.flush()
            except Exception:
                pass


sys.stdout = _Tee(sys.stdout, _LOG_FILE)
print(f"[spike] tee logging to: {_LOG_PATH}")


import atexit


def _close_log():
    try:
        _LOG_FILE.flush()
        _LOG_FILE.close()
    except Exception:
        pass


atexit.register(_close_log)

import torch
import torch.nn as nn

# 让 spike 能找到 runtime.utils
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from utils.optimizer_utils import optimizer_eval_mode  # noqa: E402

try:
    from prodigyplus import ProdigyPlusScheduleFree
except ImportError:
    print(
        "[FATAL] prodigy-plus-schedule-free 未安装。"
        "pip install 'prodigy-plus-schedule-free>=2.0.0'"
    )
    sys.exit(2)


# ---------------------------------------------------------------------------
# 工具
# ---------------------------------------------------------------------------

INPUT_DIM = 16
OUTPUT_DIM = 16
BATCH = 4
WARMUP_STEPS = 80   # 让 PPSF averaging 拉开 y 跟 averaged x
RESUME_STEPS = 2000 # 长跑看 drift 增长趋势（线性 / bounded / 指数）
TREND_SAMPLE_EVERY = 100  # 每 N 步采样 cumulative max abs dev 输出趋势

# 固定一组训练 batch，避免 RNG 状态在重启/对照之间漂移
torch.manual_seed(42)
TRAIN_INPUTS = [torch.randn(BATCH, INPUT_DIM) for _ in range(WARMUP_STEPS + RESUME_STEPS)]
TRAIN_TARGETS = [torch.randn(BATCH, OUTPUT_DIM) for _ in range(WARMUP_STEPS + RESUME_STEPS)]


def make_model_opt(seed: int = 42):
    """新建一份 model + PPSF optimizer，权重确定性。"""
    torch.manual_seed(seed)
    model = nn.Linear(INPUT_DIM, OUTPUT_DIM, bias=False)
    optimizer = ProdigyPlusScheduleFree(model.parameters(), lr=1.0)
    return model, optimizer


def train_one_step(model, optimizer, step_idx: int) -> float:
    optimizer.zero_grad()
    out = model(TRAIN_INPUTS[step_idx])
    loss = ((out - TRAIN_TARGETS[step_idx]) ** 2).mean()
    loss.backward()
    optimizer.step()
    return loss.item()


def p_data(model) -> float:
    """返回 model 第一个参数的 [0,0] 元素，用作 p.data 状态探针。"""
    return float(next(model.parameters()).data.flatten()[0].item())


def section(title: str) -> None:
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


# ---------------------------------------------------------------------------
# Phase 1: 训练 WARMUP_STEPS，建立基线
# ---------------------------------------------------------------------------

section("Phase 1 — 训练 PPSF 20 步建立基线")

model_base, opt_base = make_model_opt()
# PPSF 默认 train mode；优先调一次 train() 确保状态对齐（无害）
if hasattr(opt_base, "train"):
    opt_base.train()

warmup_loss = []
for i in range(WARMUP_STEPS):
    loss = train_one_step(model_base, opt_base, i)
    warmup_loss.append(loss)
print(f"warmup 完成: 首 loss={warmup_loss[0]:.6f} 末 loss={warmup_loss[-1]:.6f}")
print(f"末 p.data[0,0] (y) = {p_data(model_base):.6f}")

# 记录 y（train mode 下的 p.data）
y_value = p_data(model_base)


# ---------------------------------------------------------------------------
# Phase 2: 验证 optimizer_eval_mode roundtrip（dev 现状 sanity check）
# ---------------------------------------------------------------------------

section("Phase 2 — optimizer_eval_mode roundtrip sanity")

print(f"进 eval mode 前 p.data[0,0] = {p_data(model_base):.6f}")
with optimizer_eval_mode(opt_base):
    averaged_x_value = p_data(model_base)
    print(f"在 eval mode 内 p.data[0,0] = {averaged_x_value:.6f} (averaged x)")
print(f"退出 eval mode 后 p.data[0,0] = {p_data(model_base):.6f}")
diff = abs(p_data(model_base) - y_value)
print(f"还原 y 偏差: {diff:.3e} {'OK' if diff < 1e-6 else 'BAD - eval/train 切换不闭合'}")

# Phase 2 的二阶检查：averaged_x ≠ y（如果相等说明 PPSF averaging 没工作 → spike 没意义）
xy_diff = abs(averaged_x_value - y_value)
print(f"averaged x vs y 差: {xy_diff:.3e} {'OK (PPSF averaging 工作中)' if xy_diff > 1e-4 else 'WARNING - averaging 没产生明显差，spike 可能没意义'}")


# ---------------------------------------------------------------------------
# Phase 3a: Ground Truth — 不重启继续训练 5 步
# ---------------------------------------------------------------------------

section(f"Phase 3a — Ground Truth: 不重启继续训练 {RESUME_STEPS} 步")

# 先 save 一份 dev 现状的 state（在 eval mode 内 save），后面 4 个 candidate 共用
tmp_dir = Path(tempfile.mkdtemp(prefix="spike_ppsf_"))
state_path_dev = tmp_dir / "state_dev.pt"
state_path_y = tmp_dir / "state_y.pt"

# state_dev: dev 现状（loop.py / context.py 现有方式）
with optimizer_eval_mode(opt_base):
    torch.save(
        {
            "model": model_base.state_dict(),  # averaged x（在 eval mode 内 .state_dict()）
            "opt": opt_base.state_dict(),
        },
        state_path_dev,
    )

# state_y: 方案 D — save 时不在 eval mode（model.state_dict() = y）
torch.save(
    {
        "model": model_base.state_dict(),  # y（不在 eval mode 内）
        "opt": opt_base.state_dict(),
    },
    state_path_y,
)

# Ground Truth: 用 model_base 继续训练 RESUME_STEPS 步
ground_loss = []
ground_pdata = []
print(f"  跑 {RESUME_STEPS} 步，每 {TREND_SAMPLE_EVERY} 步采样打印一次:")
for i in range(RESUME_STEPS):
    step_idx = WARMUP_STEPS + i
    loss = train_one_step(model_base, opt_base, step_idx)
    ground_loss.append(loss)
    ground_pdata.append(p_data(model_base))
    if (i + 1) % TREND_SAMPLE_EVERY == 0 or i == 0:
        print(f"    step={step_idx + 1:5d} loss={loss:.6f} p.data[0,0]={p_data(model_base):.6f}")


# ---------------------------------------------------------------------------
# Phase 3b: 4 个 candidate 方案
# ---------------------------------------------------------------------------

def load_and_run(state_path: Path, post_load_action: str) -> tuple[list[float], list[float], float, str]:
    """从 state_path 重启，按 post_load_action 做恢复操作，跑 RESUME_STEPS 步。

    返回 (loss_list, pdata_list, pdata_immediately_after_load, error_str)。
    error_str 为空 = 正常跑完；非空 = 抛异常（错误消息），losses/pdatas 含成功的部分。
    """
    m, o = make_model_opt()
    if hasattr(o, "train"):
        o.train()  # 跟 model_base 起始状态对齐
    state = torch.load(state_path, map_location="cpu", weights_only=False)
    m.load_state_dict(state["model"])
    o.load_state_dict(state["opt"])
    pdata_at_load = p_data(m)

    if post_load_action == "noop":
        pass
    elif post_load_action == "train_once":
        if hasattr(o, "train"):
            o.train()
    elif post_load_action == "eval_then_train":
        if hasattr(o, "eval"):
            o.eval()
        if hasattr(o, "train"):
            o.train()
    else:
        raise ValueError(f"unknown action: {post_load_action}")

    losses, pdatas, err = [], [], ""
    for i in range(RESUME_STEPS):
        step_idx = WARMUP_STEPS + i
        try:
            loss = train_one_step(m, o, step_idx)
        except Exception as e:
            err = f"{type(e).__name__}: {e}"
            break
        losses.append(loss)
        pdatas.append(p_data(m))
    return losses, pdatas, pdata_at_load, err


def run_phase(name: str, state_path: Path, action: str):
    section(name)
    losses, pdatas, atload, err = load_and_run(state_path, action)
    print(f"  load 后 p.data[0,0] = {atload:.6f} (期待 y={y_value:.6f}, averaged x={averaged_x_value:.6f})")
    if err:
        print(f"  *** 抛异常: {err}")
        print(f"  *** 成功跑了 {len(losses)} 步")
        return losses, pdatas, atload, err
    print(f"  跑 {len(losses)} 步，每 {TREND_SAMPLE_EVERY} 步采样:")
    for i, (l, p) in enumerate(zip(losses, pdatas)):
        if (i + 1) % TREND_SAMPLE_EVERY == 0 or i == 0:
            print(f"    step={WARMUP_STEPS + i + 1:5d} loss={l:.6f} p.data[0,0]={p:.6f}")
    return losses, pdatas, atload, err


c0_loss, c0_pdata, c0_atload, c0_err = run_phase(
    "Phase 3b — 方案 0 (dev 现状): load state_dev → noop",
    state_path_dev, "noop",
)
cA_loss, cA_pdata, cA_atload, cA_err = run_phase(
    "Phase 3c — 方案 A: load state_dev → optimizer.train()",
    state_path_dev, "train_once",
)
cB_loss, cB_pdata, cB_atload, cB_err = run_phase(
    "Phase 3d — 方案 B: load state_dev → optimizer.eval(); optimizer.train()",
    state_path_dev, "eval_then_train",
)
cD_loss, cD_pdata, cD_atload, cD_err = run_phase(
    "Phase 3e — 方案 D: load state_y (save 不在 eval mode) → optimizer.train()",
    state_path_y, "train_once",
)


# ---------------------------------------------------------------------------
# Phase 4: 自动判定
# ---------------------------------------------------------------------------

section("SPIKE REPORT — candidate vs ground truth")


def max_abs_dev(cand: list[float], ground: list[float]) -> float:
    if not cand:
        return float("inf")
    return max(abs(c - g) for c, g in zip(cand, ground))


def cumulative_max_dev_at(cand: list[float], ground: list[float], n: int) -> float:
    """前 n 步的 cumulative max abs dev。"""
    if not cand or n <= 0:
        return float("inf")
    n = min(n, len(cand), len(ground))
    return max(abs(c - g) for c, g in zip(cand[:n], ground[:n]))


def linear_extrapolate(losses_cum_dev: list[tuple[int, float]], target_step: int) -> float:
    """简单线性外推：用最后两个 milestone 估 slope，外推到 target_step。

    如果 slope < 0 或者发散变快（最后两段不单调），返回 NaN 提示非线性。
    """
    if len(losses_cum_dev) < 2:
        return float("nan")
    # 用最后两个采样点估 slope
    (s1, d1), (s2, d2) = losses_cum_dev[-2], losses_cum_dev[-1]
    if s2 <= s1:
        return float("nan")
    slope = (d2 - d1) / (s2 - s1)
    return d2 + slope * (target_step - s2)


def report_trend(name: str, c_loss: list[float], c_pdata: list[float], err: str) -> None:
    if err:
        print(f"\n  {name}: [THROWS] {err}")
        return
    milestones = [m for m in (1, 100, 500, 1000, 2000, 5000) if m <= len(c_loss)]
    if len(c_loss) not in milestones:
        milestones.append(len(c_loss))
    loss_traj = [(m, cumulative_max_dev_at(c_loss, ground_loss, m)) for m in milestones]
    pdata_traj = [(m, cumulative_max_dev_at(c_pdata, ground_pdata, m)) for m in milestones]

    print(f"\n  {name} 累积 max abs dev:")
    print(f"    {'step':>6} | {'loss_dev':>12} | {'pdata_dev':>12}")
    for (s, ld), (_, pd) in zip(loss_traj, pdata_traj):
        print(f"    {s:>6} | {ld:>12.3e} | {pd:>12.3e}")

    # 线性外推到 10000 步
    pred_loss_10k = linear_extrapolate(loss_traj, 10000)
    pred_pdata_10k = linear_extrapolate(pdata_traj, 10000)
    print(
        f"    linear extrapolate -> 10000 步: "
        f"loss_dev≈{pred_loss_10k:.3e}, p.data_dev≈{pred_pdata_10k:.3e}"
    )
    # 判定趋势：取最后两段 slope 比对最后第三/二段
    if len(loss_traj) >= 3:
        (s1, d1), (s2, d2), (s3, d3) = loss_traj[-3], loss_traj[-2], loss_traj[-1]
        slope_early = (d2 - d1) / max(s2 - s1, 1)
        slope_late = (d3 - d2) / max(s3 - s2, 1)
        ratio = slope_late / slope_early if abs(slope_early) > 1e-15 else float("nan")
        if abs(ratio - 1.0) < 0.5:
            shape = "linear (slope 稳定，1w 步可线性外推)"
        elif ratio > 1.5:
            shape = f"super-linear (slope 加速 {ratio:.1f}×，1w 步可能爆)"
        elif ratio < 0.5:
            shape = f"sub-linear / bounded (slope 减速 {ratio:.1f}×，drift 可能收敛到上界)"
        else:
            shape = f"unclear (slope ratio {ratio:.1f})"
        print(f"    增长形态: {shape}")


def report_one(name: str, atload: float, losses: list[float], pdatas: list[float], err: str) -> dict:
    pdata_at_load_dev = abs(atload - y_value)
    if err:
        print(
            f"\n  {name}:"
            f"\n    p.data 装载后偏离 y          : {pdata_at_load_dev:.3e}"
            f"\n    跑步数                       : {len(losses)}/{RESUME_STEPS}"
            f"\n    异常                         : {err}"
            f"\n    judgement                    : THROWS - 不可用"
        )
        return {"name": name, "perfect": False, "throws": True, "err": err,
                "loss_dev": float("inf"), "pdata_dev": float("inf"),
                "pdata_at_load_dev": pdata_at_load_dev}
    loss_dev = max_abs_dev(losses, ground_loss)
    pdata_dev = max_abs_dev(pdatas, ground_pdata)
    perfect = loss_dev < 1e-5 and pdata_dev < 1e-5
    print(
        f"\n  {name}:"
        f"\n    p.data 装载后偏离 y          : {pdata_at_load_dev:.3e}"
        f"\n    loss   {RESUME_STEPS}-step max abs dev   : {loss_dev:.3e}"
        f"\n    p.data {RESUME_STEPS}-step max abs dev   : {pdata_dev:.3e}"
        f"\n    judgement                    : {'PERFECT' if perfect else 'DRIFT'}"
    )
    return {"name": name, "perfect": perfect, "throws": False,
            "loss_dev": loss_dev, "pdata_dev": pdata_dev,
            "pdata_at_load_dev": pdata_at_load_dev}


results = [
    report_one("方案 0  (dev 现状: load -> noop)", c0_atload, c0_loss, c0_pdata, c0_err),
    report_one("方案 A  (load -> opt.train())", cA_atload, cA_loss, cA_pdata, cA_err),
    report_one("方案 B  (load -> opt.eval(); opt.train())", cB_atload, cB_loss, cB_pdata, cB_err),
    report_one("方案 D  (save 不在 eval mode -> load -> opt.train())", cD_atload, cD_loss, cD_pdata, cD_err),
]

section("DRIFT 增长趋势 (cumulative max abs dev vs ground)")
report_trend("方案 0", c0_loss, c0_pdata, c0_err)
report_trend("方案 A", cA_loss, cA_pdata, cA_err)
report_trend("方案 B", cB_loss, cB_pdata, cB_err)
report_trend("方案 D", cD_loss, cD_pdata, cD_err)


section("结论")
perfect_candidates = [r for r in results if r["perfect"]]
throws = [r for r in results if r["throws"]]
if throws:
    print("抛异常方案 (dev 用户立刻见红):")
    for r in throws:
        print(f"  - {r['name']}: {r['err']}")
if perfect_candidates:
    print(f"\n[OK] 找到 {len(perfect_candidates)} 个完美对齐 ground truth 的方案:")
    for r in perfect_candidates:
        print(f"  -> {r['name']}")
    print(
        "\n推荐用列表中第一个 (代码改动最小)。"
        "如多个方案都对齐 ground truth，再按代码改动量 / 与现行 API 一致性裁定。"
    )
    exit_code = 0
else:
    print("\n[FAIL] 没有 candidate 跟 ground truth 完美对齐。按 loss_dev 升序:")
    for r in sorted(results, key=lambda x: x["loss_dev"]):
        tag = "THROWS" if r["throws"] else "DRIFT"
        print(f"  - [{tag}] {r['name']}: loss_dev={r['loss_dev']:.3e}")
    exit_code = 1


# Cleanup
try:
    state_path_dev.unlink()
    state_path_y.unlink()
    tmp_dir.rmdir()
except Exception:
    pass

sys.exit(exit_code)
