"""Stage 1 probe — 验证 lycoris-lora 与 Anima DiT 命中层一致性。

不加载真模型（CPU 上太慢），改为构造一个微缩 mock DiT 复现 Anima 的层命名结构，
对比当前 LoRAInjector 与 lycoris LycorisNetwork 的命中集合。

运行：
    venv\Scripts\python.exe scripts\probe_lycoris_anima.py

预期输出：
    - LoRAInjector 命中层数（mock 上约 12-20 层）
    - lycoris 命中层数
    - 两者层名 diff 应为空
    - lycoris 输出键名预览（验证 lora_prefix 与 ComfyUI 兼容）
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


# --------------------------------------------------------------------- mock model
class MockAttn(nn.Module):
    """模拟 Anima DiT 的 attention 子模块命名"""
    def __init__(self, d=64):
        super().__init__()
        self.q_proj = nn.Linear(d, d, bias=False)
        self.k_proj = nn.Linear(d, d, bias=False)
        self.v_proj = nn.Linear(d, d, bias=False)
        self.output_proj = nn.Linear(d, d, bias=False)


class MockMLP(nn.Module):
    def __init__(self, d=64):
        super().__init__()
        self.layer1 = nn.Linear(d, d * 2)
        self.layer2 = nn.Linear(d * 2, d)


class MockDiTBlock(nn.Module):
    def __init__(self, d=64):
        super().__init__()
        self.self_attn = MockAttn(d)
        self.cross_attn = MockAttn(d)
        self.mlp = MockMLP(d)
        # 与 Anima 一致：norm 层不参与 LoRA
        self.norm1 = nn.LayerNorm(d)
        self.norm2 = nn.LayerNorm(d)


class MockLLMAdapter(nn.Module):
    """这部分必须被排除（与 anima_train.py:986 一致）"""
    def __init__(self, d=64):
        super().__init__()
        self.layers = nn.ModuleList([
            MockAttn(d),  # llm_adapter.layers.0.q_proj 等不能命中
            MockAttn(d),
        ])


class MockAnimaDiT(nn.Module):
    def __init__(self, n_blocks=4, d=64):
        super().__init__()
        self.blocks = nn.ModuleList([MockDiTBlock(d) for _ in range(n_blocks)])
        self.llm_adapter = MockLLMAdapter(d)


# --------------------------------------------------------------------- baseline
def baseline_targets(model: nn.Module) -> set[str]:
    """复现 anima_train.py:LoRAInjector.inject 的命中规则"""
    targets = ["q_proj", "k_proj", "v_proj", "output_proj", "mlp.layer1", "mlp.layer2"]
    hits = set()
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if not any(t in name for t in targets):
            continue
        if "llm_adapter" in name:
            continue
        hits.add(name)
    return hits


# --------------------------------------------------------------------- lycoris
def lycoris_targets(model: nn.Module) -> tuple[set[str], object]:
    """通过反向查 lora.org_module 拿到真实 named_modules 路径，避免 _→. 还原歧义"""
    from lycoris import LycorisNetwork

    from utils.lokr_preset import apply as apply_anima_preset

    apply_anima_preset(LycorisNetwork)
    network = LycorisNetwork(
        model,
        multiplier=1.0,
        lora_dim=4,
        alpha=4,
        network_module="lokr",
        factor=8,
    )
    network.apply_to()

    # 建立 module → name 反查表（注入前用 model 自己的 named_modules，注入后通过 weakref 拿原始路径）
    name_by_id: dict[int, str] = {id(m): n for n, m in model.named_modules()}

    hits = set()
    for lora in network.loras:
        org = getattr(lora, "org_module", None)
        if org is None:
            continue
        # org_module 在 lycoris 里通常是单元素 list/tuple
        if isinstance(org, (list, tuple)):
            org = org[0]
        path = name_by_id.get(id(org))
        if path is None:
            # 兜底：用 lora_name 的下划线还原
            ln = lora.lora_name
            if ln.startswith("lora_unet_"):
                ln = ln[len("lora_unet_"):]
            path = ln
        hits.add(path)
    return hits, network


# --------------------------------------------------------------------- main
def main():
    print("=" * 60)
    print("Stage 1 probe: lycoris-lora vs LoRAInjector 命中对齐")
    print("=" * 60)

    model = MockAnimaDiT(n_blocks=4, d=64)
    total_linear = sum(1 for _ in model.named_modules() if isinstance(_[1], nn.Linear))
    print(f"\nMock DiT 总 Linear 层数: {total_linear}")

    base = baseline_targets(model)
    print(f"\n[baseline] LoRAInjector 命中: {len(base)}")
    for n in sorted(base)[:5]:
        print(f"    {n}")
    if len(base) > 5:
        print(f"    ... +{len(base) - 5} more")

    # 重新构造一份模型（lycoris.apply_to 会替换层）
    model2 = MockAnimaDiT(n_blocks=4, d=64)
    lyc, network = lycoris_targets(model2)
    print(f"\n[lycoris] LycorisNetwork 命中: {len(lyc)}")
    for n in sorted(lyc)[:5]:
        print(f"    {n}")
    if len(lyc) > 5:
        print(f"    ... +{len(lyc) - 5} more")

    print("\n--- diff ---")
    only_base = base - lyc
    only_lyc = lyc - base
    print(f"baseline 独有 ({len(only_base)}): {sorted(only_base)[:5]}")
    print(f"lycoris  独有 ({len(only_lyc)}): {sorted(only_lyc)[:5]}")

    if base == lyc:
        print("\n✓ 命中集合完全一致")
    else:
        print("\n✗ 命中集合存在差异 — 需要调整 ANIMA_PRESET")

    # 验证 llm_adapter 确实被排除
    has_llm = any("llm_adapter" in n for n in lyc)
    print(f"\nllm_adapter 排除: {'✓' if not has_llm else '✗ 命中了！'}")

    # 输出键名预览（确认 ComfyUI 兼容前缀）
    print("\n--- state_dict 键名预览（前 5 个）---")
    sd = network.state_dict()
    for k in list(sd.keys())[:5]:
        print(f"    {k}")
    print(f"  总键数: {len(sd)}")

    # 最关键：算法选对了
    sample = next(iter(network.loras))
    print(f"\n--- 第一个 lora 模块类型 ---")
    print(f"    {type(sample).__name__}")
    print(f"    has lokr_w1: {hasattr(sample, 'lokr_w1')}")
    print(f"    has lokr_w2_a: {hasattr(sample, 'lokr_w2_a')}")
    print(f"    has lokr_w2_b: {hasattr(sample, 'lokr_w2_b')}")


if __name__ == "__main__":
    main()
