"""NaViT 文本编码回归（0.20.0 t5_attn NameError）。

PR #407 把文本编码下沉 ``AnimaFamily.encode_text_for_batch``（只返回 opaque
cross）后，loop.py 的 NaViT 分支仍引用旧局部变量 ``t5_attn`` → 所有
``navit_packing=true`` 用户训练第一步 NameError 必崩。CI 无 GPU 跑不到该
分支，故两层防线：

1. ``encode_text_for_batch(return_t5_attn=True)`` 的族私有契约单测；
2. ``symtable`` 静态扫描 runtime/training/ 全部函数作用域——引用了但在任何
   作用域链上都无绑定的名字（即 pyflakes F821 类错误）直接测试失败。分支
   门控代码路径不需要执行也能被抓住。
"""
from __future__ import annotations

import builtins
import symtable
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from training.families.anima.family import AnimaFamily


# ── 1. return_t5_attn 契约 ──────────────────────────────────────────────


def _stub_text_encoding(monkeypatch, attn: "torch.Tensor") -> None:
    import training.families.anima.text_encoding as te

    B, L = attn.shape
    monkeypatch.setattr(
        te, "encode_qwen", lambda model, tok, texts, device: (torch.zeros(B, 8, 4), None)
    )
    monkeypatch.setattr(
        te,
        "tokenize_t5_comfy_literal",
        lambda tok, captions, max_length: (
            torch.zeros(B, L, dtype=torch.long),
            attn,
            torch.ones(B, L),
        ),
    )


def _stub_dit(max_len: int):
    return SimpleNamespace(
        preprocess_text_embeds=lambda qwen_emb, t5_ids, t5xxl_weights: torch.zeros(
            t5_ids.shape[0], max_len, 4
        )
    )


def test_encode_text_for_batch_returns_t5_attn_for_navit(monkeypatch) -> None:
    family = AnimaFamily()
    max_len = family.spec.text.max_seq_len
    attn = torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]], dtype=torch.long)
    _stub_text_encoding(monkeypatch, attn)

    result = family.encode_text_for_batch(
        (None, None, None), _stub_dit(max_len), ["a", "b"], "cpu", torch.float32,
        return_t5_attn=True,
    )
    assert isinstance(result, tuple) and len(result) == 2
    cross, t5_attn = result
    assert cross.shape[1] == max_len
    torch.testing.assert_close(t5_attn, attn)


def test_encode_text_for_batch_default_stays_bare_cross(monkeypatch) -> None:
    family = AnimaFamily()
    max_len = family.spec.text.max_seq_len
    _stub_text_encoding(monkeypatch, torch.ones(2, 4, dtype=torch.long))

    cross = family.encode_text_for_batch(
        (None, None, None), _stub_dit(max_len), ["a", "b"], "cpu", torch.float32,
    )
    assert isinstance(cross, torch.Tensor)


# ── 2. symtable 未定义名静态扫描 ────────────────────────────────────────

_TRAINING_ROOT = Path(__file__).resolve().parents[1] / "runtime" / "training"

_MODULE_DUNDERS = {
    "__name__", "__file__", "__doc__", "__package__", "__spec__",
    "__loader__", "__builtins__", "__debug__", "__annotations__",
    "__dict__", "__class__", "__module__", "__qualname__",
}


def _module_bindings(table: symtable.SymbolTable) -> set[str]:
    names = {
        sym.get_name()
        for sym in table.get_symbols()
        if sym.is_assigned() or sym.is_imported()
    }
    names |= {child.get_name() for child in table.get_children()}
    return names


def _walk_undefined(table: symtable.SymbolTable, module_names: set[str]) -> list[str]:
    bad = []
    if table.get_type() != "module":
        for sym in table.get_symbols():
            if not (sym.is_referenced() and sym.is_global()):
                continue
            name = sym.get_name()
            if name in module_names or name in _MODULE_DUNDERS:
                continue
            if hasattr(builtins, name):
                continue
            bad.append(f"{table.get_name()}:{table.get_lineno()} 引用未定义名 {name!r}")
    for child in table.get_children():
        bad.extend(_walk_undefined(child, module_names))
    return bad


@pytest.mark.parametrize(
    "py_file",
    sorted(_TRAINING_ROOT.rglob("*.py")),
    ids=lambda p: str(p.relative_to(_TRAINING_ROOT)),
)
def test_training_package_has_no_undefined_names(py_file: Path) -> None:
    """函数作用域里 LOAD_GLOBAL 的名字必须解析到模块级绑定或 builtin。

    抓的正是 0.20.0 ``t5_attn`` 这类"重构后残留引用、分支门控测试跑不到"
    的 NameError。flow-insensitive（同 pyflakes）：模块级晚绑定不误报。
    """
    source = py_file.read_text(encoding="utf-8")
    table = symtable.symtable(source, str(py_file), "exec")
    bad = _walk_undefined(table, _module_bindings(table))
    assert not bad, "\n".join(f"{py_file.name}: {b}" for b in bad)
