"""load_text_encoders 的 T5 tokenizer 在线 fallback 错误信息回归测试。

背景（任务 #363 截图）：t5_tokenizer_path 本地目录缺失时会静默从 Hugging Face
在线拉 google/t5-v1_1-xxl，网络超时曾以裸 httpx.ConnectTimeout 堆栈崩训练，
用户完全看不出是缺模型。现在要求：
    - fallback 前打「本地目录缺失 → 开始下载」的 warning
    - 下载失败必须 raise RuntimeError：含「下载失败」+ 原始原因 + 怎么办
      （任务详情的错误字段只显示 stderr 尾部，原因必须进 message 而不能只在
      异常链的上游 traceback 里）

transformers 用假模块顶替（sys.modules），不真加载权重、不联网。
"""
from __future__ import annotations

import logging
import sys
import types

import pytest


class _SelfReturningModel:
    """顶替 AutoModelForCausalLM 实例，兼容 .to(...).eval().requires_grad_(False) 链。"""

    def to(self, *args, **kwargs):
        return self

    def eval(self):
        return self

    def requires_grad_(self, *args, **kwargs):
        return self


def _make_fake_transformers(t5_from_pretrained):
    """返回 (假 transformers 模块, T5 from_pretrained 收到的 path 列表)。"""
    mod = types.ModuleType("transformers")
    t5_calls: list[str] = []

    class AutoTokenizer:
        @staticmethod
        def from_pretrained(path, **kwargs):
            return object()

    class AutoModelForCausalLM:
        @staticmethod
        def from_pretrained(path, **kwargs):
            return _SelfReturningModel()

    class T5Tokenizer:
        @staticmethod
        def from_pretrained(path, **kwargs):
            t5_calls.append(str(path))
            return t5_from_pretrained(str(path))

    class T5TokenizerFast(T5Tokenizer):
        pass

    mod.AutoTokenizer = AutoTokenizer
    mod.AutoModelForCausalLM = AutoModelForCausalLM
    mod.T5Tokenizer = T5Tokenizer
    mod.T5TokenizerFast = T5TokenizerFast
    return mod, t5_calls


@pytest.fixture()
def tm():
    from training.families.anima import loader as models  # noqa: PLC0415  (load_text_encoders 已迁 families，多模型 PR-2b)

    return models


def test_missing_t5_dir_download_error_is_wrapped(tm, monkeypatch, tmp_path, caplog):
    """目录缺失 + 下载抛网络错 → RuntimeError 带下载失败原因与修复指引。"""

    def _boom(path):
        raise ConnectionError("[WinError 10060] 由于连接方在一段时间后没有正确答复")

    fake, t5_calls = _make_fake_transformers(_boom)
    monkeypatch.setitem(sys.modules, "transformers", fake)

    missing = tmp_path / "no_such_dir"
    with caplog.at_level(logging.WARNING):
        with pytest.raises(RuntimeError) as excinfo:
            tm.load_text_encoders(str(tmp_path), str(missing), "cpu", None)

    msg = str(excinfo.value)
    assert "T5 tokenizer 下载失败" in msg
    assert "google/t5-v1_1-xxl" in msg
    assert "10060" in msg  # 原始原因必须进 message（错误字段只显示 stderr 尾部）
    assert "t5_tokenizer_path" in msg  # 指引用户检查配置
    assert isinstance(excinfo.value.__cause__, ConnectionError)
    assert t5_calls == ["google/t5-v1_1-xxl"]
    assert any("本地目录缺失" in r.getMessage() for r in caplog.records)


def test_missing_t5_dir_fallback_success_logs_warning(tm, monkeypatch, tmp_path, caplog):
    """目录缺失但下载成功 → 有「缺失 → 开始下载」warning，正常返回。"""
    fake, t5_calls = _make_fake_transformers(lambda path: "tok")
    monkeypatch.setitem(sys.modules, "transformers", fake)

    with caplog.at_level(logging.WARNING):
        _, _, t5_tokenizer = tm.load_text_encoders(
            str(tmp_path), str(tmp_path / "no_such_dir"), "cpu", None
        )

    assert t5_tokenizer == "tok"
    assert t5_calls == ["google/t5-v1_1-xxl"]
    warnings = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]
    assert any("本地目录缺失" in m and "google/t5-v1_1-xxl" in m for m in warnings)


def test_local_t5_dir_never_falls_back(tm, monkeypatch, tmp_path, caplog):
    """本地目录存在 → 只用本地路径，不联网、不打缺失 warning。"""
    fake, t5_calls = _make_fake_transformers(lambda path: "tok")
    monkeypatch.setitem(sys.modules, "transformers", fake)

    local = tmp_path / "t5_tokenizer"
    local.mkdir()
    with caplog.at_level(logging.WARNING):
        _, _, t5_tokenizer = tm.load_text_encoders(str(tmp_path), str(local), "cpu", None)

    assert t5_tokenizer == "tok"
    assert t5_calls == [str(local)]
    assert not any("本地目录缺失" in r.getMessage() for r in caplog.records)
