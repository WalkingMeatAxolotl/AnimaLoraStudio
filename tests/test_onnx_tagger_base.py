"""PR-3 — OnnxTaggerBase CPU fallback：推理期 CUDA 错误自动降 CPU 重试。

session 创建期的 CUDA fallback 已有 _create_session 覆盖；这里专门测
推理期（_session.run 抛 cuBLAS / CUDA 错）的降级路径。
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
from PIL import Image

from studio.services.tagging import onnx_base as onnx_tagger_base
from studio.services.tagging.onnx_base import OnnxTaggerBase


class _StubTagger(OnnxTaggerBase):
    """最小 tagger 实现，绕过文件系统 + ONNX：单图固定 shape，单 logit 直通。"""
    name = "stub"

    def __init__(self, batch: int = 2) -> None:
        super().__init__()
        self._batch = batch

    def prepare(self) -> None:  # session 由测试直接注入，prepare 不会被调
        raise AssertionError("prepare() should not be called in tests")

    def _preprocess(self, img: Image.Image) -> np.ndarray:
        return np.zeros((4, 4, 3), dtype=np.float32)

    def _postprocess_one(self, logits: np.ndarray):
        return [f"score={float(logits[0]):.2f}"], {"score": float(logits[0])}

    def _get_batch_size_cfg(self) -> int:
        return self._batch


# ---------------------------------------------------------------------------
# _is_cuda_inference_error
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("msg", [
    "CUBLAS_STATUS_INVALID_VALUE",
    "CUBLAS_STATUS_EXECUTION_FAILED while running ConvBatch",
    "CUDNN_STATUS_BAD_PARAM",
    "CUDA error: invalid device function",
    "CUDAExecutionProvider failed to bind input",
    "out of memory: tried to allocate 12 GiB",
    "OOM at layer x",
])
def test_is_cuda_inference_error_recognizes_cuda_failures(msg: str) -> None:
    assert OnnxTaggerBase._is_cuda_inference_error(RuntimeError(msg)) is True


@pytest.mark.parametrize("msg", [
    "model file truncated",
    "tensor shape mismatch [3,4] vs [4,3]",
    "permission denied",
])
def test_is_cuda_inference_error_ignores_unrelated_failures(msg: str) -> None:
    assert OnnxTaggerBase._is_cuda_inference_error(RuntimeError(msg)) is False


# ---------------------------------------------------------------------------
# _create_session — CUDA EP 静默降级检测
# ---------------------------------------------------------------------------


def _make_fake_session(providers: list[str]):
    """构造一个 mock onnxruntime session：input/output name + 指定 providers。"""
    sess = MagicMock()
    sess.get_inputs.return_value = [MagicMock()]
    sess.get_inputs.return_value[0].name = "x"
    sess.get_outputs.return_value = [MagicMock()]
    sess.get_outputs.return_value[0].name = "y"
    sess.get_providers.return_value = list(providers)
    return sess


def test_create_session_records_error_on_silent_cuda_downgrade(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """onnxruntime 在 CUDA dlopen 失败时不抛而是 silently 降 CPU；
    `_create_session` 必须比对实际 providers 并 stash 错误，否则 UI 看不到。"""
    t = _StubTagger()

    fake_session = _make_fake_session(["CPUExecutionProvider"])  # 装的是 GPU，但实际只剩 CPU
    requested_providers: list[list[str]] = []

    def fake_session_ctor(path, providers):
        requested_providers.append(list(providers))
        return fake_session

    fake_ort = MagicMock(InferenceSession=fake_session_ctor)
    fake_ort.get_available_providers.return_value = [
        "CUDAExecutionProvider",
        "CPUExecutionProvider",
    ]
    monkeypatch.setitem(__import__("sys").modules, "onnxruntime", fake_ort)
    # 起始无旧错记录
    onnx_tagger_base.onnxruntime_setup.record_cuda_load_error(None)

    try:
        t._create_session(Path("/fake/model.onnx"))
        # 用户请求过 CUDA
        assert requested_providers and "CUDAExecutionProvider" in requested_providers[0]
        # 但 onnxruntime 内部静默降 CPU → 必须有 cuda_load_error 让 UI 显示
        err = onnx_tagger_base.onnxruntime_setup.get_cuda_load_error()
        assert err is not None
        assert "静默降级" in err or "silently" in err.lower()
    finally:
        onnx_tagger_base.onnxruntime_setup.record_cuda_load_error(None)


def test_create_session_clears_error_when_cuda_actually_works(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """请求 CUDA 且实际 providers 含 CUDA → 清空旧错（成功路径）。"""
    t = _StubTagger()
    fake_session = _make_fake_session(
        ["CUDAExecutionProvider", "CPUExecutionProvider"]
    )

    fake_ort = MagicMock(InferenceSession=lambda _p, providers: fake_session)
    fake_ort.get_available_providers.return_value = [
        "CUDAExecutionProvider",
        "CPUExecutionProvider",
    ]
    monkeypatch.setitem(__import__("sys").modules, "onnxruntime", fake_ort)
    # 预置一个旧错
    onnx_tagger_base.onnxruntime_setup.record_cuda_load_error("old failure")

    try:
        t._create_session(Path("/fake/model.onnx"))
        assert onnx_tagger_base.onnxruntime_setup.get_cuda_load_error() is None
    finally:
        onnx_tagger_base.onnxruntime_setup.record_cuda_load_error(None)


def test_create_session_no_false_positive_when_cuda_not_requested(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """机器没 GPU → 根本没请求 CUDA → providers 只有 CPU 是正常的，不应记错。"""
    t = _StubTagger()
    fake_session = _make_fake_session(["CPUExecutionProvider"])

    fake_ort = MagicMock(InferenceSession=lambda _p, providers: fake_session)
    fake_ort.get_available_providers.return_value = ["CPUExecutionProvider"]
    monkeypatch.setitem(__import__("sys").modules, "onnxruntime", fake_ort)
    onnx_tagger_base.onnxruntime_setup.record_cuda_load_error(None)

    try:
        t._create_session(Path("/fake/model.onnx"))
        assert onnx_tagger_base.onnxruntime_setup.get_cuda_load_error() is None
    finally:
        onnx_tagger_base.onnxruntime_setup.record_cuda_load_error(None)


# ---------------------------------------------------------------------------
# _fallback_to_cpu_session
# ---------------------------------------------------------------------------


def test_fallback_returns_false_without_model_path() -> None:
    """没设 _model_path（异常路径，理论不该发生）→ 静默 False，不抛。"""
    t = _StubTagger()
    assert t._model_path is None
    assert t._fallback_to_cpu_session() is False


def test_fallback_creates_cpu_session(monkeypatch: pytest.MonkeyPatch) -> None:
    """有 _model_path → 用 CPUExecutionProvider 重建 session 并 stash 错误。"""
    t = _StubTagger()
    t._model_path = Path("/fake/model.onnx")

    fake_session = MagicMock()
    fake_session.get_inputs.return_value = [MagicMock(name="input")]
    fake_session.get_inputs.return_value[0].name = "input"
    fake_session.get_outputs.return_value = [MagicMock()]
    fake_session.get_outputs.return_value[0].name = "out"

    captured: dict = {}

    def fake_session_ctor(path, providers):
        captured["path"] = path
        captured["providers"] = providers
        return fake_session

    fake_ort = MagicMock(InferenceSession=fake_session_ctor)
    monkeypatch.setitem(__import__("sys").modules, "onnxruntime", fake_ort)

    # 清掉旧 stash
    onnx_tagger_base.onnxruntime_setup.record_cuda_load_error(None)
    try:
        ok = t._fallback_to_cpu_session()
        assert ok is True
        assert captured["providers"] == ["CPUExecutionProvider"]
        assert Path(captured["path"]) == Path("/fake/model.onnx")
        assert t._session is fake_session
        assert t._input_name == "input"
        assert t._output_names == ["out"]
        assert (
            "cuBLAS"
            in (onnx_tagger_base.onnxruntime_setup.get_cuda_load_error() or "")
        )
    finally:
        onnx_tagger_base.onnxruntime_setup.record_cuda_load_error(None)


def test_fallback_returns_false_when_session_ctor_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    t = _StubTagger()
    t._model_path = Path("/fake/model.onnx")

    def boom(*_a, **_k):
        raise RuntimeError("model corrupt")

    fake_ort = MagicMock(InferenceSession=boom)
    monkeypatch.setitem(__import__("sys").modules, "onnxruntime", fake_ort)
    assert t._fallback_to_cpu_session() is False


# ---------------------------------------------------------------------------
# _tag_loop CUDA 推理失败 → CPU fallback 重试
# ---------------------------------------------------------------------------


def _attach_session(tagger: _StubTagger, run_side_effect, providers=("CPUExecutionProvider",)):
    """构造一个 session 让 _tag_loop 能直接跑（绕过 prepare）。"""
    sess = MagicMock()
    sess.get_inputs.return_value = [MagicMock()]
    sess.get_inputs.return_value[0].name = "x"
    sess.get_outputs.return_value = [MagicMock()]
    sess.get_outputs.return_value[0].name = "y"
    sess.get_providers.return_value = list(providers)
    sess.run.side_effect = run_side_effect
    tagger._session = sess
    tagger._input_name = "x"
    tagger._output_names = ["y"]
    return sess


def test_inference_cuda_error_triggers_cpu_fallback_and_retries(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """CUDA error 抛出 → fallback 重建 CPU session → 用 CPU session 重试 → 成功 yield 结果。"""
    img = tmp_path / "a.png"
    Image.new("RGB", (8, 8), (255, 0, 0)).save(img)

    t = _StubTagger(batch=2)
    t._model_path = tmp_path / "fake.onnx"

    # 第一次 run 抛 cuBLAS 错；fallback 重建 session 后第二次 run 返回 logits
    fail_then_succeed = [
        RuntimeError("CUBLAS_STATUS_EXECUTION_FAILED"),
    ]
    success_logits = np.array([[0.7]], dtype=np.float32)

    cuda_sess = _attach_session(
        t, fail_then_succeed, providers=("CUDAExecutionProvider", "CPUExecutionProvider")
    )

    # fallback 重建：sys.modules["onnxruntime"].InferenceSession 返回 CPU session
    cpu_sess = MagicMock()
    cpu_sess.get_inputs.return_value = [MagicMock()]
    cpu_sess.get_inputs.return_value[0].name = "x"
    cpu_sess.get_outputs.return_value = [MagicMock()]
    cpu_sess.get_outputs.return_value[0].name = "y"
    cpu_sess.get_providers.return_value = ["CPUExecutionProvider"]
    cpu_sess.run.return_value = [success_logits]

    fake_ort = MagicMock(InferenceSession=lambda _p, providers: cpu_sess)
    monkeypatch.setitem(__import__("sys").modules, "onnxruntime", fake_ort)

    onnx_tagger_base.onnxruntime_setup.record_cuda_load_error(None)
    try:
        results = list(t.tag([img]))
        assert len(results) == 1
        # 退到 CPU session 后重试成功 → 不应有 error 字段
        assert "error" not in results[0], results[0]
        assert results[0]["tags"] == ["score=0.70"]
        # session 已切换到 CPU 实例（不是原 CUDA mock）
        assert t._session is cpu_sess
        assert "cuBLAS" in (onnx_tagger_base.onnxruntime_setup.get_cuda_load_error() or "")
    finally:
        onnx_tagger_base.onnxruntime_setup.record_cuda_load_error(None)


def test_non_cuda_inference_error_does_not_trigger_fallback(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """普通推理错（如 shape 不匹配）→ 不触发 CPU fallback，直接报错给用户。"""
    img = tmp_path / "a.png"
    Image.new("RGB", (8, 8), (0, 255, 0)).save(img)

    t = _StubTagger(batch=2)
    t._model_path = tmp_path / "fake.onnx"
    cuda_sess = _attach_session(
        t,
        [RuntimeError("tensor shape mismatch [3,4] vs [4,3]")],
        providers=("CUDAExecutionProvider", "CPUExecutionProvider"),
    )

    # 探针：fallback 路径不应被触发 → InferenceSession 不应被调
    fallback_called: list[bool] = []

    def _explode(*_a, **_k):
        fallback_called.append(True)
        raise AssertionError("CPU fallback shouldn't have been triggered")

    fake_ort = MagicMock(InferenceSession=_explode)
    monkeypatch.setitem(__import__("sys").modules, "onnxruntime", fake_ort)

    results = list(t.tag([img]))
    assert fallback_called == []
    assert len(results) == 1
    assert "error" in results[0]
    assert "shape mismatch" in results[0]["error"]


def test_fallback_failure_propagates_original_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """CUDA 错 + fallback 重建 session 也挂 → 用户看到第一次（更有诊断价值的）错误。"""
    img = tmp_path / "a.png"
    Image.new("RGB", (8, 8), (0, 0, 255)).save(img)

    t = _StubTagger(batch=2)
    t._model_path = tmp_path / "fake.onnx"
    _attach_session(
        t,
        [RuntimeError("CUDNN_STATUS_BAD_PARAM")],
        providers=("CUDAExecutionProvider", "CPUExecutionProvider"),
    )

    def boom(*_a, **_k):
        raise RuntimeError("model file truncated")

    fake_ort = MagicMock(InferenceSession=boom)
    monkeypatch.setitem(__import__("sys").modules, "onnxruntime", fake_ort)

    results = list(t.tag([img]))
    assert "error" in results[0]
    # 报原始 CUDA 错（不是 fallback 失败的 model corrupt）
    assert "CUDNN" in results[0]["error"]
