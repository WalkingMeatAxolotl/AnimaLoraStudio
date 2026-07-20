from __future__ import annotations

from types import SimpleNamespace

from tools.vram_trace import classify_stage, merge_process_rows


def test_classify_krea2_task_log_stages():
    assert classify_stage("INFO loading text encoders (TE 先行) x") == "load_te"
    assert classify_stage("INFO krea2 预编码 2 条 prompt；TE 已释放") == "te_released"
    assert classify_stage("INFO loading transformer [krea2] x") == "load_dit"
    assert classify_stage("INFO loading vae x") == "load_vae"
    assert classify_stage("INFO VAE 加载完成") == "vae_ready"
    assert classify_stage("INFO Krea2 fp8 merge：2 份 LoRA") == "lora_merge"
    assert classify_stage("ordinary log line") is None


def test_merge_process_rows_deduplicates_compute_and_graphics():
    compute = [SimpleNamespace(pid=10, usedGpuMemory=100), SimpleNamespace(pid=20, usedGpuMemory=50)]
    graphics = [SimpleNamespace(pid=10, usedGpuMemory=100), SimpleNamespace(pid=30, usedGpuMemory=None)]

    assert merge_process_rows([compute, graphics], total=1000) == {
        10: 100,
        20: 50,
        30: None,
    }


def test_merge_process_rows_rejects_wddm_not_available_sentinel():
    rows = [[SimpleNamespace(pid=10, usedGpuMemory=(1 << 64) - 1)]]

    assert merge_process_rows(rows, total=24 * 1024**3) == {10: None}
