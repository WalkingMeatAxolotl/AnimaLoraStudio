from __future__ import annotations

from types import SimpleNamespace

from tools.vram_trace import classify_stage, merge_process_rows


def test_classify_krea2_task_log_stages():
    assert classify_stage(
        "INFO Loading text encoder ahead of the transformer for pre-encoding: x"
    ) == "load_te"
    assert classify_stage("INFO 加载文本编码器: x") == "load_te"
    assert classify_stage("INFO Pre-encoded 2 prompts: text encoder released") == "te_released"
    assert classify_stage("INFO 已预编码 2 条 prompt: 文本编码器已释放") == "te_released"
    assert classify_stage("INFO Loading transformer (krea2): x") == "load_dit"
    assert classify_stage("INFO 加载 Transformer（krea2）: x") == "load_dit"
    assert classify_stage("INFO loading vae x") == "load_vae"
    assert classify_stage("INFO 加载 VAE") == "load_vae"
    assert classify_stage("INFO VAE 加载完成") == "vae_ready"
    assert classify_stage(
        "INFO LoRA merged into the fp8 base model: 2 LoRA(s) into 240 layers"
    ) == "lora_merge"
    assert classify_stage(
        "INFO LoRA 已合并进 fp8 底模: 2 份 LoRA → 240 个层"
    ) == "lora_merge"
    assert classify_stage("INFO Model unloaded: 31.2 GB RAM available") == "unload_all"
    assert classify_stage("ERROR Generation task failed: task=7") == "failed"
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
