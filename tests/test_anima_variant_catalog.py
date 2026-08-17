"""Anima 底模目录（内置官方 + 第三方条目）、用户候选两通道统一、行级架构探测。

- 内置目录条目只承载「去哪下载、怎么称呼」（AnimaVariant），层数等架构一律
  从文件 header 探测（arch_summary）——与用户手放的文件同一条路。
- 用户候选：local（PathPicker）+ download（repo+filename 落盘）两条通道此前只有
  local 进训练页下拉 / 底模下拉，现由 registered_main_paths 统一。
"""

from __future__ import annotations

import json
import struct
from pathlib import Path
from types import SimpleNamespace

from studio.infrastructure.secrets import ModelsConfig, SourceCandidate
from studio.services.models.families import anima
from studio.services.models.families.custom_paths import (
    arch_summary,
    clear_arch_cache,
    registered_main_paths,
)


def _write_anima_like(path: Path, *, num_blocks: int) -> None:
    """只写 header 的伪 Anima checkpoint（arch_summary 只读 header）。"""
    h = {"net.x_embedder.proj.1.weight": {"dtype": "BF16", "shape": [2048, 68], "data_offsets": [0, 0]}}
    for i in range(num_blocks):
        h[f"net.blocks.{i}.self_attn.q_proj.weight"] = {"dtype": "BF16", "shape": [8, 8], "data_offsets": [0, 0]}
    raw = json.dumps(h).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(struct.pack("<Q", len(raw)) + raw)


def test_variant_table_official_and_community():
    keys = [v.key for v in anima.ANIMA_VARIANT_TABLE]
    assert keys[0] == anima.LATEST_ANIMA
    # 兼容读面与表同源
    assert list(anima.ANIMA_VARIANTS) == keys
    community = [v for v in anima.ANIMA_VARIANT_TABLE if v.group == "community"]
    assert community, "应有第三方条目"
    for v in community:
        assert v.repo != anima.ANIMA_REPO
        assert v.label and v.author
        assert anima.anima_variant_repo(v.key) == v.repo
        # 官方条目排在第三方前面（find_anima_main fallback 优先官方）
        assert keys.index(v.key) > keys.index(anima.LATEST_ANIMA)
    assert anima.anima_variant_repo("1.0") == anima.ANIMA_REPO
    assert anima.anima_variant_repo("latest") == anima.ANIMA_REPO


def test_registered_main_paths_merges_local_and_download_dedup(tmp_path):
    root = tmp_path
    local = tmp_path / "elsewhere" / "mine.safetensors"
    models_cfg = ModelsConfig(custom={"anima": [str(local)]})
    source_cfg = {
        "anima": [
            SourceCandidate(kind="download", repo="someone/repo", filename="sub/dir/third.safetensors"),
            SourceCandidate(kind="local", path=str(local)),  # 与 custom 兼容面重复 → 去重
        ],
    }
    paths = registered_main_paths(root, models_cfg, "anima", source_cfg)
    assert paths == [local, root / "diffusion_models" / "third.safetensors"]
    # 不传 source_cfg → 只有 local（不偷读全局 secrets）
    assert registered_main_paths(root, models_cfg, "anima") == [local]


def test_path_choices_lists_download_candidates_and_arch(tmp_path):
    clear_arch_cache()
    root = tmp_path
    official = anima.anima_main_target(root, anima.LATEST_ANIMA)
    _write_anima_like(official, num_blocks=28)
    third = root / "diffusion_models" / "third.safetensors"
    _write_anima_like(third, num_blocks=40)
    community_key = next(v.key for v in anima.ANIMA_VARIANT_TABLE if v.group == "community")
    community_target = anima.anima_main_target(root, community_key)
    _write_anima_like(community_target, num_blocks=40)

    source_cfg = {"anima": [
        SourceCandidate(kind="download", repo="someone/repo", filename="third.safetensors"),
        # 用户又把内置条目的落盘文件注册了一遍 → 不重复列
        SourceCandidate(kind="local", path=str(community_target)),
    ]}
    rows = anima.path_choices(root, ModelsConfig(), source_cfg)["transformer_path"]
    by_label = {r["label"]: r for r in rows}
    assert set(by_label) == {official.name, community_target.name, "third.safetensors"}
    assert by_label[official.name]["group"] == "official"
    assert by_label[official.name]["arch"]["num_blocks"] == 28
    assert by_label[community_target.name]["group"] == "community"
    assert by_label[community_target.name]["arch"]["num_blocks"] == 40
    assert by_label["third.safetensors"]["group"] == "custom"
    assert by_label["third.safetensors"]["arch"]["num_blocks"] == 40


def test_catalog_sections_rows_carry_label_group_arch(tmp_path):
    clear_arch_cache()
    root = tmp_path
    official = anima.anima_main_target(root, anima.LATEST_ANIMA)
    _write_anima_like(official, num_blocks=28)
    sec = anima.catalog_sections(root, ModelsConfig(), {})["anima_main"]
    rows = {r["variant"]: r for r in sec["variants"]}
    assert rows[anima.LATEST_ANIMA]["arch"] == {"num_blocks": 28, "model_channels": 2048, "param_count": 2048 * 68 + 28 * 64}
    assert rows[anima.LATEST_ANIMA]["group"] == "official"
    community = next(r for r in sec["variants"] if r["group"] == "community")
    assert community["label"] != community["variant"]
    assert community["arch"] is None  # 没下载 → 无架构
    assert community["exists"] is False


def test_arch_summary_none_for_missing_or_non_anima(tmp_path):
    assert arch_summary(tmp_path / "nope.safetensors") is None
    bad = tmp_path / "bad.safetensors"
    bad.write_bytes(b"garbage")
    assert arch_summary(bad) is None
