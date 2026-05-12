from __future__ import annotations

from argparse import Namespace

import pytest

pytest.importorskip("torch")

from runtime.anima_train import _config_value


def test_config_value_preserves_explicit_zero() -> None:
    args = Namespace(
        lucid_ortho_reg=0.0,
        lucid_mag_reg=0.0,
        lucid_mag_amplify=0.0,
        lucid_aux_loss_weight=0.0,
        lucid_aux_warmup_ratio=0.0,
        lucid_lora_plus_ratio=0.0,
        lucid_qk_rank_ratio=0.0,
    )

    assert _config_value(args, "lucid_ortho_reg", 0.01) == 0.0
    assert _config_value(args, "lucid_mag_reg", 0.001) == 0.0
    assert _config_value(args, "lucid_mag_amplify", 2.0) == 0.0
    assert _config_value(args, "lucid_aux_loss_weight", 1.0) == 0.0
    assert _config_value(args, "lucid_aux_warmup_ratio", 0.1) == 0.0
    assert _config_value(args, "lucid_lora_plus_ratio", 16.0) == 0.0
    assert _config_value(args, "lucid_qk_rank_ratio", 0.25) == 0.0


def test_config_value_defaults_only_on_missing_or_none() -> None:
    assert _config_value(Namespace(), "lucid_mag_reg", 0.001) == 0.001
    assert _config_value(Namespace(lucid_mag_reg=None), "lucid_mag_reg", 0.001) == 0.001
