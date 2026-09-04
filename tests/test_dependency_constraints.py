"""Dependency bounds that protect compatibility-sensitive integrations."""
from __future__ import annotations

from pathlib import Path
import re


REPO_ROOT = Path(__file__).resolve().parent.parent


def test_lycoris_stays_on_supported_v3_major() -> None:
    """Fresh installs must not silently cross LyCORIS's breaking v4 boundary."""
    requirements = (REPO_ROOT / "requirements.txt").read_text(encoding="utf-8-sig")
    match = re.search(r"(?m)^lycoris-lora([^#\r\n]*)", requirements)

    assert match is not None, "requirements.txt must declare lycoris-lora"
    specifier = match.group(1).replace(" ", "")
    assert ">=3.4.0" in specifier
    assert "<4.0" in specifier
