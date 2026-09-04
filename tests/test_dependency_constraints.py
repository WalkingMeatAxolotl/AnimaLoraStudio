"""Dependency bounds that protect compatibility-sensitive integrations."""
from __future__ import annotations

from pathlib import Path
import re


REPO_ROOT = Path(__file__).resolve().parent.parent


def test_lycoris_is_pinned_to_audited_v4_artifact() -> None:
    """Fresh installs must resolve the exact LyCORIS artifact we validated."""
    requirements = (REPO_ROOT / "requirements.txt").read_text(encoding="utf-8-sig")
    match = re.search(r"(?m)^lycoris-lora([^#\r\n]*)", requirements)

    assert match is not None, "requirements.txt must declare lycoris-lora"
    specifier = match.group(1).replace(" ", "")
    assert specifier == "==4.0.0"
