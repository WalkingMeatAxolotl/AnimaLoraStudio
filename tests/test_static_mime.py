"""issue #228 regression: studio.api.static must force .js / .css / .svg etc.
to canonical MIME types so Windows registry pollution (e.g. .js → text/plain
set by IIS / 杀软 / 旧装机) can't break ES module loading and white-screen the SPA.
"""
from __future__ import annotations

import importlib
import mimetypes


def test_static_module_overrides_polluted_js_mime() -> None:
    mimetypes.add_type("text/plain", ".js")
    assert mimetypes.guess_type("foo.js")[0] == "text/plain"

    import studio.api.static as static_mod
    importlib.reload(static_mod)

    assert mimetypes.guess_type("foo.js")[0] == "application/javascript"
    assert mimetypes.guess_type("foo.mjs")[0] == "application/javascript"
    assert mimetypes.guess_type("foo.css")[0] == "text/css"
    assert mimetypes.guess_type("foo.svg")[0] == "image/svg+xml"
    assert mimetypes.guess_type("foo.json")[0] == "application/json"
