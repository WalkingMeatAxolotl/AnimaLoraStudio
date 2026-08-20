"""log_messages i18n 字典：语言解析、渲染兜底、msg_id 唯一性。"""
from __future__ import annotations

import pytest

from studio.infrastructure import log_messages as lm


@pytest.fixture(autouse=True)
def _reset(monkeypatch):
    lm._reset_lang_cache_for_tests()
    monkeypatch.delenv(lm.UI_LANG_ENV, raising=False)
    yield
    lm._reset_lang_cache_for_tests()


def test_env_selects_language(monkeypatch):
    monkeypatch.setitem(lm.MESSAGES, "t.hello", {"zh": "你好 {name}", "en": "hello {name}"})
    monkeypatch.setenv(lm.UI_LANG_ENV, "en")
    assert lm.msg("t.hello", name="x") == "hello x"
    monkeypatch.setenv(lm.UI_LANG_ENV, "zh")
    assert lm.msg("t.hello", name="x") == "你好 x"


def test_invalid_env_falls_back_to_secrets_then_zh(monkeypatch):
    monkeypatch.setitem(lm.MESSAGES, "t.hello", {"zh": "中", "en": "e"})
    monkeypatch.setenv(lm.UI_LANG_ENV, "fr")
    # secrets 读失败 → zh
    monkeypatch.setattr(lm, "_secrets_lang", None)
    import studio.infrastructure.secrets as sec
    monkeypatch.setattr(sec, "load", lambda: (_ for _ in ()).throw(RuntimeError()))
    assert lm.msg("t.hello") == "中"


def test_missing_key_is_loud_but_safe():
    assert lm.msg("no.such.key") == "no.such.key"
    out = lm.msg("no.such.key", a=1)
    assert "no.such.key" in out and "1" in str(out)


def test_bad_placeholder_returns_template(monkeypatch):
    monkeypatch.setitem(lm.MESSAGES, "t.bad", {"zh": "值 {missing}"})
    assert lm.msg("t.bad", other=1) == "值 {missing}"


def test_partial_entry_falls_back_across_languages(monkeypatch):
    monkeypatch.setitem(lm.MESSAGES, "t.zh_only", {"zh": "只有中文"})
    monkeypatch.setenv(lm.UI_LANG_ENV, "en")
    assert lm.msg("t.zh_only") == "只有中文"


def test_no_duplicate_ids_across_domain_files():
    from studio.infrastructure._messages_train import MESSAGES as a
    from studio.infrastructure._messages_worker import MESSAGES as b
    from studio.infrastructure._messages_server import MESSAGES as c
    ids = list(a) + list(b) + list(c)
    assert len(ids) == len(set(ids))


def test_all_entries_bilingual():
    for mid, entry in lm.MESSAGES.items():
        assert set(entry) >= {"zh", "en"}, f"{mid} 缺语言: {sorted(entry)}"
