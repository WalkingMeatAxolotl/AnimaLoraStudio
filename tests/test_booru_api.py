"""services/booru_api.py — search_posts 头部 / 路由回归。

hotfix 背景：danbooru 挂了 Cloudflare 后，缺 User-Agent (requests 默认
`python-requests/X.Y.Z`) 会被 CF 直接 403 挑战页。这里固化两条最关键的
不变量：
- 请求时必须带可识别的 UA (含 'AnimaLoraStudio')
- Accept: application/json 让中间件路由更确定

不真发 HTTP：用 fake session 截 sess.get 参数验证。
"""
from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

from studio.services import booru_api


def _fake_session(json_payload: Any = None) -> MagicMock:
    sess = MagicMock()
    resp = MagicMock()
    resp.json.return_value = json_payload if json_payload is not None else []
    resp.raise_for_status.return_value = None
    sess.get.return_value = resp
    return sess


def test_search_posts_sends_app_user_agent_for_danbooru() -> None:
    sess = _fake_session([{"id": 1}])
    booru_api.search_posts("danbooru", "1girl", session=sess)
    _, kwargs = sess.get.call_args
    headers = kwargs.get("headers") or {}
    assert "User-Agent" in headers
    assert "AnimaLoraStudio" in headers["User-Agent"]
    assert headers.get("Accept") == "application/json"


def test_search_posts_ua_includes_username_when_provided() -> None:
    """搜索带 username 时 UA 应为 'AnimaLoraStudio/X (by username)' —
    符合 danbooru TOS 推荐格式，让 CF 端能按账户白名单而不是匿名拦截。"""
    sess = _fake_session([])
    booru_api.search_posts("danbooru", "1girl", username="alice", session=sess)
    ua = sess.get.call_args.kwargs["headers"]["User-Agent"]
    assert "AnimaLoraStudio" in ua
    assert "(by alice)" in ua


def test_search_posts_ua_falls_back_when_no_username() -> None:
    """没传 username 时 UA 不应带空括号。"""
    sess = _fake_session([])
    booru_api.search_posts("danbooru", "1girl", session=sess)
    ua = sess.get.call_args.kwargs["headers"]["User-Agent"]
    assert "(by" not in ua
    assert "AnimaLoraStudio/" in ua


def test_search_posts_sends_app_user_agent_for_gelbooru() -> None:
    """gelbooru 没那么严，但同样应带 UA 以符合礼貌使用。"""
    sess = _fake_session({"post": [{"id": 2}]})
    booru_api.search_posts("gelbooru", "1girl", session=sess)
    _, kwargs = sess.get.call_args
    headers = kwargs.get("headers") or {}
    assert "AnimaLoraStudio" in headers["User-Agent"]


def test_build_user_agent_strips_whitespace() -> None:
    """username 含前后空格 / 全空格时不应生成 '(by   )' 这种空 UA 段。"""
    assert "(by" not in booru_api._build_user_agent("   ")
    assert booru_api._build_user_agent("  alice  ") == \
        f"{booru_api._USER_AGENT_BASE} (by alice)"


def test_search_posts_user_agent_does_not_impersonate_browser() -> None:
    """回归：之前曾用 Chrome 浏览器 UA 反而被 Cloudflare 当作"浏览器但
    不跑 JS"识破并 403。UA 必须明确说是 AnimaLoraStudio。"""
    sess = _fake_session([])
    booru_api.search_posts("danbooru", "x", session=sess)
    ua = sess.get.call_args.kwargs["headers"]["User-Agent"]
    assert "Mozilla" not in ua
    assert "Chrome" not in ua


def test_search_posts_routes_danbooru_correctly() -> None:
    sess = _fake_session([])
    booru_api.search_posts("danbooru", "1girl rating:safe", page=2, limit=50, session=sess)
    args, kwargs = sess.get.call_args
    url = args[0]
    params = kwargs["params"]
    assert url.endswith("/posts.json")
    assert params["tags"] == "1girl rating:safe"
    assert params["page"] == 2
    assert params["limit"] == 50


def test_search_posts_routes_gelbooru_correctly() -> None:
    sess = _fake_session({"post": []})
    booru_api.search_posts("gelbooru", "tag1", page=3, limit=100, session=sess)
    args, kwargs = sess.get.call_args
    url = args[0]
    params = kwargs["params"]
    assert url.endswith("/index.php")
    assert params["page"] == "dapi"
    assert params["pid"] == 2  # page-1 for gelbooru
    assert params["limit"] == 100


def test_search_posts_passes_basic_auth_for_danbooru_when_provided() -> None:
    sess = _fake_session([])
    booru_api.search_posts(
        "danbooru", "x",
        username="me", api_key="secret",
        session=sess,
    )
    _, kwargs = sess.get.call_args
    assert kwargs["auth"] == ("me", "secret")


def test_search_posts_no_auth_when_username_missing() -> None:
    sess = _fake_session([])
    booru_api.search_posts("danbooru", "x", api_key="secret", session=sess)
    assert sess.get.call_args.kwargs["auth"] is None


def test_download_image_ua_includes_username(tmp_path) -> None:
    """download 路径也走 CF；UA 同样要带 username 让账户白名单生效。"""
    from io import BytesIO
    from PIL import Image as _PILImage
    buf = BytesIO()
    _PILImage.new("RGB", (4, 4), (255, 0, 0)).save(buf, "PNG")
    png_bytes = buf.getvalue()

    sess = MagicMock()
    resp = MagicMock()
    resp.content = png_bytes
    resp.raise_for_status.return_value = None
    sess.get.return_value = resp

    booru_api.download_image(
        "https://cdn.donmai.us/x.png",
        tmp_path / "x.png",
        convert_to_png=False,
        remove_alpha_channel=False,
        username="alice",
        session=sess,
    )
    headers = sess.get.call_args.kwargs["headers"]
    assert "(by alice)" in headers["User-Agent"]
