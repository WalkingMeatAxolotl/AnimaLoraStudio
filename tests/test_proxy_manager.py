"""proxy_manager.get_proxy_dict 回退逻辑测试（issue #357 代理修复）。

背景：非 socks5 分支旧代码在 `https_proxy` 留空时不设 proxies["https"]，导致所有
booru（全 https）请求直连 → 超时；且与设置页 tips3「HTTPS 留空回退用 HTTP 代理」
的承诺不符。修复后任一字段留空回退到另一个，两字段都填时各自尊重。
"""
from __future__ import annotations

import socket
import threading

import requests
import urllib3

from studio.infrastructure import secrets as secrets_mod
from studio.services import proxy_manager


def _use_proxy(monkeypatch, **kw):
    """让 get_proxy_dict 读到一份指定的 ProxyConfig。"""
    s = secrets_mod.Secrets(proxy=secrets_mod.ProxyConfig(**kw))
    monkeypatch.setattr(proxy_manager.secrets, "load", lambda: s)


def test_disabled_returns_none(monkeypatch):
    _use_proxy(monkeypatch, enabled=False, http_proxy="http://127.0.0.1:2333")
    assert proxy_manager.get_proxy_dict() is None


def test_empty_when_enabled_but_no_addresses(monkeypatch):
    _use_proxy(monkeypatch, enabled=True)
    assert proxy_manager.get_proxy_dict() is None


def test_http_only_falls_back_to_https(monkeypatch):
    """核心回归：只填 HTTP 代理，https 目标流量也必须走代理（否则直连超时）。"""
    _use_proxy(monkeypatch, enabled=True, http_proxy="http://127.0.0.1:2333")
    assert proxy_manager.get_proxy_dict() == {
        "http": "http://127.0.0.1:2333",
        "https": "http://127.0.0.1:2333",
    }


def test_https_only_falls_back_to_http(monkeypatch):
    _use_proxy(monkeypatch, enabled=True, https_proxy="http://127.0.0.1:2333")
    assert proxy_manager.get_proxy_dict() == {
        "http": "http://127.0.0.1:2333",
        "https": "http://127.0.0.1:2333",
    }


def test_both_explicit_are_respected(monkeypatch):
    """两字段都显式填写时各自保留，不互相覆盖。"""
    _use_proxy(
        monkeypatch,
        enabled=True,
        http_proxy="http://127.0.0.1:2333",
        https_proxy="http://127.0.0.1:9999",
    )
    assert proxy_manager.get_proxy_dict() == {
        "http": "http://127.0.0.1:2333",
        "https": "http://127.0.0.1:9999",
    }


def test_socks5_single_field_covers_both(monkeypatch):
    """socks5 只填一个字段即覆盖两种目标流量（吸收原 socks5 特例）。"""
    _use_proxy(monkeypatch, enabled=True, http_proxy="socks5://127.0.0.1:1080")
    assert proxy_manager.get_proxy_dict() == {
        "http": "socks5://127.0.0.1:1080",
        "https": "socks5://127.0.0.1:1080",
    }


def test_whitespace_only_treated_as_empty(monkeypatch):
    """含空白的字段 strip 后视为空，触发回退。"""
    _use_proxy(
        monkeypatch,
        enabled=True,
        http_proxy="  http://127.0.0.1:2333  ",
        https_proxy="   ",
    )
    assert proxy_manager.get_proxy_dict() == {
        "http": "http://127.0.0.1:2333",
        "https": "http://127.0.0.1:2333",
    }


# ---------------------------------------------------------------------------
# 端到端：假本地代理 + 真实 requests
#
# 单测证明 dict 拼对；这里进一步证明 requests 拿到 dict 后，真的把 http 与
# https 目标流量都经过本地代理（https 走 CONNECT 隧道）——修复前只填 http_proxy
# 时 https 会绕过代理直连。假代理只监听 localhost，目标主机名从不被真解析
# （客户端只连代理），因此无外网依赖、CI 可跑。
# ---------------------------------------------------------------------------


class _FakeProxy:
    """最小 HTTP 代理：记录每个连接首行；CONNECT 回 200 后关，明文 GET 回固定 body。"""

    def __init__(self) -> None:
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind(("127.0.0.1", 0))
        self.sock.listen(8)
        self.port = self.sock.getsockname()[1]
        self.seen: list[str] = []
        self._stop = False

    def start(self) -> None:
        threading.Thread(target=self._serve, daemon=True).start()

    def _serve(self) -> None:
        while not self._stop:
            try:
                conn, _ = self.sock.accept()
            except OSError:
                break
            threading.Thread(target=self._handle, args=(conn,), daemon=True).start()

    def _handle(self, conn: socket.socket) -> None:
        try:
            conn.settimeout(5)
            data = b""
            while b"\r\n" not in data:
                chunk = conn.recv(4096)
                if not chunk:
                    return
                data += chunk
            first = data.split(b"\r\n", 1)[0].decode("latin1")
            self.seen.append(first)
            if first.startswith("CONNECT"):
                # https 隧道：确认建立即可，不实际转发 TLS（客户端随后握手会失败，
                # 但 CONNECT 已记录，足以证明 https 目标经过了代理）
                conn.sendall(b"HTTP/1.1 200 Connection established\r\n\r\n")
            else:
                body = b"ok-via-proxy"
                conn.sendall(
                    b"HTTP/1.1 200 OK\r\nContent-Length: %d\r\nConnection: close\r\n\r\n%b"
                    % (len(body), body)
                )
        except OSError:
            pass
        finally:
            conn.close()

    def stop(self) -> None:
        self._stop = True
        try:
            self.sock.close()
        except OSError:
            pass


def test_e2e_http_only_routes_both_schemes_through_proxy(monkeypatch):
    urllib3.disable_warnings()
    proxy = _FakeProxy()
    proxy.start()
    try:
        # 只填 http_proxy，指向假代理
        _use_proxy(
            monkeypatch, enabled=True, http_proxy=f"http://127.0.0.1:{proxy.port}"
        )
        proxies = proxy_manager.get_proxy_dict()
        assert proxies == {
            "http": f"http://127.0.0.1:{proxy.port}",
            "https": f"http://127.0.0.1:{proxy.port}",
        }

        # http 目标：完整往返，代理收到绝对形式 GET
        r = requests.get(
            "http://booru.invalid/index.php", proxies=proxies, timeout=5
        )
        assert r.text == "ok-via-proxy"
        assert any(
            s.startswith("GET http://booru.invalid/index.php") for s in proxy.seen
        ), f"代理没看到绝对形式 GET；seen={proxy.seen}"

        # https 目标：证明 CONNECT 到达代理（隧道后 TLS 会失败，忽略）
        try:
            requests.get(
                "https://gelbooru.com/index.php",
                proxies=proxies,
                timeout=5,
                verify=False,
            )
        except requests.exceptions.RequestException:
            pass
        assert any(
            s.startswith("CONNECT gelbooru.com:443") for s in proxy.seen
        ), f"代理没看到 CONNECT（https 走了直连？）；seen={proxy.seen}"
    finally:
        proxy.stop()
