"""全局 HTTP/HTTPS 代理 helpers。

读 `secrets.proxy` 配置，组装 requests 库要的 proxies dict + 给现有
`requests.Session` 注入 proxies。当前只覆盖 booru 下载链路；对
`huggingface_hub` 的 transport-level 代理注入需要单独的 monkeypatch，
不在本模块范围内（PR #162 原版本里的 `setup_global_httpx_client` 用了
`huggingface_hub.set_client_factory` —— 该符号在 huggingface_hub 内不存在，
import 时直接 ImportError；本 hotfix 一并删除该 dead 函数）。
"""
from typing import Dict, Optional
import logging

from .. import secrets

logger = logging.getLogger(__name__)


def get_proxy_dict() -> Optional[Dict[str, str]]:
    """从 secrets 读代理设置，返回 requests 库要的字典格式；未启用返 None。"""
    try:
        cfg = secrets.load().proxy
        if not cfg.enabled:
            return None

        proxies: Dict[str, str] = {}
        http_p = (cfg.http_proxy or "").strip()
        https_p = (cfg.https_proxy or "").strip()
        # 任一字段留空 → 回退到另一个。本地代理（clash / v2ray / socks5 …）通常
        # 单个端口同时转发 http 与 https 目标流量，用户往往只填一个字段。这也兑现
        # 设置页 tips3「HTTPS 留空回退用 HTTP 代理」的承诺。
        #
        # 旧逻辑只在 `http_proxy` 显式以 socks5:// 开头时才镜像到 https；明文 HTTP
        # 代理只填 http_proxy 时 proxies["https"] 不设 → 所有 booru（全 https）请求
        # 直连 → 超时。这里对所有 scheme 统一回退，socks5 特例被此逻辑吸收；两字段
        # 都显式填写时各自尊重，不再互相覆盖。
        if http_p and not https_p:
            https_p = http_p
        elif https_p and not http_p:
            http_p = https_p
        if http_p:
            proxies["http"] = http_p
        if https_p:
            proxies["https"] = https_p

        logger.info(f"Using proxies: {proxies}")
        return proxies if proxies else None
    except Exception as e:
        logger.error(f"Failed to get proxy: {e}")
        return None


def get_no_proxy_list() -> list[str]:
    """`no_proxy` 字段拆成 host 列表。"""
    try:
        proxy_cfg = secrets.load().proxy
        if not proxy_cfg.no_proxy:
            return []
        return [host.strip() for host in proxy_cfg.no_proxy.split(",")]
    except Exception:
        return []


def patch_requests_session(session):
    """给已有 requests.Session 注入 proxies；未启用代理则原样返回。"""
    proxies = get_proxy_dict()
    if proxies:
        session.proxies.update(proxies)
        logger.info(f"Patched session proxies: {session.proxies}")
    else:
        logger.info("No proxy configured, session unchanged")
    return session
