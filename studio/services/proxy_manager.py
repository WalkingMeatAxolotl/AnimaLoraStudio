from typing import Dict, Optional
import logging
import os
import httpx
from huggingface_hub import set_client_factory

logger = logging.getLogger(__name__)

# 导入Secrets模块，用于获取用户的代理配置
from .. import secrets 

def get_proxy_dict() -> Optional[Dict[str, str]]:
    """
    从secrets配置中读取代理设置，并返回requests库需要的字典格式。
    如果代理未启用或配置无效，返回None。
    """
    try:
        cfg = secrets.load().proxy
        if not cfg.enabled:
            return None
        
        proxies = {}
        # 直接使用用户填写的地址，requests库会自行处理
        if cfg.http_proxy and cfg.http_proxy.startswith('socks5://'):
            proxies['http'] = cfg.http_proxy
            proxies['https'] = cfg.http_proxy
        else:
            # 原有的HTTP代理逻辑
            if cfg.http_proxy:
                proxies['http'] = cfg.http_proxy
            if cfg.https_proxy:
                proxies['https'] = cfg.https_proxy
        
        logger.info(f"Using proxies: {proxies}")
        return proxies if proxies else None
    except Exception as e:
        logger.error(f"Failed to get proxy: {e}")
        return None

def setup_global_httpx_client():
    """全局配置 huggingface_hub 的 HTTP 客户端以支持 SOCKS5"""
    proxy_config = get_proxy_dict()
    if not proxy_config:
        return
    proxy_url = proxy_config.get('http', proxy_config.get('https'))
    if proxy_url and proxy_url.startswith('socks5://'):
        # 为 httpx 客户端配置 SOCKS5 代理
        client = httpx.Client(proxies=proxy_url, transport=httpx.HTTPTransport())
    else:
        # 为一般的 HTTP 代理配置
        client = httpx.Client(proxies=proxy_url)
    set_client_factory(lambda: client)

def get_no_proxy_list() -> list[str]:
    """获取不需要代理的地址列表"""
    try:
        proxy_cfg = secrets.get().proxy
        if not proxy_cfg.no_proxy:
            return []
        return [host.strip() for host in proxy_cfg.no_proxy.split(',')]
    except Exception:
        return []

def patch_requests_session(session):
    proxies = get_proxy_dict()
    if proxies:
        session.proxies.update(proxies)
        logger.info(f"Patched session proxies: {session.proxies}")
    else:
        logger.info("No proxy configured, session unchanged")
    return session