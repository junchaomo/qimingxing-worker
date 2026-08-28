"""Supabase Storage 访问层：以 Service Role Key 下载原始音频/视频。"""
import logging
import os

import requests

from config import settings

logger = logging.getLogger("worker.storage")


def download(bucket: str, path: str, dest_dir: str) -> str:
    """下载存储对象到本地，返回本地绝对路径。"""
    if not settings.SUPABASE_URL or not settings.SUPABASE_SERVICE_ROLE_KEY:
        raise RuntimeError("SUPABASE_URL / SUPABASE_SERVICE_ROLE_KEY 未配置")

    os.makedirs(dest_dir, exist_ok=True)
    local_path = os.path.join(dest_dir, os.path.basename(path))

    url = f"{settings.SUPABASE_URL}/storage/v1/object/{bucket}/{path}"
    headers = {
        "apikey": settings.SUPABASE_SERVICE_ROLE_KEY,
        "Authorization": f"Bearer {settings.SUPABASE_SERVICE_ROLE_KEY}",
    }
    logger.info("downloading %s -> %s", url, local_path)
    resp = requests.get(url, headers=headers, timeout=300)
    resp.raise_for_status()
    with open(local_path, "wb") as f:
        f.write(resp.content)
    return local_path
