"""Supabase Storage 访问层：下载、上传、生成签名 URL。"""
import logging
import os
import uuid

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


def upload(bucket: str, path: str, local_path: str, content_type: str = "audio/wav") -> str:
    """上传本地文件到 Storage，返回存储路径。"""
    if not settings.SUPABASE_URL or not settings.SUPABASE_SERVICE_ROLE_KEY:
        raise RuntimeError("SUPABASE_URL / SUPABASE_SERVICE_ROLE_KEY 未配置")

    url = f"{settings.SUPABASE_URL}/storage/v1/object/{bucket}/{path}"
    headers = {
        "apikey": settings.SUPABASE_SERVICE_ROLE_KEY,
        "Authorization": f"Bearer {settings.SUPABASE_SERVICE_ROLE_KEY}",
        "Content-Type": content_type,
        "x-upsert": "true",
    }
    with open(local_path, "rb") as f:
        resp = requests.post(url, headers=headers, data=f, timeout=300)
    resp.raise_for_status()
    logger.info("uploaded %s -> %s/%s", local_path, bucket, path)
    return path


def create_signed_url(bucket: str, path: str, expires_in: int = 3600) -> str:
    """生成临时签名 URL，供外部服务访问（如 DashScope Filetrans）。"""
    if not settings.SUPABASE_URL or not settings.SUPABASE_SERVICE_ROLE_KEY:
        raise RuntimeError("SUPABASE_URL / SUPABASE_SERVICE_ROLE_KEY 未配置")

    url = f"{settings.SUPABASE_URL}/storage/v1/object/sign/{bucket}/{path}"
    headers = {
        "apikey": settings.SUPABASE_SERVICE_ROLE_KEY,
        "Authorization": f"Bearer {settings.SUPABASE_SERVICE_ROLE_KEY}",
        "Content-Type": "application/json",
    }
    payload = {"expiresIn": expires_in}
    resp = requests.post(url, headers=headers, json=payload, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    signed_path = data.get("signedURL") or data.get("signed_url")
    if not signed_path:
        raise RuntimeError(f"生成签名URL失败: {data}")

    # 拼接完整 URL
    if signed_path.startswith("http"):
        return signed_path
    return f"{settings.SUPABASE_URL}{signed_path}"


def upload_and_get_url(bucket: str, local_path: str, prefix: str = "temp", content_type: str = "audio/wav") -> tuple[str, str]:
    """上传文件并生成签名 URL，返回 (存储路径, 签名URL)。"""
    ext = os.path.splitext(local_path)[1] or ".wav"
    path = f"{prefix}/{uuid.uuid4().hex}{ext}"
    upload(bucket, path, local_path, content_type)
    url = create_signed_url(bucket, path, expires_in=7200)  # 2小时有效
    return path, url
