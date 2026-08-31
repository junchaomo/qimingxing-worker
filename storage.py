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


def upload_to_oss_and_sign_url(local_path: str) -> tuple[str | None, str | None]:
    """上传临时音频到阿里 OSS 并生成签名 URL（供 DashScope Filetrans 下载）。

    阿里服务器访问 Supabase（国外 Cloudflare）不稳定，改用阿里 OSS 中转。
    返回 (oss_key, 签名URL)；未配置 OSS 时返回 (None, None)。

    Args:
        local_path: 本地音频文件路径

    Returns:
        (oss_key, signed_url)，其中 signed_url 是带签名的临时公网 URL。
    """
    if not (settings.OSS_ACCESS_KEY_ID and settings.OSS_ACCESS_KEY_SECRET
            and settings.OSS_BUCKET and settings.OSS_ENDPOINT):
        logger.warning("OSS 未配置，跳过 OSS 中转")
        return None, None

    try:
        import oss2
    except ImportError:
        logger.warning("oss2 未安装，跳过 OSS 中转")
        return None, None

    ext = os.path.splitext(local_path)[1] or ".wav"
    key = f"{settings.OSS_TEMP_PREFIX}/{uuid.uuid4().hex}{ext}"

    auth = oss2.Auth(settings.OSS_ACCESS_KEY_ID, settings.OSS_ACCESS_KEY_SECRET)
    bucket = oss2.Bucket(auth, settings.OSS_ENDPOINT, settings.OSS_BUCKET)
    bucket.put_object_from_file(key, local_path)
    # 2 小时有效的签名 URL，供 DashScope 下载
    signed_url = bucket.sign_url("GET", key, 7200)
    logger.info("uploaded %s -> OSS %s", local_path, key)
    return key, signed_url


def delete_oss_object(key: str) -> None:
    """删除 OSS 上的临时对象。"""
    if not (settings.OSS_ACCESS_KEY_ID and settings.OSS_ACCESS_KEY_SECRET
            and settings.OSS_BUCKET and settings.OSS_ENDPOINT):
        return
    try:
        import oss2
        auth = oss2.Auth(settings.OSS_ACCESS_KEY_ID, settings.OSS_ACCESS_KEY_SECRET)
        bucket = oss2.Bucket(auth, settings.OSS_ENDPOINT, settings.OSS_BUCKET)
        bucket.delete_object(key)
        logger.info("deleted OSS object: %s", key)
    except Exception as exc:  # noqa: BLE001
        logger.warning("删除 OSS 对象失败 %s: %s", key, exc)
