# -*- coding: utf-8 -*-
"""
阿里云函数计算 - 音频下载服务
接收视频/音频链接，用 yt-dlp 下载音频，上传到 OSS，返回签名 URL。
支持 YouTube、Bilibili、直接音频/视频文件链接等 1000+ 平台。
"""
import os
import sys
import json
import uuid
import shutil

# 确保代码包目录在 Python 路径中
_code_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _code_dir)

import oss2

try:
    import yt_dlp
except ImportError as e:
    yt_dlp = None
    _import_error = str(e)
    _dir_contents = os.listdir(_code_dir)[:30]
    _yt_dlp_exists = os.path.exists(os.path.join(_code_dir, 'yt_dlp', '__init__.py'))
else:
    _import_error = None
    _dir_contents = None
    _yt_dlp_exists = None


def handler(event, context):
    """函数计算入口函数。"""
    # 解析请求
    try:
        if isinstance(event, str):
            event = json.loads(event)
        elif isinstance(event, bytes):
            event = json.loads(event.decode('utf-8'))
    except Exception:
        pass

    # 支持 HTTP 触发器格式
    body = event
    if 'body' in event:
        try:
            body = json.loads(event['body']) if isinstance(event['body'], str) else event['body']
        except Exception:
            body = {}

    url = (body.get('url') or '').strip()
    if not url:
        return _response(400, {'error': 'url is required'})

    # 检查 yt_dlp 是否导入成功
    if yt_dlp is None:
        return _response(500, {
            'error': f'yt_dlp import failed: {_import_error}',
            'code_dir': _code_dir,
            'dir_contents': _dir_contents,
            'yt_dlp_exists': _yt_dlp_exists,
            'python_path': sys.path[:5],
        })

    # 创建临时目录
    workdir = f'/tmp/download_{uuid.uuid4().hex[:8]}'
    os.makedirs(workdir, exist_ok=True)

    try:
        # 1. yt-dlp 下载音频（用 Python API，不转码，转码交给 Worker）
        output_template = os.path.join(workdir, 'audio.%(ext)s')
        ydl_opts = {
            'format': 'bestaudio/best',
            'outtmpl': output_template,
            'noplaylist': True,
            'max_filesize': 500 * 1024 * 1024,  # 500MB
            'no_thumbnails': True,
            'nocheckcertificate': True,
            'quiet': True,
            'no_warnings': True,
            'http_headers': {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
                'Referer': 'https://www.bilibili.com/',
            },
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])

        # 2. 查找下载的文件
        audio_path = None
        for f in os.listdir(workdir):
            if f.startswith('audio.'):
                audio_path = os.path.join(workdir, f)
                break

        if not audio_path or os.path.getsize(audio_path) == 0:
            return _response(500, {'error': 'audio file not found after download'})

        file_size = os.path.getsize(audio_path)
        file_ext = os.path.splitext(audio_path)[1]

        # 3. 上传到 OSS
        oss_key = f'transcribe/{uuid.uuid4().hex}{file_ext}'
        signed_url = _upload_to_oss(audio_path, oss_key)

        return _response(200, {
            'success': True,
            'file_name': f'audio{file_ext}',
            'file_size': file_size,
            'file_ext': file_ext,
            'oss_key': oss_key,
            'signed_url': signed_url,
        })

    except Exception as e:
        return _response(500, {'error': f'internal error: {str(e)}'})
    finally:
        # 清理临时文件
        shutil.rmtree(workdir, ignore_errors=True)


def _upload_to_oss(file_path, oss_key):
    """上传文件到 OSS 并返回签名 URL。"""
    access_key_id = os.environ.get('OSS_ACCESS_KEY_ID', '')
    access_key_secret = os.environ.get('OSS_ACCESS_KEY_SECRET', '')
    bucket_name = os.environ.get('OSS_BUCKET', '')
    endpoint = os.environ.get('OSS_ENDPOINT', '')

    if not all([access_key_id, access_key_secret, bucket_name, endpoint]):
        raise RuntimeError('OSS credentials not configured')

    auth = oss2.Auth(access_key_id, access_key_secret)
    bucket = oss2.Bucket(auth, endpoint, bucket_name)

    # 上传文件
    bucket.put_object_from_file(oss_key, file_path)

    # 生成签名 URL（有效期 1 小时）
    signed_url = bucket.sign_url('GET', oss_key, 3600)
    return signed_url


def _response(status_code, body):
    """构造 HTTP 响应。"""
    return {
        'statusCode': status_code,
        'headers': {
            'Content-Type': 'application/json',
            'Access-Control-Allow-Origin': '*',
            'Access-Control-Allow-Methods': 'POST, OPTIONS',
            'Access-Control-Allow-Headers': 'Content-Type',
        },
        'body': json.dumps(body, ensure_ascii=False),
    }
