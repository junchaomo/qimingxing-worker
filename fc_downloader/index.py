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
import urllib.request
import urllib.parse

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
        # 判断是否是直接的音频/视频文件链接
        direct_extensions = ('.mp3', '.wav', '.m4a', '.aac', '.flac', '.ogg', '.wma',
                           '.mp4', '.webm', '.mkv', '.avi', '.mov', '.flv')
        parsed_url = urllib.parse.urlparse(url)
        path_lower = parsed_url.path.lower()
        is_direct_file = any(path_lower.endswith(ext) for ext in direct_extensions)

        if is_direct_file:
            # 直接文件链接：用 urllib 直接下载
            logger_info("直接文件链接，用 urllib 下载: " + url)
            file_ext = os.path.splitext(parsed_url.path)[1] or '.mp3'
            audio_path = os.path.join(workdir, f'audio{file_ext}')

            req = urllib.request.Request(url, headers={
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            })
            with urllib.request.urlopen(req, timeout=120) as resp:
                with open(audio_path, 'wb') as f:
                    while True:
                        chunk = resp.read(8192)
                        if not chunk:
                            break
                        f.write(chunk)
        elif any(host in (parsed_url.hostname or '') for host in ('bilibili.com', 'b23.tv', 'bili2233.cn')):
            # Bilibili 专用：官方 API 获取音频流（yt-dlp 请求特征会被 Bilibili 412 风控）
            logger_info("Bilibili 链接，用官方 API 下载: " + url)
            audio_path = _download_bilibili(url, workdir)
        else:
            # 视频平台链接：用 yt-dlp 下载
            logger_info("视频平台链接，用 yt-dlp 下载: " + url)
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

            # 查找下载的文件
            audio_path = None
            for f in os.listdir(workdir):
                if f.startswith('audio.'):
                    audio_path = os.path.join(workdir, f)
                    break

            if not audio_path:
                return _response(500, {'error': 'audio file not found after download'})

            file_ext = os.path.splitext(audio_path)[1]

        if os.path.getsize(audio_path) == 0:
            return _response(500, {'error': 'downloaded audio file is empty'})

        file_size = os.path.getsize(audio_path)

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


def _download_bilibili(url, workdir):
    """Bilibili 官方 API 下载音频流（m4s），返回文件路径。

    yt-dlp 的请求特征会被 Bilibili 风控返回 412，改用公开 API：
    view 拿 cid -> playurl 拿 dash 音频直链 -> 下载（带 UA + Referer）。
    """
    import re

    UA = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'

    def _get_json(u, referer):
        req = urllib.request.Request(u, headers={'User-Agent': UA, 'Referer': referer})
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read().decode('utf-8', 'ignore'))

    # 短链跳转
    if 'b23.tv' in url or 'bili2233.cn' in url:
        req = urllib.request.Request(url, headers={'User-Agent': UA})
        with urllib.request.urlopen(req, timeout=20) as resp:
            url = resp.geturl()
            logger_info('Bilibili 短链解析为: ' + url)

    bvid = None
    aid = None
    m = re.search(r'BV[0-9A-Za-z]{10}', url)
    if m:
        bvid = m.group(0)
    else:
        m = re.search(r'/(av\d+)', url)
        if m:
            aid = m.group(1)
    if not bvid and not aid:
        raise RuntimeError('无法解析 Bilibili 视频 ID: ' + url)

    # 1. view API 拿 cid
    if bvid:
        view_url = f'https://api.bilibili.com/x/web-interface/view?bvid={bvid}'
    else:
        view_url = f'https://api.bilibili.com/x/web-interface/view?aid={aid[2:]}'
    view = _get_json(view_url, 'https://www.bilibili.com/')
    if view.get('code') != 0:
        raise RuntimeError('Bilibili view API 失败: ' + str(view.get('message')))
    cid = view['data']['cid']

    # 2. playurl API 拿音频直链
    if bvid:
        play_url = f'https://api.bilibili.com/x/player/playurl?bvid={bvid}&cid={cid}&fnval=16&fourk=1'
    else:
        play_url = f'https://api.bilibili.com/x/player/playurl?aid={aid[2:]}&cid={cid}&fnval=16&fourk=1'
    play = _get_json(play_url, 'https://www.bilibili.com/')
    if play.get('code') != 0:
        raise RuntimeError('Bilibili playurl API 失败: ' + str(play.get('message')))

    audio_url = None
    dash = play.get('data', {}).get('dash') or {}
    audio_list = dash.get('audio') or []
    if audio_list:
        best = max(audio_list, key=lambda a: a.get('bandwidth', 0))
        audio_url = best.get('baseUrl') or best.get('base_url')
    if not audio_url:
        durl = play.get('data', {}).get('durl') or []
        if durl:
            audio_url = durl[0].get('url')
    if not audio_url:
        raise RuntimeError('无法获取 Bilibili 音频流地址')

    # 3. 下载音频流（m4s），带 Referer 防盗链
    audio_path = os.path.join(workdir, 'audio.m4s')
    logger_info('开始下载 Bilibili 音频流: ' + audio_url[:120])
    req = urllib.request.Request(audio_url, headers={
        'User-Agent': UA,
        'Referer': f'https://www.bilibili.com/video/{bvid or aid}',
    })
    with urllib.request.urlopen(req, timeout=300) as resp:
        with open(audio_path, 'wb') as f:
            while True:
                chunk = resp.read(65536)
                if not chunk:
                    break
                f.write(chunk)
    if os.path.getsize(audio_path) == 0:
        raise RuntimeError('Bilibili 下载的文件为空')
    return audio_path


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


def logger_info(msg):
    """打印日志到函数计算日志。"""
    print(msg, flush=True)
