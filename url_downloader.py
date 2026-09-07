"""从视频/音频链接下载音频。

优先调用阿里云函数计算下载服务（通过 FC SDK 调用），
函数计算用 yt-dlp 下载后上传 OSS，返回签名 URL，Worker 再下载到本地转码。
如果未配置函数计算 AccessKey，回退到本地 yt-dlp 下载（保持向后兼容）。
"""
import logging
import os
import subprocess
import uuid
import json
import base64
import urllib.request
import urllib.error
import urllib.parse

logger = logging.getLogger("worker.url_downloader")

# 函数计算配置
FC_ACCESS_KEY_ID = os.environ.get("FC_ACCESS_KEY_ID", os.environ.get("OSS_ACCESS_KEY_ID", "")).strip()
FC_ACCESS_KEY_SECRET = os.environ.get("FC_ACCESS_KEY_SECRET", os.environ.get("OSS_ACCESS_KEY_SECRET", "")).strip()
FC_REGION = os.environ.get("FC_REGION", "cn-hangzhou").strip()
FC_ACCOUNT_ID = os.environ.get("FC_ACCOUNT_ID", "").strip()
FC_FUNCTION_NAME = os.environ.get("FC_FUNCTION_NAME", "svc-8ecfe18f$downloader").strip()

# 延迟导入 SDK，避免未配置时影响启动
_fc_client = None


def _get_fc_client():
    """获取函数计算客户端（懒加载）。"""
    global _fc_client
    if _fc_client is not None:
        return _fc_client
    
    if not all([FC_ACCESS_KEY_ID, FC_ACCESS_KEY_SECRET, FC_ACCOUNT_ID]):
        return None
    
    try:
        from alibabacloud_fc20230330.client import Client as FCClient
        from alibabacloud_tea_openapi import models as open_api_models
        
        from alibabacloud_fc20230330 import models as fc_models
        from alibabacloud_tea_openapi import models as open_api_models
        
        config = open_api_models.Config(
            access_key_id=FC_ACCESS_KEY_ID,
            access_key_secret=FC_ACCESS_KEY_SECRET,
            endpoint=f"{FC_ACCOUNT_ID}.{FC_REGION}.fc.aliyuncs.com"
        )
        # 设置较长的超时时间，函数计算下载音频可能需要几分钟
        config.connect_timeout = 30000  # 30秒连接超时
        config.read_timeout = 600000   # 10分钟读取超时
        
        _fc_client = FCClient(config)
        logger.info("函数计算客户端初始化成功")
        return _fc_client
    except Exception as e:
        logger.warning("函数计算客户端初始化失败: %s", e)
        return None


def _invoke_fc(url: str) -> dict:
    """调用函数计算下载音频。

    FC 3.0 API 路径为 POST /2023-03-30/functions/{functionName}/invocations，
    其中 functionName 是 "service$function" 形式（如 svc-8ecfe18f$downloader）。
    SDK 的 invoke_function 内部 percent_encode 会把 $ 当作查询参数分隔符，
    导致函数名被截断成 "svc-8ecfe18f" 而报 FunctionNotFound，因此这里手动把
    $ 编码为 %24，并直接用 call_api 调用自定义 pathname。
    """
    from alibabacloud_tea_openapi import utils_models as open_api_util_models
    from alibabacloud_tea_util import models as util_models

    client = _get_fc_client()
    if client is None:
        raise RuntimeError("函数计算客户端未初始化")

    payload = json.dumps({"url": url}).encode("utf-8")

    func_enc = FC_FUNCTION_NAME.replace("$", "%24")
    params = open_api_util_models.Params(
        action="InvokeFunction",
        version="2023-03-30",
        protocol="HTTPS",
        pathname=f"/2023-03-30/functions/{func_enc}/invocations",
        method="POST",
        auth_type="AK",
        style="ROA",
        req_body_type="json",
        body_type="binary",
    )
    req = open_api_util_models.OpenApiRequest(
        headers={"x-fc-invocation-type": "Sync", "x-fc-log-type": "None"},
        query={},
        body=payload,
        stream=payload,
    )
    runtime = util_models.RuntimeOptions(connect_timeout=30000, read_timeout=600000)
    resp = client.call_api(params, req, runtime)

    # 解析响应（body 可能是 bytes 或可读流）
    raw_body = resp.get("body")
    if isinstance(raw_body, bytes):
        body = raw_body.decode("utf-8", "ignore")
    elif hasattr(raw_body, "read"):
        body = raw_body.read().decode("utf-8", "ignore")
    else:
        body = str(raw_body)

    try:
        result = json.loads(body)
    except json.JSONDecodeError:
        raise RuntimeError(f"函数计算返回非 JSON: {body[:500]}")

    # 函数计算的响应可能嵌套在 body 字段中
    if "body" in result and isinstance(result["body"], str):
        try:
            result = json.loads(result["body"])
        except json.JSONDecodeError:
            pass

    if result.get("statusCode", 200) != 200:
        raise RuntimeError(f"函数计算错误: {result.get('error', result)}")

    if not result.get("success"):
        raise RuntimeError(f"函数计算下载失败: {result.get('error', 'unknown error')}")

    return result


def download_audio_from_url(url: str, workdir: str) -> tuple[str, float]:
    """从 URL 下载音频并转码为 wav。

    Args:
        url: 视频/音频链接
        workdir: 工作目录

    Returns:
        (wav 文件路径, 时长秒数)

    Raises:
        RuntimeError: 下载或转码失败
    """
    os.makedirs(workdir, exist_ok=True)

    # 判断是否是直接的音频/视频文件链接
    direct_extensions = ('.mp3', '.wav', '.m4a', '.aac', '.flac', '.ogg', '.wma',
                       '.mp4', '.webm', '.mkv', '.avi', '.mov', '.flv')
    parsed = urllib.parse.urlparse(url)
    is_direct_file = any(parsed.path.lower().endswith(ext) for ext in direct_extensions)

    if is_direct_file:
        # 直接文件链接：直接用 urllib 下载，不经过函数计算
        logger.info("直接文件链接，本地直接下载: %s", url)
        return _download_direct_file(url, workdir)

    if all([FC_ACCESS_KEY_ID, FC_ACCESS_KEY_SECRET, FC_ACCOUNT_ID]):
        logger.info("使用阿里云函数计算下载: %s", url)
        try:
            return _download_via_fc(url, workdir)
        except Exception as e:
            logger.warning("函数计算下载失败，回退到本地 yt-dlp: %s", e)
            return _download_local(url, workdir)
    else:
        logger.info("未配置函数计算 AccessKey，使用本地 yt-dlp 下载")
        return _download_local(url, workdir)


def _download_direct_file(url: str, workdir: str) -> tuple[str, float]:
    """直接下载音频/视频文件并转码为 wav。"""
    parsed = urllib.parse.urlparse(url)
    file_ext = os.path.splitext(parsed.path)[1] or '.mp3'
    raw_path = os.path.join(workdir, f"raw_{uuid.uuid4().hex[:8]}{file_ext}")

    logger.info("开始下载: %s -> %s", url, raw_path)

    req = urllib.request.Request(url, headers={
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    })
    try:
        with urllib.request.urlopen(req, timeout=300) as resp:
            with open(raw_path, 'wb') as f:
                while True:
                    chunk = resp.read(8192)
                    if not chunk:
                        break
                    f.write(chunk)
    except Exception as e:
        raise RuntimeError(f"直接下载失败: {str(e)}")

    if not os.path.exists(raw_path) or os.path.getsize(raw_path) == 0:
        raise RuntimeError("下载的文件为空")

    file_size = os.path.getsize(raw_path)
    logger.info("下载完成，文件大小: %d bytes", file_size)

    # 转码为 wav
    wav_path = _transcode_to_wav(raw_path, workdir)

    # 探测时长
    duration = _probe_duration(wav_path)
    logger.info("直接下载+转码完成: %s, 时长: %.1fs", wav_path, duration)

    return wav_path, duration


def _download_via_fc(url: str, workdir: str) -> tuple[str, float]:
    """通过阿里云函数计算下载音频。"""
    # 1. 调用函数计算 API
    result = _invoke_fc(url)
    
    signed_url = result.get("signed_url", "")
    if not signed_url:
        raise RuntimeError("函数计算未返回 signed_url")

    file_ext = result.get("file_ext", ".webm")
    logger.info("函数计算下载完成，文件大小: %d bytes", result.get("file_size", 0))

    # 2. 从 OSS 签名 URL 下载文件到本地
    raw_path = os.path.join(workdir, f"raw_{uuid.uuid4().hex[:8]}{file_ext}")
    logger.info("从 OSS 下载到本地: %s", raw_path)

    try:
        with urllib.request.urlopen(signed_url, timeout=300) as resp:
            with open(raw_path, "wb") as f:
                while True:
                    chunk = resp.read(8192)
                    if not chunk:
                        break
                    f.write(chunk)
    except Exception as e:
        raise RuntimeError(f"从 OSS 下载失败: {str(e)}")

    if not os.path.exists(raw_path) or os.path.getsize(raw_path) == 0:
        raise RuntimeError("OSS 下载的文件为空")

    # 3. 用 ffmpeg 转码为 wav
    wav_path = _transcode_to_wav(raw_path, workdir)

    # 4. 探测时长
    duration = _probe_duration(wav_path)
    logger.info("下载+转码完成: %s, 时长: %.1fs", wav_path, duration)

    return wav_path, duration


def _transcode_to_wav(raw_path: str, workdir: str) -> str:
    """用 ffmpeg 将音频/视频文件转码为 16kHz 单声道 wav。"""
    wav_path = os.path.join(workdir, f"audio_{uuid.uuid4().hex[:8]}.wav")
    logger.info("转码为 wav: %s -> %s", raw_path, wav_path)

    try:
        subprocess.run(
            [
                "ffmpeg", "-y",
                "-i", raw_path,
                "-ar", "16000",
                "-ac", "1",
                "-vn",
                wav_path,
            ],
            capture_output=True,
            timeout=300,
            check=True,
        )
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"ffmpeg 转码失败: {e.stderr[-300:] if e.stderr else str(e)}")
    except Exception as e:
        raise RuntimeError(f"ffmpeg 转码失败: {str(e)}")

    # 清理原始文件
    try:
        os.remove(raw_path)
    except Exception:
        pass

    if not os.path.exists(wav_path) or os.path.getsize(wav_path) == 0:
        raise RuntimeError("转码后的 wav 文件为空")

    return wav_path


def _download_local(url: str, workdir: str) -> tuple[str, float]:
    """本地 yt-dlp 下载（回退方案）。"""
    base_name = f"audio_{uuid.uuid4().hex[:8]}"
    output_template = os.path.join(workdir, f"{base_name}.%(ext)s")

    logger.info("开始用本地 yt-dlp 下载: %s", url)
    cmd = [
        "yt-dlp",
        "-f", "bestaudio/best",
        "--extract-audio",
        "--audio-format", "wav",
        "--audio-quality", "0",
        "-o", output_template,
        "--no-playlist",
        "--max-filesize", "500M",
        "--no-check-certificate",
        "--user-agent", "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    ]
    cmd.append(url)

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600,
        )
        if result.returncode != 0:
            logger.error("yt-dlp 下载失败: %s", result.stderr[-500:])
            raise RuntimeError(f"yt-dlp 下载失败: {result.stderr[-300:]}")
    except subprocess.TimeoutExpired:
        raise RuntimeError("yt-dlp 下载超时（超过 10 分钟）")
    except FileNotFoundError:
        raise RuntimeError("yt-dlp 未安装，请在 Worker 环境中安装 yt-dlp")

    # 查找下载的 wav 文件
    wav_path = os.path.join(workdir, f"{base_name}.wav")
    if not os.path.exists(wav_path):
        candidates = [f for f in os.listdir(workdir) if f.startswith(base_name)]
        if candidates:
            wav_path = os.path.join(workdir, candidates[0])
        else:
            raise RuntimeError("yt-dlp 下载完成但未找到输出文件")

    duration = _probe_duration(wav_path)
    logger.info("本地下载完成: %s, 时长: %.1fs", wav_path, duration)

    return wav_path, duration


def _probe_duration(wav_path: str) -> float:
    """用 ffprobe 探测音频时长。"""
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                wav_path,
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        return float(result.stdout.strip())
    except Exception:
        logger.warning("ffprobe 时长探测失败，返回 0")
        return 0.0
