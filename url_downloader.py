"""从视频/音频链接下载音频。

优先调用阿里云函数计算下载服务（FC_DOWNLOADER_URL），
函数计算用 yt-dlp 下载后上传 OSS，返回签名 URL，Worker 再下载到本地转码。
如果未配置函数计算地址，回退到本地 yt-dlp 下载（保持向后兼容）。
"""
import logging
import os
import subprocess
import uuid
import json
import urllib.request
import urllib.error

logger = logging.getLogger("worker.url_downloader")


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

    fc_url = os.environ.get("FC_DOWNLOADER_URL", "").strip()

    if fc_url:
        logger.info("使用阿里云函数计算下载: %s", url)
        return _download_via_fc(url, workdir, fc_url)
    else:
        logger.info("未配置 FC_DOWNLOADER_URL，回退到本地 yt-dlp 下载")
        return _download_local(url, workdir)


def _download_via_fc(url: str, workdir: str, fc_url: str) -> tuple[str, float]:
    """通过阿里云函数计算下载音频。"""
    # 1. 调用函数计算 API
    payload = json.dumps({"url": url}).encode("utf-8")
    req = urllib.request.Request(
        fc_url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=600) as resp:
            result = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")[:500]
        raise RuntimeError(f"函数计算下载失败 (HTTP {e.code}): {body}")
    except Exception as e:
        raise RuntimeError(f"函数计算调用失败: {str(e)}")

    if not result.get("success"):
        raise RuntimeError(f"函数计算下载失败: {result.get('error', 'unknown error')}")

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

    # 3. 用 ffmpeg 转码为 wav（16kHz 单声道）
    wav_path = os.path.join(workdir, f"audio_{uuid.uuid4().hex[:8]}.wav")
    logger.info("转码为 wav: %s", wav_path)

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

    # 4. 探测时长
    duration = _probe_duration(wav_path)
    logger.info("下载+转码完成: %s, 时长: %.1fs", wav_path, duration)

    return wav_path, duration


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
        "--no-thumbnails",
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
