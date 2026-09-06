"""从视频/音频链接下载音频（yt-dlp 封装）。

支持 YouTube、Bilibili、直接音频/视频文件链接等 1000+ 平台。
下载最佳质量音频，输出为 wav 格式供后续转写。
"""
import logging
import os
import subprocess
import uuid

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
    base_name = f"audio_{uuid.uuid4().hex[:8]}"
    output_template = os.path.join(workdir, f"{base_name}.%(ext)s")

    # yt-dlp 下载最佳质量音频
    logger.info("开始用 yt-dlp 下载: %s", url)
    cmd = [
        "yt-dlp",
        "-f", "bestaudio/best",
        "--extract-audio",
        "--audio-format", "wav",
        "--audio-quality", "0",
        "-o", output_template,
        "--no-playlist",  # 只下载单个视频，不下载整个播放列表
        "--max-filesize", "500M",  # 限制最大 500MB
        "--extractor-args", "youtube:player_client=web,ios",  # 使用 web/ios 客户端绕过认证
        "--no-check-certificate",
        "--user-agent", "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    ]

    # 如果配置了 YouTube cookies，写入临时文件并使用
    cookies_content = os.environ.get("YOUTUBE_COOKIES", "").strip()
    cookies_file = None
    if cookies_content:
        cookies_file = os.path.join(workdir, "cookies.txt")
        with open(cookies_file, "w", encoding="utf-8") as f:
            f.write(cookies_content)
        cmd.extend(["--cookies", cookies_file])
        logger.info("使用配置的 YouTube cookies")

    cmd.append(url)

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600,  # 10 分钟超时
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
        # 可能扩展名不同，查找所有以 base_name 开头的文件
        candidates = [f for f in os.listdir(workdir) if f.startswith(base_name)]
        if candidates:
            wav_path = os.path.join(workdir, candidates[0])
        else:
            raise RuntimeError("yt-dlp 下载完成但未找到输出文件")

    # 探测时长
    duration = _probe_duration(wav_path)
    logger.info("下载完成: %s, 时长: %.1fs", wav_path, duration)

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
