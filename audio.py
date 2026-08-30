"""音频处理：FFmpeg 统一转码为 16kHz 单声道 PCM WAV + 时长探测。"""
import logging
import os
import shutil
import subprocess

import soundfile as sf

from config import settings

logger = logging.getLogger("worker.audio")

SAMPLE_RATE = 16000


def _cmd(name: str) -> str:
    """解析 ffmpeg/ffprobe 可执行文件：优先 FFMPEG_BIN_DIR，其次系统 PATH。"""
    if settings.FFMPEG_BIN_DIR:
        candidate = os.path.join(settings.FFMPEG_BIN_DIR, f"{name}.exe")
        if os.path.exists(candidate):
            return candidate
    found = shutil.which(name)
    if found:
        return found
    raise RuntimeError(f"找不到 {name}，请配置 FFMPEG_BIN_DIR 或加入系统 PATH")


def probe_duration(path: str) -> float:
    """用 ffprobe 探测音频/视频时长（秒）。"""
    ffprobe = _cmd("ffprobe")
    cmd = [
        ffprobe, "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        path,
    ]
    out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, timeout=120)
    return float(out.decode().strip())


def transcode_to_wav(src: str, dst: str) -> str:
    """任意格式 -> 16kHz 单声道 PCM s16le WAV（Qwen3-ASR API 要求）。"""
    ffmpeg = _cmd("ffmpeg")
    cmd = [
        ffmpeg, "-y",
        "-i", src,
        "-ar", str(SAMPLE_RATE),
        "-ac", "1",
        "-c:a", "pcm_s16le",
        "-f", "wav",
        dst,
    ]
    logger.info("transcoding %s -> %s", src, dst)
    proc = subprocess.run(cmd, capture_output=True, timeout=1800)
    if proc.returncode != 0:
        raise RuntimeError(f"FFmpeg 转码失败: {proc.stderr.decode('utf-8', errors='ignore')[-2000:]}")
    return dst


def load_wav(path: str):
    """读取 WAV 为 float32 numpy 数组（16kHz 单声道）。"""
    data, sr = sf.read(path, dtype="float32")
    if sr != SAMPLE_RATE:
        raise ValueError(f"采样率异常: {sr} != {SAMPLE_RATE}，请检查转码步骤")
    if data.ndim > 1:
        data = data.mean(axis=1)
    return data


def trim_wav(src: str, dst: str, start_sec: float, end_sec: float) -> str:
    """裁剪 WAV 文件：保留 [start_sec, end_sec] 时间段。

    Args:
        src: 源 WAV 文件路径
        dst: 目标 WAV 文件路径
        start_sec: 开始时间（秒）
        end_sec: 结束时间（秒）

    Returns:
        目标文件路径
    """
    ffmpeg = _cmd("ffmpeg")
    duration = end_sec - start_sec
    cmd = [
        ffmpeg, "-y",
        "-i", src,
        "-ss", str(start_sec),
        "-t", str(duration),
        "-ar", str(SAMPLE_RATE),
        "-ac", "1",
        "-c:a", "pcm_s16le",
        "-f", "wav",
        dst,
    ]
    logger.info("trimming %s -> %s (%.1fs - %.1fs, duration %.1fs)",
                src, dst, start_sec, end_sec, duration)
    proc = subprocess.run(cmd, capture_output=True, timeout=600)
    if proc.returncode != 0:
        raise RuntimeError(f"FFmpeg 裁剪失败: {proc.stderr.decode('utf-8', errors='ignore')[-2000:]}")
    return dst
