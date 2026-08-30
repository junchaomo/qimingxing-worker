"""VAD 分段：在静音处把长音频切成 ≤ MAX_SEGMENT_S 的段。

使用轻量的能量阈值 VAD（无需 silero-vad/torch）：
1. 将音频分成 20ms 帧，计算每帧 RMS 能量；
2. 能量超过阈值判定为语音，否则为静音；
3. 合并连续语音帧得到语音段，过滤过短的语音/静音；
4. 以语音段起点作为候选切分点，按目标时长就近在静音处切；
5. 任何段超过单段硬上限时再均分，确保不超 API 限制。
"""
import logging

import numpy as np

from config import settings
from audio import SAMPLE_RATE

logger = logging.getLogger("worker.vad")

# 帧长 20ms
FRAME_MS = 20
FRAME_SAMPLES = int(SAMPLE_RATE * FRAME_MS / 1000)


def _compute_energy(wav: np.ndarray) -> np.ndarray:
    """计算每帧 RMS 能量。"""
    n_frames = len(wav) // FRAME_SAMPLES
    if n_frames == 0:
        return np.array([])
    frames = wav[: n_frames * FRAME_SAMPLES].reshape(n_frames, FRAME_SAMPLES)
    # RMS 能量
    rms = np.sqrt(np.mean(frames ** 2, axis=1) + 1e-10)
    return rms


def _detect_speech_segments(wav: np.ndarray) -> list[tuple[int, int]]:
    """能量阈值 VAD，返回语音段列表 [(start_sample, end_sample), ...]。"""
    rms = _compute_energy(wav)
    if len(rms) == 0:
        return [(0, len(wav))]

    # 动态阈值：取能量分布的某个分位数作为基准
    # 使用 20 分位数作为静音基准，阈值 = 静音基准 * 3
    baseline = np.percentile(rms, 20)
    threshold = max(baseline * 3, 0.01)  # 最低阈值保护

    # 二值化
    is_speech = rms > threshold

    # 合并连续语音帧，最小语音时长 300ms，最小静音时长 300ms
    min_speech_frames = max(1, int(settings.MIN_SPEECH_MS / FRAME_MS))
    min_silence_frames = max(1, int(settings.MIN_SILENCE_MS / FRAME_MS))

    # 先过滤过短的语音段
    segments = []
    i = 0
    while i < len(is_speech):
        if is_speech[i]:
            j = i
            while j < len(is_speech) and is_speech[j]:
                j += 1
            if j - i >= min_speech_frames:
                segments.append((i, j))
            i = j
        else:
            i += 1

    # 再合并间隔过短的语音段（中间静音太短）
    merged = []
    for seg in segments:
        if merged and seg[0] - merged[-1][1] < min_silence_frames:
            merged[-1] = (merged[-1][0], seg[1])
        else:
            merged.append(seg)

    # 转换为样本点
    result = []
    for start_frame, end_frame in merged:
        start_sample = start_frame * FRAME_SAMPLES
        end_sample = min(end_frame * FRAME_SAMPLES, len(wav))
        result.append((start_sample, end_sample))

    if not result:
        # 没检测到语音，返回整段
        result = [(0, len(wav))]

    return result


def segment_audio(wav: np.ndarray) -> list[tuple[int, int, np.ndarray]]:
    """输入 16kHz mono float32 数组，返回 [(start_sample, end_sample, segment), ...]。"""
    max_samples = settings.MAX_SEGMENT_S * SAMPLE_RATE

    # 短于 3 分钟直接整段
    if len(wav) / SAMPLE_RATE < 180:
        return [(0, len(wav), wav)]

    try:
        speech_segments = _detect_speech_segments(wav)
        if not speech_segments:
            raise ValueError("VAD 未检测到语音")

        # 候选切分点 = 0 + 每段语音起点 + 末尾
        split_points = {0, len(wav)}
        for start, _ in speech_segments:
            split_points.add(start)
        sorted_points = sorted(split_points)

        # 按目标时长就近选候选点
        final_points = {0, len(wav)}
        step = settings.VAD_SEGMENT_THRESHOLD_S * SAMPLE_RATE
        target = step
        while target < len(wav):
            closest = min(sorted_points, key=lambda p: abs(p - target))
            final_points.add(closest)
            target += step
        ordered = sorted(final_points)

        # 超硬上限的段再均分
        new_points = [0]
        for i in range(1, len(ordered)):
            start, end = ordered[i - 1], ordered[i]
            length = end - start
            if length <= max_samples:
                new_points.append(end)
            else:
                n = int(np.ceil(length / max_samples))
                sub = length / n
                for j in range(1, n):
                    new_points.append(start + int(j * sub))
                new_points.append(end)

        segments = []
        for i in range(len(new_points) - 1):
            s, e = int(new_points[i]), int(new_points[i + 1])
            segments.append((s, e, wav[s:e]))
        return segments
    except Exception as exc:  # noqa: BLE001
        logger.warning("VAD 分段失败，降级为等长硬切: %s", exc)
        return _fallback_split(wav, max_samples)


def _fallback_split(wav: np.ndarray, max_samples: int) -> list[tuple[int, int, np.ndarray]]:
    segments = []
    for s in range(0, len(wav), max_samples):
        e = min(s + max_samples, len(wav))
        if e > s:
            segments.append((s, e, wav[s:e]))
    return segments
