"""VAD 分段：在静音处把长音频切成 ≤ MAX_SEGMENT_S 的段。

实现与官方 Qwen3-ASR-Toolkit 的 process_vad 保持一致：
1. 用 silero_vad.get_speech_timestamps 找出语音段；
2. 以语音段起点作为候选切分点，按目标时长(默认120s)就近在静音处切；
3. 任何段超过单段硬上限(180s)时再均分，确保不超 API 的 3 分钟限制。
"""
import logging

import numpy as np
from silero_vad import get_speech_timestamps

from config import settings
from audio import SAMPLE_RATE

logger = logging.getLogger("worker.vad")


def segment_audio(wav: np.ndarray) -> list[tuple[int, int, np.ndarray]]:
    """输入 16kHz mono float32 数组，返回 [(start_sample, end_sample, segment), ...]。"""
    max_samples = settings.MAX_SEGMENT_S * SAMPLE_RATE

    # 短于 3 分钟直接整段
    if len(wav) / SAMPLE_RATE < 180:
        return [(0, len(wav), wav)]

    try:
        from silero_vad import load_silero_vad
        vad = load_silero_vad(onnx=True)
        speech_timestamps = get_speech_timestamps(
            wav,
            vad,
            sampling_rate=SAMPLE_RATE,
            return_seconds=False,
            min_speech_duration_ms=settings.MIN_SPEECH_MS,
            min_silence_duration_ms=settings.MIN_SILENCE_MS,
        )
        if not speech_timestamps:
            raise ValueError("VAD 未检测到语音")

        # 候选切分点 = 0 + 每段语音起点 + 末尾
        split_points = {0, len(wav)}
        for ts in speech_timestamps:
            split_points.add(ts["start"])
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
