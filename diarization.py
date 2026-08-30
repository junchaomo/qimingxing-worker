"""说话人分离模块：使用 pyannote.audio 进行说话人分离。

设计原则：
1. 默认禁用，通过 ENABLE_SPEAKER_DIARIZATION 环境变量控制
2. 短音频启用（< MAX_DIARIZATION_DURATION_S），长音频跳过
3. 失败降级，说话人分离失败时返回 None，由调用方使用启发式算法
4. 超时保护，避免长时间阻塞
"""
import logging
import os
import time
from typing import Optional

logger = logging.getLogger("worker.diarization")

# 延迟导入，避免未启用时加载 torch
_pipeline = None
_pipeline_loaded = False


def _load_pipeline(hf_token: str):
    """延迟加载 pyannote.audio 模型。"""
    global _pipeline, _pipeline_loaded
    if _pipeline_loaded:
        return _pipeline

    try:
        from pyannote.audio import Pipeline

        logger.info("加载 pyannote.audio 说话人分离模型...")
        _pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            token=hf_token,
        )
        _pipeline_loaded = True
        logger.info("pyannote.audio 模型加载完成")
        return _pipeline
    except Exception as e:
        logger.error("pyannote.audio 模型加载失败: %s", e)
        _pipeline_loaded = True  # 标记为已尝试，避免重复加载
        _pipeline = None
        return None


def diarize_audio(
    wav_path: str,
    hf_token: str,
    max_duration_s: int = 600,
    timeout_s: int = 300,
) -> Optional[list[tuple[float, float, str]]]:
    """对音频进行说话人分离。

    Args:
        wav_path: 16kHz mono WAV 文件路径
        hf_token: HuggingFace token
        max_duration_s: 最大音频时长（秒），超过则跳过
        timeout_s: 超时时间（秒）

    Returns:
        说话人时间轴列表：[(start, end, speaker), ...]
        失败或跳过时返回 None
    """
    if not hf_token:
        logger.warning("未配置 HuggingFace token，跳过说话人分离")
        return None

    # 检查音频时长（通过文件大小估算，16kHz mono 16bit = 32000 bytes/s）
    try:
        file_size = os.path.getsize(wav_path)
        estimated_duration = file_size / 32000.0
        if estimated_duration > max_duration_s:
            logger.info(
                "音频时长 %.0fs 超过上限 %ds，跳过说话人分离",
                estimated_duration,
                max_duration_s,
            )
            return None
    except Exception as e:
        logger.warning("无法获取音频文件大小: %s", e)

    # 加载模型
    pipeline = _load_pipeline(hf_token)
    if pipeline is None:
        return None

    # 执行说话人分离（带超时）
    start_time = time.time()
    try:
        logger.info("开始说话人分离...")
        diarization = pipeline(wav_path)

        # 解析结果（兼容不同版本的 pyannote.audio）
        results = []
        try:
            # 新版本：DiarizeOutput 对象，尝试获取 annotation
            if hasattr(diarization, "annotation"):
                annotation = diarization.annotation
            elif hasattr(diarization, "diarization"):
                annotation = diarization.diarization
            else:
                annotation = diarization

            for turn, _, speaker in annotation.itertracks(yield_label=True):
                results.append((turn.start, turn.end, speaker))
        except AttributeError:
            # 旧版本或其他格式，尝试直接迭代
            try:
                for turn, _, speaker in diarization.itertracks(yield_label=True):
                    results.append((turn.start, turn.end, speaker))
            except Exception:
                logger.warning("无法解析说话人分离结果格式")
                return None

        elapsed = time.time() - start_time
        logger.info(
            "说话人分离完成，耗时 %.1fs，共 %d 个片段，%d 个说话人",
            elapsed,
            len(results),
            len(set(r[2] for r in results)),
        )
        return results

    except Exception as e:
        elapsed = time.time() - start_time
        logger.error("说话人分离失败（耗时 %.1fs）: %s", elapsed, e)
        return None


def assign_speaker_to_segment(
    start: float,
    end: float,
    diarization: list[tuple[float, float, str]],
) -> str:
    """根据片段时间范围，从说话人分离结果中分配说话人。

    使用重叠时间最长的说话人。

    Args:
        start: 片段开始时间
        end: 片段结束时间
        diarization: 说话人时间轴列表

    Returns:
        说话人标签（如 SPEAKER_00、SPEAKER_01）
    """
    if not diarization:
        return "A"

    segment_duration = end - start
    if segment_duration <= 0:
        return diarization[0][2] if diarization else "A"

    # 计算每个说话人与该片段的重叠时间
    speaker_overlap: dict[str, float] = {}
    for d_start, d_end, speaker in diarization:
        overlap_start = max(start, d_start)
        overlap_end = min(end, d_end)
        overlap = max(0, overlap_end - overlap_start)
        if overlap > 0:
            speaker_overlap[speaker] = speaker_overlap.get(speaker, 0) + overlap

    if not speaker_overlap:
        # 没有重叠，取最近的说话人
        min_distance = float("inf")
        nearest_speaker = "A"
        for d_start, d_end, speaker in diarization:
            distance = min(abs(start - d_end), abs(end - d_start))
            if distance < min_distance:
                min_distance = distance
                nearest_speaker = speaker
        return nearest_speaker

    # 返回重叠时间最长的说话人
    return max(speaker_overlap, key=speaker_overlap.get)
