"""流式转写管线：长音频分段处理，每完成一段就实时更新结果。

优化后的流程：
1. 先并发转写所有段，每完成一段就立即更新结果（用户快速看到内容）
2. 转写完成后，再做说话人分离（后台进行，不阻塞用户）
3. 说话人分离完成后，更新说话人标签

分段策略：
- 每段 75 秒，重叠 3 秒（避免截断句子）
"""
import logging
import os
import shutil
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import soundfile as sf

import db
from audio import load_wav, transcode_to_wav, trim_wav
from asr_client import transcribe_segment
from config import settings
from diarization import assign_speaker_to_segment, diarize_audio
from postprocess import aggregate_language, clean_text
from srt import build_srt
from storage import download

logger = logging.getLogger("worker.streaming")

# 流式分段参数
STREAM_SEGMENT_S = 75      # 每段目标时长（秒）
STREAM_OVERLAP_S = 3       # 相邻段重叠时长（秒）


def split_audio_streaming(wav: np.ndarray, sample_rate: int = 16000) -> list[tuple[int, int, np.ndarray]]:
    """把长音频切成固定时长的段，相邻段重叠。"""
    seg_samples = STREAM_SEGMENT_S * sample_rate
    overlap_samples = STREAM_OVERLAP_S * sample_rate
    step = seg_samples - overlap_samples

    segments = []
    start = 0
    while start < len(wav):
        end = min(start + seg_samples, len(wav))
        segments.append((start, end, wav[start:end]))
        if end >= len(wav):
            break
        start += step
    return segments


def align_speakers(
    segment_diarizations: list[list[tuple[float, float, str]] | None],
) -> list[dict[str, str]]:
    """对齐不同段的说话人标签。"""
    mappings = []
    current_map: dict[str, str] = {}
    next_label = 0

    for diar in segment_diarizations:
        if not diar:
            mappings.append(dict(current_map))
            continue

        speakers_in_order = []
        seen = set()
        for _, _, spk in diar:
            if spk not in seen:
                seen.add(spk)
                speakers_in_order.append(spk)

        seg_map = {}
        for spk in speakers_in_order:
            if spk in current_map:
                seg_map[spk] = current_map[spk]
            else:
                label = chr(ord('A') + next_label)
                seg_map[spk] = label
                current_map[spk] = label
                next_label += 1

        mappings.append(seg_map)

    return mappings


def build_result(
    results: dict[int, tuple[str, str | None]],
    seg_paths: list[tuple[int, float, float, str]],
    segment_diarizations: list[list[tuple[float, float, str]] | None] | None = None,
    speaker_mappings: list[dict[str, str]] | None = None,
) -> tuple[str, str]:
    """根据已完成的段构建完整的 result_text 和 result_srt。"""
    sorted_indices = sorted(results.keys())
    merged_texts = []
    srt_segments = []

    for i, seg_idx in enumerate(sorted_indices):
        text, _ = results[seg_idx]
        _, seg_start, seg_end, _ = seg_paths[seg_idx]
        merged_texts.append(text)

        # 分配说话人
        speaker = None
        if segment_diarizations and speaker_mappings:
            diar = segment_diarizations[seg_idx]
            spk_map = speaker_mappings[seg_idx]
            if diar and spk_map:
                raw_speaker = assign_speaker_to_segment(0, seg_end - seg_start, diar)
                speaker = spk_map.get(raw_speaker, "A")

        srt_segments.append((i, seg_start, seg_end, text, speaker))

    full_text = clean_text("\n\n".join(merged_texts))
    srt_text = build_srt(
        [(i, s, e, t) for i, s, e, t, _ in srt_segments],
        speakers=[spk for _, _, _, _, spk in srt_segments],
    )
    return full_text, srt_text


def run_task_streaming(task: dict) -> None:
    """流式处理一个任务：先转写（实时更新），后做说话人分离。"""
    task_id = task["id"]
    audio_file_id = task["audio_file_id"]
    language = task.get("language")
    trim_start = task.get("trim_start")
    trim_end = task.get("trim_end")

    workdir = os.path.join(settings.TMP_DIR, f"{task_id}_{uuid.uuid4().hex[:8]}")
    os.makedirs(workdir, exist_ok=True)
    try:
        # 1. 下载 + 转码
        audio_file = db.get_audio_file(audio_file_id)
        if not audio_file:
            raise RuntimeError(f"audio_file {audio_file_id} 不存在")
        raw_path = download(settings.STORAGE_BUCKET, audio_file["storage_path"], workdir)

        wav_path = os.path.join(workdir, "full.wav")
        transcode_to_wav(raw_path, wav_path)

        # 2. 裁剪（如果指定）
        if trim_start is not None or trim_end is not None:
            trimmed_path = os.path.join(workdir, "trimmed.wav")
            start = float(trim_start) if trim_start is not None else 0.0
            if trim_end is not None:
                end = float(trim_end)
            else:
                from audio import probe_duration
                end = probe_duration(wav_path)
            if end > start:
                trim_wav(wav_path, trimmed_path, start, end)
                wav_path = trimmed_path
                logger.info("task=%s 已裁剪 %.1fs - %.1fs", task_id, start, end)

        # 3. 加载音频 + 分段
        wav = load_wav(wav_path)
        duration_s = int(round(len(wav) / 16000.0))
        db.update_audio_duration(audio_file_id, duration_s)

        segments = split_audio_streaming(wav)
        total = len(segments)
        logger.info("task=%s 流式分段共 %d 段（每段约 %ds）", task_id, total, STREAM_SEGMENT_S)
        db.update_progress(task_id, total, 0)

        # 4. 落盘每段为独立 wav
        seg_paths = []
        for idx, (start, end, seg) in enumerate(segments):
            seg_file = os.path.join(workdir, f"seg_{idx:04d}.wav")
            sf.write(seg_file, seg, 16000)
            seg_paths.append((idx, start / 16000.0, end / 16000.0, seg_file))

        # 5. 先并发转写所有段，每完成一段就更新结果（不等待说话人分离）
        logger.info("task=%s 开始并发转写（不等待说话人分离）...", task_id)
        results: dict[int, tuple[str, str | None]] = {}
        lang_list: list[str | None] = []
        done = 0

        with ThreadPoolExecutor(max_workers=min(settings.WORKER_CONCURRENCY, 4)) as pool:
            futures = {
                pool.submit(transcribe_segment, seg_file, language): (idx, start, end)
                for idx, start, end, seg_file in seg_paths
            }
            for fut in as_completed(futures):
                idx, start, end = futures[fut]
                text, seg_lang = fut.result()
                results[idx] = (text, seg_lang)
                lang_list.append(seg_lang)
                done += 1

                # 存入 segments 表
                db.upsert_segment(task_id, idx, start, end, text, "done")

                # 实时更新结果（说话人标签先用 None，前端用启发式算法）
                full_text, srt_text = build_result(results, seg_paths)
                db.update_partial_result(task_id, full_text, srt_text, total, done)
                logger.info("task=%s 转写段 %d/%d 完成，已实时更新", task_id, done, total)

        logger.info("task=%s 所有段转写完成，开始说话人分离...", task_id)

        # 6. 转写完成后，再做说话人分离（串行，因为 pyannote.audio 不支持多线程）
        segment_diarizations: list[list[tuple[float, float, str]] | None] = []
        if settings.ENABLE_SPEAKER_DIARIZATION and settings.HUGGINGFACE_TOKEN:
            for idx, start, end, seg_file in seg_paths:
                diar = diarize_audio(
                    seg_file,
                    settings.HUGGINGFACE_TOKEN,
                    max_duration_s=STREAM_SEGMENT_S + 10,
                    timeout_s=120,
                )
                segment_diarizations.append(diar)
                if diar:
                    logger.info("task=%s 说话人分离段 %d/%d 完成，%d 个说话人",
                                task_id, idx + 1, total, len(set(d[2] for d in diar)))
                else:
                    logger.info("task=%s 说话人分离段 %d/%d 跳过/失败", task_id, idx + 1, total)

                # 每完成一段说话人分离就更新结果
                speaker_mappings = align_speakers(segment_diarizations + [None] * (total - len(segment_diarizations)))
                full_text, srt_text = build_result(results, seg_paths, segment_diarizations, speaker_mappings)
                db.update_partial_result(task_id, full_text, srt_text, total, total)
        else:
            segment_diarizations = [None] * total

        # 7. 全部完成，标记任务完成
        speaker_mappings = align_speakers(segment_diarizations)
        full_text, srt_text = build_result(results, seg_paths, segment_diarizations, speaker_mappings)
        detected_lang = aggregate_language(lang_list) or language

        db.mark_task_completed(task_id, full_text, srt_text, total)

        if audio_file["user_id"]:
            db.insert_usage_record(audio_file["user_id"], task_id, duration_s)

        logger.info("task=%s 流式转写完成，语言=%s，时长=%ss，共 %d 段",
                    task_id, detected_lang, duration_s, total)

    finally:
        shutil.rmtree(workdir, ignore_errors=True)
