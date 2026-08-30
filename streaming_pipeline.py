"""流式转写管线：长音频分段处理，每完成一段就实时更新结果。

流程：
1. VAD 按静音处切分，每段约 30-60 秒
2. 并发转写所有段，每完成一段就立即更新结果（用户快速看到内容）
3. 所有段完成后，合并生成最终结果

注意：已取消 pyannote 说话人分离，只保留 Qwen3-ASR 纯转写。
"""
import logging
import os
import shutil
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import soundfile as sf

import db
from audio import load_wav, transcode_to_wav, trim_wav
from asr_client import transcribe_segment
from config import settings
from postprocess import aggregate_language, clean_text
from srt import build_srt
from storage import download
from vad import segment_audio

logger = logging.getLogger("worker.streaming")


def map_speaker_labels(diarization: list[tuple[float, float, str]]) -> dict[str, str]:
    """把 pyannote 的 SPEAKER_00 等标签映射为 A、B、C..."""
    speakers = []
    seen = set()
    for _, _, spk in diarization:
        if spk not in seen:
            seen.add(spk)
            speakers.append(spk)
    return {spk: chr(ord("A") + i) for i, spk in enumerate(speakers)}


def assign_speaker_to_time(
    t: float, diarization: list[tuple[float, float, str]]
) -> str | None:
    """根据时间点从全局说话人时间轴中分配说话人。"""
    for start, end, speaker in diarization:
        if start <= t < end:
            return speaker
    return None


def build_dialogue_from_words(
    all_words: list[tuple[float, float, str]],
    diarization: list[tuple[float, float, str]] | None,
    speaker_map: dict[str, str] | None,
) -> list[dict]:
    """根据词级时间戳和说话人时间轴，生成对话分段。

    返回：[{"start": float, "end": float, "speaker": str, "text": str}, ...]
    """
    if not all_words:
        return []

    # 按时间排序
    all_words.sort(key=lambda x: x[0])

    segments = []
    current_speaker = None
    current_text = []
    current_start = None
    current_end = None

    for start, end, word in all_words:
        # 分配说话人
        raw_speaker = None
        if diarization and speaker_map:
            raw_speaker = assign_speaker_to_time(start, diarization)
        speaker = speaker_map.get(raw_speaker, "A") if raw_speaker and speaker_map else "A"

        # 如果说话人变化，开始新段
        if speaker != current_speaker and current_text:
            segments.append({
                "start": current_start,
                "end": current_end,
                "speaker": current_speaker,
                "text": "".join(current_text).strip(),
            })
            current_text = []
            current_start = None

        current_speaker = speaker
        if current_start is None:
            current_start = start
        current_end = end
        current_text.append(word)

    # 最后一段
    if current_text:
        segments.append({
            "start": current_start,
            "end": current_end,
            "speaker": current_speaker,
            "text": "".join(current_text).strip(),
        })

    return segments


def format_dialogue_text(segments: list[dict]) -> str:
    """把对话分段格式化为 markdown 文本。"""
    lines = []
    for seg in segments:
        speaker = seg["speaker"] or "A"
        text = seg["text"]
        if text:
            lines.append(f"**{speaker}**: {text}")
    return "\n\n".join(lines)


def build_result_simple(
    results: dict[int, tuple[str, str | None, list]],
    seg_paths: list[tuple[int, float, float, str]],
) -> tuple[str, str]:
    """转写过程中使用的简单结果构建（无说话人标签）。"""
    sorted_indices = sorted(results.keys())
    merged_texts = []
    srt_segments = []

    for i, seg_idx in enumerate(sorted_indices):
        text, _, _ = results[seg_idx]
        _, seg_start, seg_end, _ = seg_paths[seg_idx]
        merged_texts.append(text)
        srt_segments.append((i, seg_start, seg_end, text, None))

    full_text = clean_text("\n\n".join(merged_texts))
    srt_text = build_srt(
        [(i, s, e, t) for i, s, e, t, _ in srt_segments],
        speakers=[None] * len(srt_segments),
    )
    return full_text, srt_text


def build_result_with_speakers(
    all_words: list[tuple[float, float, str]],
    diarization: list[tuple[float, float, str]] | None,
) -> tuple[str, str]:
    """说话人分离完成后，生成带说话人标签的结果。"""
    logger.info("生成结果：词数=%d, 说话人分离片段=%d", len(all_words), len(diarization) if diarization else 0)

    if not all_words:
        logger.warning("词级时间戳为空，无法生成对话格式，使用纯文本")
        return "", ""

    if not diarization:
        # 没有说话人分离结果，用简单格式
        logger.warning("没有说话人分离结果，使用无标签格式")
        text = clean_text("".join(w[2] for w in sorted(all_words, key=lambda x: x[0])))
        return text, ""

    speaker_map = map_speaker_labels(diarization)
    logger.info("说话人映射: %s", speaker_map)
    segments = build_dialogue_from_words(all_words, diarization, speaker_map)
    logger.info("生成对话分段数=%d", len(segments))

    # 生成 markdown 文本
    full_text = format_dialogue_text(segments)

    # 生成 SRT
    srt_entries = []
    for i, seg in enumerate(segments):
        srt_entries.append((i, seg["start"], seg["end"], seg["text"], seg["speaker"]))
    srt_text = build_srt(
        [(i, s, e, t) for i, s, e, t, _ in srt_entries],
        speakers=[spk for _, _, _, _, spk in srt_entries],
    )

    return full_text, srt_text


def run_task_streaming(task: dict) -> None:
    """流式处理一个任务：先转写（实时更新），后做整段说话人分离。"""
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
        trim_offset = 0.0
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
                trim_offset = start
                logger.info("task=%s 已裁剪 %.1fs - %.1fs", task_id, start, end)

        # 3. 加载音频 + 分段
        wav = load_wav(wav_path)
        duration_s = int(round(len(wav) / 16000.0))
        db.update_audio_duration(audio_file_id, duration_s)

        segments = segment_audio(wav)
        total = len(segments)
        logger.info("task=%s VAD分段共 %d 段（按静音处切分）", task_id, total)
        db.update_progress(task_id, total, 0)

        # 4. 落盘每段为独立 wav
        seg_paths = []
        for idx, (start, end, seg) in enumerate(segments):
            seg_file = os.path.join(workdir, f"seg_{idx:04d}.wav")
            sf.write(seg_file, seg, 16000)
            # 时间加上裁剪偏移量，得到全局时间戳
            global_start = start / 16000.0 + trim_offset
            global_end = end / 16000.0 + trim_offset
            seg_paths.append((idx, global_start, global_end, seg_file))

        # 5. 先并发转写所有段，每完成一段就更新结果（不等待说话人分离）
        logger.info("task=%s 开始并发转写（不等待说话人分离）...", task_id)
        results: dict[int, tuple[str, str | None, list]] = {}
        lang_list: list[str | None] = []
        all_words: list[tuple[float, float, str]] = []
        done = 0

        with ThreadPoolExecutor(max_workers=min(settings.WORKER_CONCURRENCY, 4)) as pool:
            futures = {
                pool.submit(transcribe_segment, seg_file, language): (idx, global_start)
                for idx, global_start, _, seg_file in seg_paths
            }
            for fut in as_completed(futures):
                idx, seg_start = futures[fut]
                text, seg_lang, words = fut.result()
                results[idx] = (text, seg_lang, words)
                lang_list.append(seg_lang)
                done += 1

                # 把词级时间戳加上段偏移量，存入全局列表
                for w_start, w_end, w_text in words:
                    all_words.append((w_start + seg_start, w_end + seg_start, w_text))

                # 存入 segments 表
                _, _, seg_end, _ = seg_paths[idx]
                db.upsert_segment(task_id, idx, seg_start, seg_end, text, "done")

                # 实时更新结果（说话人标签先用 None）
                full_text, srt_text = build_result_simple(results, seg_paths)
                db.update_partial_result(task_id, full_text, srt_text, total, done)
                logger.info("task=%s 转写段 %d/%d 完成，已实时更新", task_id, done, total)

        logger.info("task=%s 所有段转写完成，共收集 %d 个词级时间戳", task_id, len(all_words))

        # 6. 生成最终结果（纯文本，无说话人分离）
        logger.info("task=%s 开始生成最终结果，all_words=%d", task_id, len(all_words))
        full_text, srt_text = build_result_simple(results, seg_paths)
        detected_lang = aggregate_language(lang_list) or language
        logger.info("task=%s 最终结果生成完成，文本长度=%d", task_id, len(full_text))

        db.mark_task_completed(task_id, full_text, srt_text, total)
        logger.info("task=%s 任务已标记为完成，progress=1.0", task_id)

        if audio_file["user_id"]:
            db.insert_usage_record(audio_file["user_id"], task_id, duration_s)

        logger.info(
            "task=%s 转写完成，语言=%s，时长=%ss，共 %d 段",
            task_id, detected_lang, duration_s, total,
        )

    finally:
        shutil.rmtree(workdir, ignore_errors=True)
