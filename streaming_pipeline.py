"""转写管线：支持两种模式。

1. 单人声模式（默认）：
   - VAD 分段 + 并发同步转写
   - 实时更新进度和结果
   - 不支持说话人分离

2. 多人声模式（diarization_enabled=true）：
   - 使用 Qwen-Audio-3.0-ASR-Flash-Filetrans
   - 整段转写，支持说话人分离
   - 无实时进度，任务完成后显示结果
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
from asr_client import transcribe_file_diarization, transcribe_segment
from audio import load_wav, probe_duration, transcode_to_wav, trim_wav
from config import settings
from postprocess import aggregate_language
from storage import download, upload_and_get_url
from vad import segment_audio

logger = logging.getLogger("worker.streaming")


def build_result_simple(
    results: dict[int, tuple[str, str | None, list]],
    seg_paths: list[tuple[int, float, float, str]],
) -> tuple[str, str]:
    """生成简单的纯文本结果和 SRT（单人声模式）。"""
    full_text_parts = []
    srt_lines = []
    srt_idx = 1

    for idx, start, end, _ in seg_paths:
        if idx not in results:
            continue
        text, _, _ = results[idx]
        if not text.strip():
            continue
        full_text_parts.append(text.strip())

        # SRT
        start_h = int(start // 3600)
        start_m = int((start % 3600) // 60)
        start_s = int(start % 60)
        end_h = int(end // 3600)
        end_m = int((end % 3600) // 60)
        end_s = int(end % 60)
        srt_lines.append(f"{srt_idx}")
        srt_lines.append(f"{start_h:02d}:{start_m:02d}:{start_s:02d},000 --> {end_h:02d}:{end_m:02d}:{end_s:02d},000")
        srt_lines.append(text.strip())
        srt_lines.append("")
        srt_idx += 1

    return "\n\n".join(full_text_parts), "\n".join(srt_lines)


def build_dialogue_result(sentences: list[dict]) -> tuple[str, str]:
    """根据句子级结果生成对话格式的 Markdown 和 SRT（多人声模式）。"""
    if not sentences:
        return "", ""

    dialogue_segments = []
    current_speaker = None
    current_text = ""
    current_start = 0
    current_end = 0

    for sent in sentences:
        speaker = sent.get("speaker_id")
        text = sent.get("text", "").strip()
        if not text:
            continue

        if speaker != current_speaker:
            if current_speaker is not None and current_text:
                dialogue_segments.append({
                    "speaker": current_speaker,
                    "text": current_text.strip(),
                    "start": current_start,
                    "end": current_end,
                })
            current_speaker = speaker
            current_text = text
            current_start = sent["begin_time"]
            current_end = sent["end_time"]
        else:
            current_text += text
            current_end = sent["end_time"]

    if current_speaker is not None and current_text:
        dialogue_segments.append({
            "speaker": current_speaker,
            "text": current_text.strip(),
            "start": current_start,
            "end": current_end,
        })

    # Markdown
    md_lines = []
    for seg in dialogue_segments:
        speaker_label = f"Speaker {chr(65 + seg['speaker'])}" if seg["speaker"] is not None else "Speaker"
        time_str = f"{int(seg['start']//60):02d}:{int(seg['start']%60):02d}"
        md_lines.append(f"**{speaker_label}** ({time_str})\n\n{seg['text']}\n")
    full_text = "\n".join(md_lines)

    # SRT
    srt_lines = []
    for i, seg in enumerate(dialogue_segments, 1):
        start_h = int(seg["start"] // 3600)
        start_m = int((seg["start"] % 3600) // 60)
        start_s = int(seg["start"] % 60)
        start_ms = int((seg["start"] % 1) * 1000)
        end_h = int(seg["end"] // 3600)
        end_m = int((seg["end"] % 3600) // 60)
        end_s = int(seg["end"] % 60)
        end_ms = int((seg["end"] % 1) * 1000)
        speaker_label = f"Speaker {chr(65 + seg['speaker'])}" if seg["speaker"] is not None else "Speaker"
        srt_lines.append(f"{i}")
        srt_lines.append(f"{start_h:02d}:{start_m:02d}:{start_s:02d},{start_ms:03d} --> {end_h:02d}:{end_m:02d}:{end_s:02d},{end_ms:03d}")
        srt_lines.append(f"{speaker_label}: {seg['text']}")
        srt_lines.append("")
    srt_text = "\n".join(srt_lines)

    return full_text, srt_text


def run_task_streaming(task_id: str) -> None:
    """执行转写任务，根据 diarization_enabled 选择模式。"""
    workdir = os.path.join(settings.TMP_DIR, f"task_{task_id}_{uuid.uuid4().hex[:8]}")
    os.makedirs(workdir, exist_ok=True)
    temp_storage_path = None

    try:
        # 1. 获取任务信息
        task = db.get_task(task_id)
        if not task:
            raise RuntimeError(f"任务不存在: {task_id}")

        audio_file_id = task["audio_file_id"]
        audio_file = db.get_audio_file(audio_file_id)
        if not audio_file:
            raise RuntimeError(f"音频文件不存在: {audio_file_id}")

        language = task.get("language")
        trim_start = task.get("trim_start")
        trim_end = task.get("trim_end")
        diarization_enabled = bool(task.get("diarization_enabled", False))

        logger.info("task=%s 开始处理，模式=%s", task_id, "多人声(说话人分离)" if diarization_enabled else "单人声(分段转写)")

        # 2. 下载原始音频
        storage_path = audio_file["storage_path"]
        raw_path = download(settings.STORAGE_BUCKET, storage_path, workdir)
        logger.info("task=%s 已下载原始音频", task_id)

        # 3. 转码为 WAV
        wav_path = os.path.join(workdir, "input.wav")
        transcode_to_wav(raw_path, wav_path)
        logger.info("task=%s 已转码为 WAV", task_id)

        # 4. 裁剪（如果指定）
        trim_offset = 0.0
        if trim_start is not None or trim_end is not None:
            trimmed_path = os.path.join(workdir, "trimmed.wav")
            start = float(trim_start) if trim_start is not None else 0.0
            if trim_end is not None:
                end = float(trim_end)
            else:
                end = probe_duration(wav_path)
            if end > start:
                trim_wav(wav_path, trimmed_path, start, end)
                wav_path = trimmed_path
                trim_offset = start
                logger.info("task=%s 已裁剪 %.1fs - %.1fs", task_id, start, end)

        # 5. 获取音频时长
        duration_s = int(round(probe_duration(wav_path)))
        db.update_audio_duration(audio_file_id, duration_s)
        logger.info("task=%s 音频时长: %ds", task_id, duration_s)

        # 6. 根据模式选择转写方式
        if diarization_enabled:
            # ===== 多人声模式：Filetrans 异步转写 + 说话人分离 =====
            logger.info("task=%s 使用 Filetrans 模式（说话人分离）", task_id)

            # 上传到 Supabase 并生成签名 URL
            logger.info("task=%s 上传音频到 Storage...", task_id)
            temp_storage_path, file_url = upload_and_get_url(
                settings.STORAGE_BUCKET, wav_path, prefix="asr_temp"
            )
            logger.info("task=%s 已生成签名 URL", task_id)

            db.update_task_stage(task_id, "processing", 0.1)

            # 调用 Filetrans
            start_time = time.time()
            full_text, detected_lang, sentences = transcribe_file_diarization(
                file_url,
                language=language,
                poll_interval=5,
                timeout=max(duration_s * 3, 600),
            )
            elapsed = time.time() - start_time
            logger.info("task=%s Filetrans 转写完成，耗时 %.0fs，句子数=%d", task_id, elapsed, len(sentences))

            # 生成对话格式结果
            result_text, result_srt = build_dialogue_result(sentences)

        else:
            # ===== 单人声模式：VAD 分段 + 并发同步转写 =====
            logger.info("task=%s 使用分段转写模式", task_id)

            wav = load_wav(wav_path)
            segments = segment_audio(wav)
            total = len(segments)
            logger.info("task=%s VAD分段共 %d 段", task_id, total)
            db.update_progress(task_id, total, 0)

            # 落盘每段
            seg_paths = []
            for idx, (start, end, seg) in enumerate(segments):
                seg_file = os.path.join(workdir, f"seg_{idx:04d}.wav")
                sf.write(seg_file, seg, 16000)
                global_start = start / 16000.0 + trim_offset
                global_end = end / 16000.0 + trim_offset
                seg_paths.append((idx, global_start, global_end, seg_file))

            # 并发转写
            results: dict[int, tuple[str, str | None, list]] = {}
            lang_list: list[str | None] = []
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

                    db.upsert_segment(task_id, idx, seg_start, seg_paths[idx][2], text, "done")
                    full_text, srt_text = build_result_simple(results, seg_paths)
                    db.update_partial_result(task_id, full_text, srt_text, total, done)
                    logger.info("task=%s 转写段 %d/%d 完成", task_id, done, total)

            detected_lang = aggregate_language(lang_list) or language
            result_text, result_srt = build_result_simple(results, seg_paths)

        # 7. 标记任务完成
        db.mark_task_completed(task_id, result_text, result_srt, 1)
        logger.info("task=%s 任务已标记为完成", task_id)

        # 8. 扣除额度
        if audio_file["user_id"]:
            db.insert_usage_record(audio_file["user_id"], task_id, duration_s)

        logger.info(
            "task=%s 转写完成，语言=%s，时长=%ds，模式=%s",
            task_id, detected_lang or language, duration_s,
            "多人声" if diarization_enabled else "单人声",
        )

    except Exception as e:
        logger.error("task=%s 转写失败: %s", task_id, e, exc_info=True)
        db.mark_task_failed(task_id, f"{type(e).__name__}: {e}")
        raise
    finally:
        shutil.rmtree(workdir, ignore_errors=True)
        # 清理临时上传的音频
        if temp_storage_path:
            try:
                import requests
                url = f"{settings.SUPABASE_URL}/storage/v1/object/{settings.STORAGE_BUCKET}/{temp_storage_path}"
                headers = {
                    "apikey": settings.SUPABASE_SERVICE_ROLE_KEY,
                    "Authorization": f"Bearer {settings.SUPABASE_SERVICE_ROLE_KEY}",
                }
                requests.delete(url, headers=headers, timeout=30)
                logger.info("已清理临时音频: %s", temp_storage_path)
            except Exception as e:
                logger.warning("清理临时音频失败: %s", e)
