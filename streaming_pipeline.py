"""转写管线：使用 Qwen-Audio-3.0-ASR-Flash-Filetrans 异步模型。

流程：
1. 下载原始音频
2. 转码为单声道 WAV（说话人分离要求单声道）
3. 裁剪（如果指定）
4. 上传到 Supabase 并生成签名 URL
5. 调用 Filetrans 异步转写（支持说话人分离）
6. 解析结果，生成对话格式（带 speaker 标签）
7. 更新数据库

注意：Filetrans 是整段转写，不需要 VAD 分段。
"""
import logging
import os
import shutil
import time
import uuid

import soundfile as sf

import db
from asr_client import transcribe_file
from audio import load_wav, probe_duration, transcode_to_wav, trim_wav
from config import settings
from postprocess import aggregate_language
from storage import create_signed_url, download, upload_and_get_url

logger = logging.getLogger("worker.streaming")


def build_dialogue_result(sentences: list[dict]) -> tuple[str, str]:
    """根据句子级结果生成对话格式的 Markdown 和 SRT。

    连续相同说话人的句子合并为一段。
    """
    if not sentences:
        return "", ""

    # 按说话人连续分段
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

    # 生成 Markdown
    md_lines = []
    for seg in dialogue_segments:
        speaker_label = f"Speaker {chr(65 + seg['speaker'])}" if seg["speaker"] is not None else "Speaker"
        time_str = f"{int(seg['start']//60):02d}:{int(seg['start']%60):02d}"
        md_lines.append(f"**{speaker_label}** ({time_str})\n\n{seg['text']}\n")
    full_text = "\n".join(md_lines)

    # 生成 SRT
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
    """执行转写任务（Filetrans 异步模式）。"""
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

        logger.info("task=%s 开始处理，音频文件=%s", task_id, audio_file_id)

        # 2. 下载原始音频
        storage_path = audio_file["storage_path"]
        raw_path = download(settings.STORAGE_BUCKET, storage_path, workdir)
        logger.info("task=%s 已下载原始音频: %s", task_id, raw_path)

        # 3. 转码为单声道 16kHz WAV（说话人分离要求单声道）
        wav_path = os.path.join(workdir, "input.wav")
        transcode_to_wav(raw_path, wav_path)
        logger.info("task=%s 已转码为单声道 WAV", task_id)

        # 4. 裁剪（如果指定）
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
                logger.info("task=%s 已裁剪 %.1fs - %.1fs", task_id, start, end)

        # 5. 获取音频时长
        duration_s = int(round(probe_duration(wav_path)))
        db.update_audio_duration(audio_file_id, duration_s)
        logger.info("task=%s 音频时长: %ds", task_id, duration_s)

        # 6. 上传到 Supabase 并生成签名 URL
        logger.info("task=%s 上传音频到 Storage 以生成可访问 URL...", task_id)
        temp_storage_path, file_url = upload_and_get_url(
            settings.STORAGE_BUCKET,
            wav_path,
            prefix="asr_temp",
            content_type="audio/wav",
        )
        logger.info("task=%s 已生成签名 URL", task_id)

        # 7. 更新任务状态为处理中
        db.update_task_stage(task_id, "processing", 0.1)

        # 8. 调用 Filetrans 转写
        logger.info("task=%s 开始调用 Filetrans 转写（说话人分离=%s）...",
                    task_id, settings.ENABLE_SPEAKER_DIARIZATION)
        start_time = time.time()

        full_text, detected_lang, sentences = transcribe_file(
            file_url,
            language=language,
            poll_interval=5,
            timeout=max(duration_s * 3, 600),
        )

        elapsed = time.time() - start_time
        logger.info("task=%s 转写完成，耗时 %.0fs，句子数=%d", task_id, elapsed, len(sentences))

        # 9. 生成对话格式结果
        if sentences and settings.ENABLE_SPEAKER_DIARIZATION:
            result_text, result_srt = build_dialogue_result(sentences)
        else:
            # 无说话人分离，用纯文本
            result_text = full_text
            result_srt = ""
            if sentences:
                # 简单 SRT
                srt_lines = []
                for i, sent in enumerate(sentences, 1):
                    srt_lines.append(f"{i}")
                    srt_lines.append(f"{int(sent['begin_time']//3600):02d}:{int((sent['begin_time']%3600)//60):02d}:{int(sent['begin_time']%60):02d},000 --> {int(sent['end_time']//3600):02d}:{int((sent['end_time']%3600)//60):02d}:{int(sent['end_time']%60):02d},000")
                    srt_lines.append(sent["text"])
                    srt_lines.append("")
                result_srt = "\n".join(srt_lines)

        # 10. 标记任务完成
        db.mark_task_completed(task_id, result_text, result_srt, 1)
        logger.info("task=%s 任务已标记为完成", task_id)

        # 11. 扣除额度
        if audio_file["user_id"]:
            db.insert_usage_record(audio_file["user_id"], task_id, duration_s)

        logger.info(
            "task=%s 转写完成，语言=%s，时长=%ds，说话人=%s",
            task_id,
            detected_lang or language,
            duration_s,
            "已分离" if settings.ENABLE_SPEAKER_DIARIZATION and sentences else "未分离",
        )

    except Exception as e:
        logger.error("task=%s 转写失败: %s", task_id, e, exc_info=True)
        db.mark_task_failed(task_id, f"{type(e).__name__}: {e}")
        raise
    finally:
        # 清理临时文件
        shutil.rmtree(workdir, ignore_errors=True)
        # 清理临时上传的音频
        if temp_storage_path:
            try:
                from storage import settings as st
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
