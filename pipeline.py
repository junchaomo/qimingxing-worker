"""转写管线编排：单任务 下载 → 转码 → VAD → 并发转写 → 合并清洗 → 回写。"""
import logging
import os
import shutil
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed

import soundfile as sf

import db
from audio import load_wav, transcode_to_wav
from asr_client import transcribe_segment
from config import settings
from postprocess import aggregate_language, clean_text, merge_text
from srt import build_srt
from storage import download
from vad import segment_audio

logger = logging.getLogger("worker.pipeline")


def run_task(task: dict) -> None:
    """处理一个已抢占的任务。任何未捕获异常由调用方兜底重试/失败。"""
    task_id = task["id"]
    audio_file_id = task["audio_file_id"]
    language = task.get("language")  # 可为 None（自动识别）

    workdir = os.path.join(settings.TMP_DIR, f"{task_id}_{uuid.uuid4().hex[:8]}")
    os.makedirs(workdir, exist_ok=True)
    try:
        # 1. 取音频文件元数据 + 下载
        audio_file = db.get_audio_file(audio_file_id)
        if not audio_file:
            raise RuntimeError(f"audio_file {audio_file_id} 不存在")
        raw_path = download(settings.STORAGE_BUCKET, audio_file["storage_path"], workdir)

        # 2. 统一转码 16kHz mono WAV
        wav_path = os.path.join(workdir, "full.wav")
        transcode_to_wav(raw_path, wav_path)

        # 3. VAD 分段
        wav = load_wav(wav_path)
        duration_s = int(round(len(wav) / 16000.0))
        db.update_audio_duration(audio_file_id, duration_s)  # 回填真实时长（上传时无法探测）
        segments = segment_audio(wav)
        total = len(segments)
        logger.info("task=%s 共 %d 段", task_id, total)
        db.update_progress(task_id, total, 0)

        # 4. 每段落盘为独立 wav，供 API 调用
        seg_paths = []
        for idx, (start, end, seg) in enumerate(segments):
            seg_file = os.path.join(workdir, f"seg_{idx:04d}.wav")
            sf.write(seg_file, seg, 16000)
            seg_paths.append((idx, start / 16000.0, end / 16000.0, seg_file))

        # 5. 并发调用 DashScope
        results: list[tuple[int, str]] = []
        lang_list: list[str | None] = []
        done = 0
        with ThreadPoolExecutor(max_workers=settings.WORKER_CONCURRENCY) as pool:
            futures = {
                pool.submit(transcribe_segment, seg_file, language): (idx, start, end)
                for idx, start, end, seg_file in seg_paths
            }
            for fut in as_completed(futures):
                idx, start, end = futures[fut]
                text, seg_lang = fut.result()  # 失败会抛异常，整任务走重试
                results.append((idx, text))
                lang_list.append(seg_lang)
                done += 1
                db.update_progress(task_id, total, done)
                db.upsert_segment(task_id, idx, start, end, text, "done")
                logger.info("task=%s 段 %d/%d 完成", task_id, done, total)

        # 6. 合并 + 清洗 + 语言判定 + SRT
        full_text = clean_text(merge_text(results))
        detected_lang = aggregate_language(lang_list) or language
        srt_text = build_srt([
            (idx, start, end, text)
            for (idx, text), (seg_idx, start, end, _) in zip(sorted(results), seg_paths)
        ])

        # 7. 回写
        db.mark_task_completed(task_id, full_text, srt_text, total)
        if audio_file["user_id"]:
            # 用量按真实转写时长计（转码后已回填 duration）
            db.insert_usage_record(audio_file["user_id"], task_id, duration_s)
        logger.info("task=%s 完成，语言=%s，时长=%ss", task_id, detected_lang, duration_s)
    finally:
        shutil.rmtree(workdir, ignore_errors=True)
