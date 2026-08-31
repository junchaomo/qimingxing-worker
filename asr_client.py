"""DashScope ASR 调用层：支持两种模式。

1. 同步模式（qwen-audio-3.0-asr-flash）：
   - 适合单人声音频
   - 速度快，支持分段并发和实时进度
   - 不支持说话人分离

2. 异步模式（qwen-audio-3.0-asr-flash-filetrans）：
   - 适合多人声音频
   - 支持说话人分离（speaker diarization）
   - 整段转写，无实时进度
"""
import base64
import logging
import threading
import time

import requests

from config import settings

logger = logging.getLogger("worker.asr_client")

# 全局并发护栏：限制同时进行的 API 请求总数
_global_semaphore = threading.BoundedSemaphore(settings.GLOBAL_API_SEMAPHORE)


# ============================================================
# 同步模式：qwen-audio-3.0-asr-flash（单人声，分段转写）
# ============================================================

def _wav_to_data_uri(wav_path: str) -> str:
    with open(wav_path, "rb") as f:
        raw = f.read()
    return f"data:audio/{settings.ASR_AUDIO_FORMAT};base64," + base64.b64encode(raw).decode()


def _call_sync_once(wav_path: str, language: str | None) -> dict:
    """同步调用 qwen-audio-3.0-asr-flash。"""
    url = "https://dashscope.aliyuncs.com/api/v1/services/aigc/multimodal-generation/generation"
    payload = {
        "model": "qwen-audio-3.0-asr-flash",
        "input": {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_audio",
                            "input_audio": {"data": _wav_to_data_uri(wav_path)},
                        }
                    ],
                }
            ]
        },
        "parameters": {
            "format": settings.ASR_AUDIO_FORMAT,
            "sample_rate": str(settings.ASR_SAMPLE_RATE),
        },
    }
    if language:
        payload["parameters"]["asr_options"] = {"language": language}

    headers = {
        "Authorization": f"Bearer {settings.DASHSCOPE_API_KEY}",
        "Content-Type": "application/json",
        "X-DashScope-SSE": "disable",
    }
    resp = requests.post(url, json=payload, headers=headers, timeout=settings.API_TIMEOUT_S)
    if resp.status_code != 200:
        raise RuntimeError(f"DashScope HTTP {resp.status_code}: {resp.text[:500]}")
    return resp.json()


def _parse_words(sentence: dict | None) -> list[tuple[float, float, str]]:
    """从 sentence 中解析词级时间戳。"""
    if not sentence or not isinstance(sentence, dict):
        return []
    word_list = sentence.get("words") or []
    words: list[tuple[float, float, str]] = []
    for w in word_list:
        if not isinstance(w, dict):
            continue
        start = w.get("begin_time") or w.get("start") or 0
        end = w.get("end_time") or w.get("end") or start
        text = w.get("text") or w.get("word") or ""
        if not text:
            continue
        if start > 1000:
            start = start / 1000.0
            end = end / 1000.0
        words.append((float(start), float(end), str(text)))
    return words


def _parse_sync(data: dict) -> tuple[str, str | None, list[tuple[float, float, str]]]:
    """解析同步调用结果，返回 (text, language, words)。"""
    sentence = data.get("sentence") or (data.get("output") or {}).get("sentence")
    words = _parse_words(sentence)

    text = data.get("text") or (data.get("output") or {}).get("text") or ""
    if text:
        return text, None, words

    try:
        choices = data["output"]["choices"]
        message = choices[0]["message"]
        text = message["content"][0].get("text", "")
        annotations = message.get("annotations") or []
        language = None
        for ann in annotations:
            if ann.get("type") == "audio_info":
                language = ann.get("language")
                break
        return text, language, words
    except (KeyError, IndexError, TypeError):
        raise RuntimeError(f"DashScope 响应解析失败: {str(data)[:800]}") from None


def transcribe_segment(
    wav_path: str, language: str | None = None
) -> tuple[str, str | None, list[tuple[float, float, str]]]:
    """同步转写单个段，带全局限流与指数退避重试。返回 (text, language, words)。"""
    if not settings.DASHSCOPE_API_KEY:
        raise RuntimeError("DASHSCOPE_API_KEY 未配置")

    with _global_semaphore:
        last_exc: Exception | None = None
        for attempt in range(settings.SEGMENT_MAX_RETRIES):
            try:
                data = _call_sync_once(wav_path, language)
                return _parse_sync(data)
            except Exception as exc:
                last_exc = exc
                wait = 2 ** attempt
                logger.warning("segment 调用失败(第%d次): %s，%.0fs 后重试",
                               attempt + 1, exc, wait)
                time.sleep(wait)
        raise RuntimeError(f"段转写重试耗尽: {last_exc}")


# ============================================================
# 异步模式：qwen-audio-3.0-asr-flash-filetrans（多人声，说话人分离）
# ============================================================

def submit_task(file_url: str, language: str | None = None) -> str:
    """提交异步转写任务，返回 task_id。"""
    url = f"{settings.DASHSCOPE_BASE_URL}/services/audio/asr/transcription"

    parameters: dict = {"diarization_enabled": True}
    if language:
        parameters["language_hints"] = [language]

    payload = {
        "model": settings.DASHSCOPE_MODEL,
        "input": {"file_urls": [file_url]},
        "parameters": parameters,
    }

    headers = {
        "Authorization": f"Bearer {settings.DASHSCOPE_API_KEY}",
        "Content-Type": "application/json",
        "X-DashScope-Async": "enable",
    }

    resp = requests.post(url, json=payload, headers=headers, timeout=30)
    if resp.status_code != 200:
        raise RuntimeError(f"提交任务失败 HTTP {resp.status_code}: {resp.text[:500]}")

    data = resp.json()
    task_id = data.get("output", {}).get("task_id")
    if not task_id:
        raise RuntimeError(f"提交任务失败，未返回 task_id: {data}")

    logger.info("已提交转写任务: task_id=%s", task_id)
    return task_id


def query_task(task_id: str) -> dict:
    """查询任务状态。"""
    url = f"{settings.DASHSCOPE_BASE_URL}/tasks/{task_id}"
    headers = {"Authorization": f"Bearer {settings.DASHSCOPE_API_KEY}"}
    resp = requests.get(url, headers=headers, timeout=30)
    if resp.status_code != 200:
        raise RuntimeError(f"查询任务失败 HTTP {resp.status_code}: {resp.text[:500]}")
    return resp.json()


def download_result(transcription_url: str) -> dict:
    """下载转写结果 JSON。"""
    resp = requests.get(transcription_url, timeout=60)
    if resp.status_code != 200:
        raise RuntimeError(f"下载结果失败 HTTP {resp.status_code}: {resp.text[:500]}")
    return resp.json()


def parse_filetrans_result(result_data: dict) -> tuple[str, str | None, list[dict]]:
    """解析 Filetrans 结果，返回 (完整文本, 语言, 句子列表)。"""
    transcripts = result_data.get("transcripts", [])
    if not transcripts:
        return "", None, []

    transcript = transcripts[0]
    full_text = transcript.get("text", "")
    language = transcript.get("language") or None

    sentences = []
    for sent in transcript.get("sentences", []):
        begin = (sent.get("begin_time") or 0) / 1000.0
        end = (sent.get("end_time") or 0) / 1000.0
        text = sent.get("text", "")
        speaker_id = sent.get("speaker_id")
        sentences.append({
            "begin_time": begin,
            "end_time": end,
            "text": text,
            "speaker_id": speaker_id,
        })

    logger.info(
        "解析结果: 文本长度=%d, 句子数=%d, 说话人数=%d",
        len(full_text),
        len(sentences),
        len(set(s["speaker_id"] for s in sentences if s["speaker_id"] is not None)),
    )
    return full_text, language, sentences


def transcribe_file_diarization(
    file_url: str,
    language: str | None = None,
    poll_interval: int = 5,
    timeout: int = 1800,
) -> tuple[str, str | None, list[dict]]:
    """完整的 Filetrans 转写流程：提交→轮询→下载→解析。

    返回 (完整文本, 语言, 句子列表)，句子带 speaker_id。
    """
    start_time = time.time()
    task_id = submit_task(file_url, language)

    while True:
        elapsed = time.time() - start_time
        if elapsed > timeout:
            raise RuntimeError(f"转写超时（{timeout}秒）")

        task_data = query_task(task_id)
        output = task_data.get("output", {})
        task_status = output.get("task_status", "")
        logger.info("任务状态: %s, 已用 %.0fs", task_status, elapsed)

        if task_status == "SUCCEEDED":
            results = output.get("results", [])
            if not results:
                raise RuntimeError("任务成功但无结果")
            transcription_url = results[0].get("transcription_url")
            if not transcription_url:
                raise RuntimeError(f"结果中无 transcription_url: {results[0]}")
            result_data = download_result(transcription_url)
            return parse_filetrans_result(result_data)

        elif task_status == "FAILED":
            results = output.get("results", [])
            # 记录阿里完整响应，便于定位失败原因（code/message 常为空或位于不同层级）
            logger.error("DashScope Filetrans 任务 FAILED，完整响应: %s", task_data)
            error_msg = "未知错误"
            code = ""
            if results:
                r0 = results[0]
                if isinstance(r0, dict):
                    code = r0.get("code", "") or ""
                    msg = r0.get("message") or r0.get("error") or r0.get("error_message") or ""
                    if msg:
                        error_msg = str(msg)
            if code:
                error_msg = f"{error_msg} (阿里错误码: {code})"
            raise RuntimeError(f"转写任务失败: {error_msg}")

        elif task_status in ("PENDING", "RUNNING", "QUEUED"):
            time.sleep(poll_interval)
        else:
            logger.warning("未知任务状态: %s", task_status)
            time.sleep(poll_interval)
