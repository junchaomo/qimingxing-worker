"""DashScope / Qwen-ASR 调用层：同步多模态接口 + 全局限流 + 指数退避重试。

输入为 16kHz 单声道 WAV 段（本地路径），单段 ≤ MAX_SEGMENT_S。
当前适配模型: qwen-audio-3.0-asr-flash（qwen3-asr-flash 已标记即将部分下线）。

请求要点：
  - sk-ws- 工作空间专属 Key 必须配合业务空间专属网关域名（DASHSCOPE_BASE_URL）；
  - 音频以 base64 data URI 形式放在 input_audio.data 中；
  - parameters 需带 format / sample_rate，与 audio.py 转换输出一致；
  - 响应文本位于 output.text（旧格式的 output.choices 不再返回，保留解析兜底）。
"""
import base64
import logging
import threading
import time

import requests

from config import settings

logger = logging.getLogger("worker.asr_client")

# 全局并发护栏：限制同时进行的 API 请求总数，防触发 DashScope 限流/封禁
_global_semaphore = threading.BoundedSemaphore(settings.GLOBAL_API_SEMAPHORE)


def _wav_to_data_uri(wav_path: str) -> str:
    with open(wav_path, "rb") as f:
        raw = f.read()
    return f"data:audio/{settings.ASR_AUDIO_FORMAT};base64," + base64.b64encode(raw).decode()


def _call_once(wav_path: str, language: str | None) -> dict:
    payload = {
        "model": settings.DASHSCOPE_MODEL,
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
    # qwen-audio-3.0-asr-flash 支持 30 语种自动识别；如传入 language 则作为可选约束下发
    if language:
        payload["parameters"]["asr_options"] = {"language": language}

    headers = {
        "Authorization": f"Bearer {settings.DASHSCOPE_API_KEY}",
        "Content-Type": "application/json",
        "X-DashScope-SSE": "disable",
    }
    resp = requests.post(
        settings.DASHSCOPE_BASE_URL,
        json=payload,
        headers=headers,
        timeout=settings.API_TIMEOUT_S,
    )
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
        # 时间单位可能是毫秒或秒，根据数值判断
        if start > 1000:
            start = start / 1000.0
            end = end / 1000.0
        words.append((float(start), float(end), str(text)))
    return words


def _parse(data: dict) -> tuple[str, str | None, list[tuple[float, float, str]]]:
    """从响应中解析 (text, language, words)。

    words 格式：[(start_s, end_s, word), ...]，时间相对于该段开始。
    """
    # 解析词级时间戳
    sentence = data.get("sentence") or (data.get("output") or {}).get("sentence")
    words = _parse_words(sentence)

    # 新格式：顶层 / output.text
    text = data.get("text") or (data.get("output") or {}).get("text") or ""
    if text:
        return text, None, words

    # 旧格式兜底：output.choices[0].message.content[0].text
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
    """转写单个段，带全局限流与指数退避重试。返回 (text, language, words)。"""
    if not settings.DASHSCOPE_API_KEY:
        raise RuntimeError("DASHSCOPE_API_KEY 未配置")

    with _global_semaphore:
        last_exc: Exception | None = None
        for attempt in range(settings.SEGMENT_MAX_RETRIES):
            try:
                data = _call_once(wav_path, language)
                return _parse(data)
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                wait = 2 ** attempt  # 1s -> 2s -> 4s
                logger.warning("segment 调用失败(第%d次): %s，%.0fs 后重试",
                               attempt + 1, exc, wait)
                time.sleep(wait)
        raise RuntimeError(f"段转写重试耗尽: {last_exc}")
