"""DashScope / Qwen-Audio-3.0-ASR-Flash-Filetrans 异步调用层。

Filetrans 模型支持说话人分离（speaker diarization），采用异步任务模式：
1. 提交任务（POST /services/audio/asr/transcription）
2. 轮询任务状态（GET /tasks/{task_id}）
3. 下载结果 JSON（transcription_url）
4. 解析文本、句子级时间戳、说话人标签

输入为音频文件的可访问 URL（HTTP/HTTPS）。
"""
import logging
import time

import requests

from config import settings

logger = logging.getLogger("worker.asr_client")


def submit_task(file_url: str, language: str | None = None) -> str:
    """提交异步转写任务，返回 task_id。"""
    url = f"{settings.DASHSCOPE_BASE_URL}/services/audio/asr/transcription"

    parameters: dict = {}
    if settings.ENABLE_SPEAKER_DIARIZATION:
        parameters["diarization_enabled"] = True
    if language:
        parameters["language_hints"] = [language]

    payload = {
        "model": settings.DASHSCOPE_MODEL,
        "input": {
            "file_urls": [file_url],
        },
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
    """查询任务状态，返回任务详情。"""
    url = f"{settings.DASHSCOPE_BASE_URL}/tasks/{task_id}"

    headers = {
        "Authorization": f"Bearer {settings.DASHSCOPE_API_KEY}",
    }

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


def parse_result(result_data: dict) -> tuple[str, str | None, list[dict]]:
    """解析转写结果，返回 (完整文本, 语言, 句子列表)。

    每个句子包含: begin_time(s), end_time(s), text, speaker_id, words[]
    """
    transcripts = result_data.get("transcripts", [])
    if not transcripts:
        return "", None, []

    transcript = transcripts[0]
    full_text = transcript.get("text", "")
    language = transcript.get("language") or None

    sentences = []
    for sent in transcript.get("sentences", []):
        # 时间戳是毫秒，转成秒
        begin = (sent.get("begin_time") or 0) / 1000.0
        end = (sent.get("end_time") or 0) / 1000.0
        text = sent.get("text", "")
        speaker_id = sent.get("speaker_id")
        words = sent.get("words", [])

        sentences.append({
            "begin_time": begin,
            "end_time": end,
            "text": text,
            "speaker_id": speaker_id,
            "words": words,
        })

    logger.info(
        "解析结果: 文本长度=%d, 句子数=%d, 说话人数=%d",
        len(full_text),
        len(sentences),
        len(set(s["speaker_id"] for s in sentences if s["speaker_id"] is not None)),
    )
    return full_text, language, sentences


def transcribe_file(
    file_url: str,
    language: str | None = None,
    poll_interval: int = 3,
    timeout: int = 1800,
) -> tuple[str, str | None, list[dict]]:
    """完整的转写流程：提交→轮询→下载→解析。

    返回 (完整文本, 语言, 句子列表)
    """
    start_time = time.time()

    # 1. 提交任务
    task_id = submit_task(file_url, language)

    # 2. 轮询任务状态
    while True:
        elapsed = time.time() - start_time
        if elapsed > timeout:
            raise RuntimeError(f"转写超时（{timeout}秒）")

        task_data = query_task(task_id)
        output = task_data.get("output", {})
        task_status = output.get("task_status", "")

        logger.info("任务状态: %s, 已用 %.0fs", task_status, elapsed)

        if task_status == "SUCCEEDED":
            # 3. 下载结果
            results = output.get("results", [])
            if not results:
                raise RuntimeError("任务成功但无结果")

            transcription_url = results[0].get("transcription_url")
            if not transcription_url:
                raise RuntimeError(f"结果中无 transcription_url: {results[0]}")

            result_data = download_result(transcription_url)

            # 4. 解析结果
            return parse_result(result_data)

        elif task_status == "FAILED":
            results = output.get("results", [])
            error_msg = "未知错误"
            if results and results[0].get("message"):
                error_msg = results[0]["message"]
            raise RuntimeError(f"转写任务失败: {error_msg}")

        elif task_status in ("PENDING", "RUNNING", "QUEUED"):
            time.sleep(poll_interval)
        else:
            logger.warning("未知任务状态: %s", task_status)
            time.sleep(poll_interval)
