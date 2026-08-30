"""qimingxing ASR Worker 配置。

所有配置从环境变量读取，默认值对应本地开发。
"""
import os

from dotenv import load_dotenv

load_dotenv()  # 本地开发读取 worker/.env；生产用环境变量注入


def _int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except ValueError:
        return default


def _float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, str(default)))
    except ValueError:
        return default


class Settings:
    # --- 数据库 ---
    DATABASE_URL: str = os.environ.get("DATABASE_URL", "")

    # --- Supabase Storage（下载原始音频）---
    SUPABASE_URL: str = os.environ.get("SUPABASE_URL", "")
    SUPABASE_SERVICE_ROLE_KEY: str = os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "")
    STORAGE_BUCKET: str = os.environ.get("STORAGE_BUCKET", "audio")

    # --- DashScope / Qwen-ASR ---
    DASHSCOPE_API_KEY: str = os.environ.get("DASHSCOPE_API_KEY", "")
    # 默认使用 qwen-audio-3.0-asr-flash（qwen3-asr-flash 已标记即将部分下线）
    DASHSCOPE_MODEL: str = os.environ.get("DASHSCOPE_MODEL", "qwen-audio-3.0-asr-flash")
    # sk-ws- 工作空间专属 Key 必须使用业务空间专属网关域名；生产环境用环境变量覆盖
    DASHSCOPE_BASE_URL: str = os.environ.get(
        "DASHSCOPE_BASE_URL",
        "https://dashscope.aliyuncs.com/api/v1/services/aigc/multimodal-generation/generation",
    )
    ASR_AUDIO_FORMAT: str = os.environ.get("ASR_AUDIO_FORMAT", "wav")   # 与 audio.py 转换结果一致
    ASR_SAMPLE_RATE: int = _int("ASR_SAMPLE_RATE", 16000)              # 与 audio.py 转换结果一致
    ENABLE_ITN: bool = os.environ.get("ENABLE_ITN", "true").lower() in ("1", "true", "yes")

    # --- 并发与限流 ---
    WORKER_CONCURRENCY: int = _int("WORKER_CONCURRENCY", 4)   # 每任务内并发段数
    WORKER_THREADS: int = _int("WORKER_THREADS", 2)           # Worker 进程内并发任务数
    GLOBAL_API_SEMAPHORE: int = _int("GLOBAL_API_SEMAPHORE", 16)  # 全局同时进行的 API 请求上限

    # --- VAD 分段 ---
    VAD_SEGMENT_THRESHOLD_S: int = _int("VAD_SEGMENT_THRESHOLD_S", 30)   # 目标分段时长（按句子切分）
    MAX_SEGMENT_S: int = _int("MAX_SEGMENT_S", 60)                       # 单段硬上限（1分钟，更细粒度）
    MIN_SPEECH_MS: int = _int("MIN_SPEECH_MS", 500)                      # 最短语音段
    MIN_SILENCE_MS: int = _int("MIN_SILENCE_MS", 300)                     # 静音检测阈值

    # --- 轮询与重试 ---
    POLL_INTERVAL_S: float = _float("POLL_INTERVAL_S", 5.0)  # Worker 扫描队列间隔
    SEGMENT_MAX_RETRIES: int = _int("SEGMENT_MAX_RETRIES", 3)   # 单段重试次数
    TASK_MAX_RETRIES: int = _int("TASK_MAX_RETRIES", 2)          # 任务级重试次数
    API_TIMEOUT_S: int = _int("API_TIMEOUT_S", 120)              # DashScope 单次调用超时

    # --- 本地临时目录 ---
    TMP_DIR: str = os.environ.get("TMP_DIR", os.path.join(os.path.dirname(__file__), "tmp"))

    # --- FFmpeg 定位 ---
    # 生产环境建议把 ffmpeg 装进系统 PATH；本地可显式指定 bin 目录
    FFMPEG_BIN_DIR: str = os.environ.get("FFMPEG_BIN_DIR", "")

    # --- 说话人分离（pyannote.audio，可选） ---
    ENABLE_SPEAKER_DIARIZATION: bool = os.environ.get("ENABLE_SPEAKER_DIARIZATION", "true").lower() in ("1", "true", "yes")
    HUGGINGFACE_TOKEN: str = os.environ.get("HUGGINGFACE_TOKEN", "")
    # 超过此时长的音频跳过说话人分离（CPU 运行较慢）
    MAX_DIARIZATION_DURATION_S: int = _int("MAX_DIARIZATION_DURATION_S", 3600)
    # 说话人分离超时时间
    DIARIZATION_TIMEOUT_S: int = _int("DIARIZATION_TIMEOUT_S", 600)


settings = Settings()
