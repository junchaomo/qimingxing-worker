"""SRT 字幕生成：以 VAD 段的起止时间为时间轴。"""
import logging

logger = logging.getLogger("worker.srt")


def _ts(seconds: float) -> str:
    """秒 -> SRT 时间格式 00:00:00,000"""
    ms = int(round(seconds * 1000))
    h, rem = divmod(ms, 3600_000)
    m, rem = divmod(rem, 60_000)
    s, mss = divmod(rem, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{mss:03d}"


def build_srt(segments: list[tuple[int, float, float, str]]) -> str:
    """segments: [(index, start_s, end_s, text), ...] -> SRT 全文。"""
    lines = []
    for idx, start, end, text in sorted(segments, key=lambda x: x[0]):
        if not text:
            continue
        lines.append(str(idx + 1))
        lines.append(f"{_ts(start)} --> {_ts(end)}")
        lines.append(text.strip())
        lines.append("")
    return "\n".join(lines)
