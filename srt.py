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


def build_srt(
    segments: list[tuple[int, float, float, str]],
    speakers: list[str] | None = None,
) -> str:
    """segments: [(index, start_s, end_s, text), ...] -> SRT 全文。

    Args:
        segments: 片段列表
        speakers: 每个片段的说话人标签（可选，长度与 segments 一致）
    """
    lines = []
    sorted_segs = sorted(segments, key=lambda x: x[0])
    for i, (idx, start, end, text) in enumerate(sorted_segs):
        if not text:
            continue
        lines.append(str(idx + 1))
        lines.append(f"{_ts(start)} --> {_ts(end)}")
        # 添加说话人标签
        if speakers and i < len(speakers) and speakers[i]:
            lines.append(f"[{speakers[i]}] {text.strip()}")
        else:
            lines.append(text.strip())
        lines.append("")
    return "\n".join(lines)
