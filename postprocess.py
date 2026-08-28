"""结果后处理：按段序合并、去幻觉/重复清洗。"""
import logging
import re

logger = logging.getLogger("worker.postprocess")


def merge_text(results: list[tuple[int, str]]) -> str:
    """按 segment_index 升序拼接全文。results: [(idx, text), ...]"""
    results.sort(key=lambda x: x[0])
    return " ".join(text for _, text in results)


def clean_text(text: str) -> str:
    """清洗常见 ASR 幻觉与重复：
    - 连续重复片段（同一子串出现 ≥3 次）折叠为 1 次
    - 去除大量无意义语气词循环（嗯嗯嗯/啊啊啊）
    - 规整空白与标点前空格
    """
    if not text:
        return text

    # 连续重复词/短语折叠：形如 "AB AB AB ..." 折叠成 "AB"
    pattern = re.compile(r"(.{2,40}?)(?:\s+\1){2,}")
    prev = None
    while prev != text:
        prev = text
        text = pattern.sub(r"\1", text)

    # 语气词循环：三个以上重复语气词折叠为单个
    text = re.sub(r"(嗯|啊|呃|哦|额|哼){3,}", r"\1", text)

    # 规整空白
    text = re.sub(r"\s+", " ", text).strip()
    return text


def aggregate_language(languages: list[str | None]) -> str | None:
    """取众数作为整段语言判定；空则 None。"""
    from collections import Counter
    counts = Counter(l for l in languages if l)
    return counts.most_common(1)[0][0] if counts else None
