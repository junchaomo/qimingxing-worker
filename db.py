"""数据库访问层：基于 psycopg3，支持事务与 FOR UPDATE SKIP LOCKED 抢单。"""
import logging
from contextlib import contextmanager

import psycopg
from psycopg.rows import dict_row

from config import settings

logger = logging.getLogger("worker.db")


@contextmanager
def get_conn():
    """每个调用拿一条新连接；用完自动归还/关闭。"""
    conn = psycopg.connect(settings.DATABASE_URL, row_factory=dict_row)
    try:
        yield conn
    finally:
        conn.close()


def reap_stale_tasks(stale_minutes: int = 15) -> int:
    """回收卡死的任务：把 processing 超过 stale_minutes 的任务重置回 queued。

    适用场景：临时 Worker（如 GitHub Actions）被强杀后，任务会卡在
    processing。下次任意 Worker 启动时调用本函数即可兜底恢复。
    返回回收的任务数。
    """
    sql = """
        update transcription_tasks
        set status='queued', error_message='stale task reaped', started_at=null
        where status='processing'
          and started_at < now() - make_interval(mins => %s)
        returning id
    """
    with get_conn() as conn:
        with conn.transaction():
            rows = conn.execute(sql, (stale_minutes,)).fetchall()
    if rows:
        logger.warning("reaped %d stale task(s): %s", len(rows), [r["id"] for r in rows])
    return len(rows)


def claim_task() -> dict | None:
    """抢占一个 queued 任务（多 Worker 安全）。

    使用 FOR UPDATE SKIP LOCKED：仅锁住被本进程抢到的行，
    不会阻塞其他 Worker 处理不同任务。
    """
    sql = """
        update transcription_tasks
        set status = 'processing', started_at = now()
        where id = (
            select id from transcription_tasks
            where status = 'queued'
            order by created_at
            limit 1
            for update skip locked
        )
        returning *
    """
    with get_conn() as conn:
        with conn.transaction():
            row = conn.execute(sql).fetchone()
            if row:
                logger.info("claimed task=%s", row["id"])
            return row


def mark_task_failed_for_retry(task_id: str, message: str) -> None:
    """任务失败但未超重试次数：回到 queued 等待下次抢占。"""
    sql = """
        update transcription_tasks
        set status = 'queued', error_message = %s, started_at = null
        where id = %s
    """
    with get_conn() as conn:
        with conn.transaction():
            conn.execute(sql, (message, task_id))


def mark_task_failed(task_id: str, message: str) -> None:
    with get_conn() as conn:
        with conn.transaction():
            conn.execute(
                "update transcription_tasks set status='failed', error_message=%s, completed_at=now() where id=%s",
                (message, task_id),
            )
            # 关联音频文件同步失败态
            conn.execute(
                """
                update audio_files f set status='failed'
                from transcription_tasks t
                where t.id=%s and f.id=t.audio_file_id and f.status in ('uploading','uploaded','transcribing')
                """,
                (task_id,),
            )
    logger.error("task=%s failed: %s", task_id, message)


def mark_task_completed(task_id: str, result_text: str, result_srt: str, total_segments: int) -> None:
    with get_conn() as conn:
        with conn.transaction():
            conn.execute(
                """
                update transcription_tasks
                set status='completed', result_text=%s, result_srt=%s,
                    progress=1, total_segments=%s, completed_segments=%s,
                    completed_at=now()
                where id=%s
                """,
                (result_text, result_srt, total_segments, total_segments, task_id),
            )
            # 关联音频文件完成态
            conn.execute(
                """
                update audio_files f set status='done'
                from transcription_tasks t
                where t.id=%s and f.id=t.audio_file_id
                """,
                (task_id,),
            )
    logger.info("task=%s completed", task_id)


def update_progress(task_id: str, total: int, completed: int) -> None:
    with get_conn() as conn:
        with conn.transaction():
            conn.execute(
                """
                update transcription_tasks
                set total_segments=%s, completed_segments=%s,
                    progress = case when %s = 0 then 0 else round(%s::numeric / %s, 4) end
                where id=%s
                """,
                (total, completed, total, completed, total, task_id),
            )


def get_audio_file(audio_file_id: str) -> dict | None:
    with get_conn() as conn:
        return conn.execute(
            "select * from audio_files where id=%s", (audio_file_id,)
        ).fetchone()


def update_audio_duration(audio_file_id: str, seconds: int) -> None:
    """转码后回填真实时长（上传阶段无法探测）。"""
    with get_conn() as conn:
        with conn.transaction():
            conn.execute(
                "update audio_files set duration=%s, status='transcribing' where id=%s",
                (seconds, audio_file_id),
            )


def get_task_retry_count(task_id: str) -> int:
    with get_conn() as conn:
        row = conn.execute(
            "select retry_count from transcription_tasks where id=%s",
            (task_id,),
        ).fetchone()
        return row["retry_count"] if row else 0


def bump_task_retry(task_id: str) -> None:
    with get_conn() as conn:
        with conn.transaction():
            conn.execute(
                "update transcription_tasks set retry_count = retry_count + 1 where id=%s",
                (task_id,),
            )


def insert_usage_record(user_id: str, task_id: str, seconds: int) -> None:
    with get_conn() as conn:
        with conn.transaction():
            conn.execute(
                "insert into usage_records (user_id, task_id, seconds) values (%s, %s, %s)",
                (user_id, task_id, seconds),
            )


def upsert_segment(task_id: str, idx: int, start: float, end: float, text: str, status: str) -> None:
    with get_conn() as conn:
        with conn.transaction():
            conn.execute(
                """
                insert into transcription_segments (task_id, segment_index, start_time, end_time, text, status)
                values (%s, %s, %s, %s, %s, %s)
                on conflict do nothing
                """,
                (task_id, idx, start, end, text, status),
            )
