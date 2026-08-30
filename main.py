"""qimingxing ASR Worker 入口：多线程并发轮询抢占任务并转写。

用法:
    python main.py                 # 前台常驻运行，多线程并发
    python main.py --once          # 只处理一个任务后退出（便于调试）
    python main.py --idle-exit 30  # 连续 30 次轮询无任务后退出
                                   # （适配 GitHub Actions 等临时环境）
    python main.py --reap-only     # 只回收卡死任务后退出

并发数由 WORKER_CONCURRENCY 环境变量控制（默认 2）。
"""
import argparse
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import db
from config import settings
from streaming_pipeline import run_task_streaming

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] [%(threadName)s] %(message)s",
)
logger = logging.getLogger("worker.main")

# 全局停止信号
_stop_event = threading.Event()


def _handle_failure(task_id: str, exc: Exception) -> None:
    retries = db.get_task_retry_count(task_id)
    if retries < settings.TASK_MAX_RETRIES:
        db.bump_task_retry(task_id)
        db.mark_task_failed_for_retry(task_id, f"{type(exc).__name__}: {exc}")
        logger.warning("task=%s 失败但将重试(%d/%d): %s",
                       task_id, retries + 1, settings.TASK_MAX_RETRIES, exc)
    else:
        db.mark_task_failed(task_id, f"{type(exc).__name__}: {exc}")


def worker_loop(worker_id: int, idle_exit: int = 0) -> None:
    """单个 Worker 线程的主循环。"""
    logger.info("Worker-%d 启动", worker_id)
    idle = 0
    while not _stop_event.is_set():
        try:
            task = db.claim_task()
            if not task:
                idle += 1
                if idle_exit and idle >= idle_exit:
                    logger.info("Worker-%d 连续 %d 次轮询无任务，空闲退出", worker_id, idle)
                    break
                time.sleep(settings.POLL_INTERVAL_S)
                continue

            idle = 0
            logger.info("Worker-%d 抢到任务 task=%s，开始处理", worker_id, task["id"])
            try:
                run_task_streaming(task)
                logger.info("Worker-%d 完成任务 task=%s", worker_id, task["id"])
            except Exception as exc:  # noqa: BLE001
                _handle_failure(task["id"], exc)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Worker-%d 轮询异常: %s", worker_id, exc)
            time.sleep(settings.POLL_INTERVAL_S)

    logger.info("Worker-%d 退出", worker_id)


def main() -> None:
    if not settings.DATABASE_URL:
        raise SystemExit("缺少 DATABASE_URL，请先配置环境变量")

    parser = argparse.ArgumentParser()
    parser.add_argument("--once", action="store_true", help="只处理一个任务后退出")
    parser.add_argument("--idle-exit", type=int, default=0, metavar="N",
                        help="连续 N 次轮询无任务后退出（0 表示不退出）")
    parser.add_argument("--reap-only", action="store_true",
                        help="只回收卡死任务后退出（不处理新任务）")
    args = parser.parse_args()

    # 启动即回收可能卡死的任务（临时 Worker 被强杀的场景）
    try:
        db.reap_stale_tasks()
    except Exception as exc:  # noqa: BLE001
        logger.warning("回收卡死任务失败（忽略继续）: %s", exc)

    if args.reap_only:
        logger.info("reap-only 模式完成，退出")
        return

    if args.once:
        # 单任务模式：只处理一个任务
        logger.info("once 模式：只处理一个任务")
        task = db.claim_task()
        if task:
            try:
                run_task_streaming(task)
            except Exception as exc:  # noqa: BLE001
                _handle_failure(task["id"], exc)
        else:
            logger.info("没有待处理任务")
        return

    # 多线程并发模式
    concurrency = settings.WORKER_THREADS
    logger.info(
        "ASR Worker 启动（多线程模式，并发任务数=%d，每任务段并发=%d），轮询间隔 %.1fs，模型=%s，idle_exit=%d",
        concurrency, settings.WORKER_CONCURRENCY, settings.POLL_INTERVAL_S, settings.DASHSCOPE_MODEL, args.idle_exit,
    )

    with ThreadPoolExecutor(max_workers=concurrency, thread_name_prefix="Worker") as executor:
        futures = [
            executor.submit(worker_loop, i + 1, args.idle_exit)
            for i in range(concurrency)
        ]

        try:
            # 等待所有线程完成（正常情况下不会完成，除非 idle_exit 触发）
            for future in futures:
                future.result()
        except KeyboardInterrupt:
            logger.info("收到中断信号，正在停止所有 Worker...")
            _stop_event.set()
            # 等待线程退出（给正在处理的任务一点时间完成当前段）
            time.sleep(2)

    logger.info("所有 Worker 已退出")


if __name__ == "__main__":
    main()
