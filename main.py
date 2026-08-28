"""qimingxing ASR Worker 入口：常驻轮询抢占任务并转写。

用法:
    python main.py                 # 前台常驻运行
    python main.py --once          # 只处理一个任务后退出（便于调试）
    python main.py --idle-exit 30  # 连续 30 次轮询无任务后退出
                                   # （适配 GitHub Actions 等临时环境）
    python main.py --reap-only     # 只回收卡死任务后退出
"""
import argparse
import logging
import time

import db
from config import settings
from pipeline import run_task

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger("worker.main")


def _handle_failure(task_id: str, exc: Exception) -> None:
    retries = db.get_task_retry_count(task_id)
    if retries < settings.TASK_MAX_RETRIES:
        db.bump_task_retry(task_id)
        db.mark_task_failed_for_retry(task_id, f"{type(exc).__name__}: {exc}")
        logger.warning("task=%s 失败但将重试(%d/%d): %s",
                       task_id, retries + 1, settings.TASK_MAX_RETRIES, exc)
    else:
        db.mark_task_failed(task_id, f"{type(exc).__name__}: {exc}")


def run_once() -> bool:
    """抢一个任务并处理。返回是否处理了任务。"""
    task = db.claim_task()
    if not task:
        return False
    try:
        run_task(task)
    except Exception as exc:  # noqa: BLE001
        _handle_failure(task["id"], exc)
    return True


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

    logger.info("ASR Worker 启动，轮询间隔 %.1fs，模型=%s，idle_exit=%d",
                settings.POLL_INTERVAL_S, settings.DASHSCOPE_MODEL, args.idle_exit)
    idle = 0
    while True:
        try:
            handled = run_once()
        except Exception as exc:  # noqa: BLE001
            logger.exception("轮询异常: %s", exc)
            handled = False
        if args.once:
            break
        if not handled:
            idle += 1
            if args.idle_exit and idle >= args.idle_exit:
                logger.info("连续 %d 次轮询无任务，空闲退出", idle)
                break
            time.sleep(settings.POLL_INTERVAL_S)
        else:
            idle = 0


if __name__ == "__main__":
    main()
