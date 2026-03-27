"""hibernate_core.py — 休眠生命周期

bot 在 watcher 阶段主动选择休眠后，由此模块接管等待阶段。
自然醒来或被取消均由此模块负责清理 app_state 与 activity_log。
"""

import asyncio
import logging

import app_state
import llm.activity_log as activity_log

logger = logging.getLogger("AICQ.hibernate")


async def run_hibernate(conv_key: str, minutes: int) -> bool:
    """执行完整的休眠等待阶段。

    调用前，调用者须已完成：
      - activity_log.close_current() 关闭前一条 watcher 记录
      - activity_log.open_entry("hibernate", hibernate_minutes=minutes)
      - app_state.watcher_hibernating = True

    返回 True  = 自然醒来，watcher 循环应继续。
    返回 False = 被取消（CancelledError），watcher 循环应退出。
    """
    try:
        await asyncio.sleep(minutes * 60)
    except asyncio.CancelledError:
        app_state.watcher_hibernating = False
        logger.info("[hibernate] 休眠期间被取消 conv=%s", conv_key)
        return False

    # 自然醒来
    app_state.watcher_hibernating = False
    await activity_log.close_current(
        end_attitude="active",
        end_action="woke_up",
        end_motivation="自然醒",
    )
    await activity_log.open_entry("watcher")
    logger.info("[hibernate] 自然醒来，重新进入窥屏 conv=%s", conv_key)
    return True
