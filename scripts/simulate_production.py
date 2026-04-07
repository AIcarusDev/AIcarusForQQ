#!/usr/bin/env python3
"""simulate_production.py — 生产环境模拟

向数据库写入测试记忆三元组，展示真实 system prompt（含 <active_memory> 注入），
然后向 siliconflow 发出真实 LLM API 调用并打印原始响应。

API Key 从项目根目录 .env 文件读取 (SILICONFLOW_API_KEY)。

用法:
    python scripts/simulate_production.py
    python scripts/simulate_production.py --no-llm    # 只看 prompt，不发请求
    python scripts/simulate_production.py --keep       # 保留临时 DB
    python scripts/simulate_production.py --verbose    # 启用 DEBUG 日志
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import re
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from zoneinfo import ZoneInfo

# ── 路径配置（在所有业务导入之前完成）────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

# 静默 jieba 日志（含 StreamHandler 直接往 stderr 打印的部分）
logging.getLogger("jieba").setLevel(logging.WARNING)
import jieba as _jieba_pre  # noqa: E402
_jieba_pre.setLogLevel(logging.WARNING)

# ── 加载 .env ──────────────────────────────────────────────────────
from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")

# ── 业务模块（路径已就绪后才导入）────────────────────────────────────
import app_state  # noqa: E402
import database  # noqa: E402
from config_loader import load_config  # noqa: E402
from llm.core.provider import create_adapter  # noqa: E402
from llm.core.llm_core import call_model_and_process  # noqa: E402
from llm.memory_tokenizer import load_custom_dict_from_triples, tokenize as _tokenize  # noqa: E402
from llm.prompt.memory import configure as configure_memory, restore as restore_memory  # noqa: E402
from llm.session import ChatSession, sessions as global_sessions  # noqa: E402


# ─────────────────────────────────────────────────────────────────────────────
# 工具函数
# ─────────────────────────────────────────────────────────────────────────────

def _banner(title: str) -> None:
    w = 72
    print(f"\n{'═' * w}")
    print(f"  {title}")
    print(f"{'═' * w}")


async def _write_test_triples(triples: list[dict]) -> list[int]:
    """批量写入测试三元组，返回 id 列表。"""
    ids: list[int] = []
    for t in triples:
        tok = _tokenize(t["object_text"])
        row_id = await database.write_triple(
            subject=t["subject"],
            predicate=t["predicate"],
            object_text=t["object_text"],
            object_text_tok=tok,
            source=t.get("source", "simulate_production"),
            reason=t.get("reason", ""),
            conv_type=t.get("conv_type", "group"),
            conv_id=t.get("conv_id", "test_group_001"),
            conv_name=t.get("conv_name", "测试群"),
            confidence=t.get("confidence", 0.8),
        )
        ids.append(row_id)
    return ids


# ─────────────────────────────────────────────────────────────────────────────
# 主流程
# ─────────────────────────────────────────────────────────────────────────────

async def main(args: argparse.Namespace) -> None:
    db_path: str | None = None

    try:
        # ── §1 加载配置 ─────────────────────────────────────────────────────
        _banner("§1  加载配置")
        config, persona, instructions = load_config()
        print(f"  provider    : {config.get('provider')}")
        print(f"  model       : {config.get('model')}")
        print(f"  model_name  : {config.get('model_name')}")
        print(f"  bot_name    : {config.get('bot_name')}")
        print(f"  timezone    : {config.get('timezone')}")
        print(f"  persona     : {len(persona)} 字节")
        print(f"  instructions: {len(instructions)} 字节")

        # ── §2 初始化 app_state ──────────────────────────────────────────────
        _banner("§2  初始化 app_state")
        tz = ZoneInfo(config.get("timezone", "Asia/Shanghai"))
        app_state.config = config
        app_state.persona = persona
        app_state.instructions = instructions
        app_state.MODEL = config.get("model", "")
        app_state.MODEL_NAME = config.get("model_name", "")
        app_state.GEN = config.get("generation", {})
        app_state.TIMEZONE = tz
        app_state.BOT_NAME = config.get("bot_name", "吹雪")
        app_state.MAX_CONTEXT = config.get("max_context", 20)
        app_state.napcat_client = None
        app_state.vision_bridge = None
        app_state.rate_limiter = None
        app_state.main_loop = asyncio.get_event_loop()

        app_state.adapter = create_adapter(config)
        print(f"  adapter     : {type(app_state.adapter).__name__}")
        print(f"  adapter.provider : {app_state.adapter.provider}")
        print(f"  adapter.model    : {app_state.adapter.model}")

        # 检查 API Key 是否存在
        api_key = os.getenv("SILICONFLOW_API_KEY", "")
        if not api_key:
            print("  ⚠️  警告: 未找到 SILICONFLOW_API_KEY，LLM 调用将失败")
        else:
            print(f"  API Key     : {api_key[:8]}…（已隐去后续字符）")

        # ── §3 临时数据库 ────────────────────────────────────────────────────
        _banner("§3  初始化临时数据库")
        if args.keep:
            db_path = str(ROOT / "data" / "simulate_production.db")
            print(f"  DB 路径  : {db_path}")
            print(f"  模式     : --keep 模式，运行后不清理")
        else:
            tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
            db_path = tmp.name
            tmp.close()
            print(f"  DB 路径  : {db_path}")
            print(f"  模式     : 临时文件，运行结束后自动清理")

        database.DB_PATH = db_path
        await database.init_db()
        print("  DB 初始化完毕")

        # ── §4 写入测试记忆三元组 ────────────────────────────────────────────
        _banner("§4  写入测试记忆三元组")
        TEST_USER_ID = "10001"
        TEST_SUBJECT = f"User:qq_{TEST_USER_ID}"

        test_triples: list[dict] = [
            {
                "subject": TEST_SUBJECT,
                "predicate": "喜欢",
                "object_text": "打游戏，尤其喜欢角色扮演类游戏",
                "reason": "用户自述",
            },
            {
                "subject": TEST_SUBJECT,
                "predicate": "养了",
                "object_text": "一只橘猫，名字叫小橘",
                "reason": "聊天中提到",
            },
            {
                "subject": TEST_SUBJECT,
                "predicate": "职业是",
                "object_text": "后端程序员，主要写 Python 和 Go",
                "reason": "用户自我介绍",
            },
            {
                "subject": TEST_SUBJECT,
                "predicate": "讨厌",
                "object_text": "早起，经常熬夜写代码到凌晨",
                "reason": "聊天中多次提到",
            },
            {
                "subject": "Self",
                "predicate": "[note]",
                "object_text": "这是一次生产环境模拟测试，当前群是测试群，用于验证记忆注入流程",
                "reason": "脚本注入",
            },
        ]

        ids = await _write_test_triples(test_triples)
        for i, (t, tid) in enumerate(zip(test_triples, ids)):
            obj_preview = t["object_text"][:40]
            print(f"  [{i+1}] id={tid:3d}  {t['subject']:20s} /{t['predicate']:6s}/  {obj_preview}")

        # ── §5 恢复内存缓存 + jieba 词典 ─────────────────────────────────────
        _banner("§5  恢复记忆缓存")
        memory_cfg = config.get("memory", {})
        max_entries = int(memory_cfg.get("max_entries", 15))
        configure_memory(max_entries)

        rows = await database.load_all_triples()
        restore_memory(rows)
        load_custom_dict_from_triples(rows)
        print(f"  已恢复 {len(rows)} 条三元组，jieba 词典已同步")

        # ── §6 构建 ChatSession ──────────────────────────────────────────────
        _banner("§6  构建 ChatSession（伪造对话上下文）")
        now_iso = datetime.now(tz).isoformat()

        session = ChatSession()
        session._persona = persona
        session._instructions = instructions
        session._model_name = app_state.MODEL_NAME
        session._qq_id = "fake_bot_777"
        session._qq_name = app_state.BOT_NAME
        session._qq_card = ""
        session._guardian_name = ""
        session._guardian_id = ""
        session._timezone = tz
        session._max_context = app_state.MAX_CONTEXT
        session.conv_type = "group"
        session.conv_id = "test_group_001"
        session.conv_name = "测试群"
        session.recalled_memories = []
        session.quoted_extra = {}

        user_msg = "你还记得我喜欢什么、养了什么吗？"
        session.context_messages = [
            {
                "role": "user",
                "message_id": 20010,
                "sender_id": TEST_USER_ID,
                "sender_name": "小明",
                "sender_role": "member",
                "timestamp": now_iso,
                "content": user_msg,
                "content_type": "text",
                "content_segments": [{"type": "text", "text": user_msg}],
            }
        ]

        # 注册到全局会话字典（prepare_chat_log_with_unread 需要）
        _session_key = f"{session.conv_type}_{session.conv_id}"
        global_sessions[_session_key] = session
        print(f"  会话键      : {_session_key}")
        print(f"  用户消息    : {user_msg}")
        print(f"  消息条数    : {len(session.context_messages)}")

        # ── §7 FTS5 记忆召回 ─────────────────────────────────────────────────
        _banner("§7  执行 FTS5 记忆召回")
        await session.prepare_memory_recall()
        n_recalled = len(session.recalled_memories)
        print(f"  召回条数    : {n_recalled}")
        for m in session.recalled_memories:
            print(
                f"  [{m.get('id'):3}] conf={m.get('confidence', 0):.2f}  "
                f"{m.get('subject'):20s} /{m.get('predicate'):6s}/ "
                f"{m.get('object_text', '')[:45]}"
            )

        # ── §8 System Prompt 展示 ────────────────────────────────────────────
        _banner("§8  System Prompt 全文")
        sp = session.build_system_prompt()
        print(sp)

        # 单独提取 <active …>…</active> 块
        active_match = re.search(r"(<active\b[^>]*>.*?</active>)", sp, flags=re.DOTALL)
        if not active_match:
            # 自闭合形式 <active … />
            active_match = re.search(r"(<active\b[^/]*/\s*>)", sp)
        if active_match:
            _banner("§8b  <active_memory> 块（摘录）")
            print(active_match.group(1))

        if args.no_llm:
            _banner("§9  已跳过 LLM 调用（--no-llm 标志）")
            return

        # ── §9 真实 LLM 调用 ─────────────────────────────────────────────────
        _banner("§9  调用 LLM（siliconflow）")
        print("  正在发送请求，请稍候…")

        t0 = time.monotonic()
        result, grounding, _sp, chat_log_display, repaired, tool_calls_log = (
            await asyncio.to_thread(call_model_and_process, session)
        )
        elapsed = time.monotonic() - t0

        _banner("§9b  LLM 原始响应（result dict）")
        print(json.dumps(result, ensure_ascii=False, indent=2) if result else "None")

        if tool_calls_log:
            _banner("§9c  工具调用记录")
            print(json.dumps(tool_calls_log, ensure_ascii=False, indent=2))

        if repaired:
            print("\n  ⚠️  响应 JSON 已经过自动修复（repaired=True）")

        _banner("§9  调用完成")
        print(f"  耗时        : {elapsed:.2f}s")
        print(f"  result      : {'OK' if result else 'None（调用失败）'}")
        print(f"  工具调用次数: {len(tool_calls_log)}")

        # ── §10 自动归档（archive_turn_memories 真实调用）────────────────────
        if not args.no_archiver:
            _banner("§10  自动归档演示（archive_turn_memories）")
            from llm.memory_archiver import archive_turn_memories

            # 构造一段含有新用户信息的对话（和 §6 消息不同，模拟下一轮聊天）
            arch_msgs = [
                {
                    "role": "user",
                    "sender_name": "小明",
                    "content": "对了，我最近换工作了，现在在一家 AI 初创公司做技术负责人",
                },
                {
                    "role": "bot",
                    "content": "哇，技术负责人！恭喜你升职～",
                },
                {
                    "role": "user",
                    "sender_name": "小明",
                    "content": "而且今年开始学吉他了，每天练半小时，感觉还挺有趣的",
                },
                {
                    "role": "bot",
                    "content": "吉他入门要坚持，加油！",
                },
            ]

            arch_session = SimpleNamespace(
                context_messages=arch_msgs,
                conv_type="group",
                conv_id="test_group_001",
                conv_name="测试群",
            )

            before_rows = await database.load_all_triples()
            before_ids  = {r["id"] for r in before_rows}
            print(f"  DB 当前记忆条数  : {len(before_rows)}")

            # 先单独调用 one_shot_json 展示原始响应（可观测中间结果）
            from llm.memory_archiver import _EXTRACT_SYSTEM
            dialogue_lines = []
            for m in arch_msgs:
                role = m.get("role", "")
                text = m.get("content", "")
                if role == "user":
                    dialogue_lines.append(f"User({m.get('sender_name','User')}): {text}")
                elif role == "bot":
                    dialogue_lines.append(f"Bot: {text}")
            dialogue_preview = "\n".join(dialogue_lines)
            print(f"\n  【待归档对话】\n{chr(10).join('    ' + l for l in dialogue_preview.splitlines())}\n")

            print("  正在调用 one_shot_json（原始响应预览）…")
            t_raw = time.monotonic()
            raw_resp = await asyncio.to_thread(
                app_state.adapter.one_shot_json, _EXTRACT_SYSTEM, dialogue_preview
            )
            print(f"  耗时 {time.monotonic() - t_raw:.2f}s  →  {json.dumps(raw_resp, ensure_ascii=False) if raw_resp is not None else 'None'}")
            print()

            print("  正在调用 archive_turn_memories（完整写入流程）…")

            t_arch = time.monotonic()
            await archive_turn_memories(arch_session, TEST_USER_ID, tool_calls_log)
            arch_elapsed = time.monotonic() - t_arch

            after_rows = await database.load_all_triples()
            new_rows   = [r for r in after_rows if r["id"] not in before_ids]

            print(f"  归档耗时         : {arch_elapsed:.2f}s")
            print(f"  新写入条数       : {len(new_rows)}（DB 共 {len(after_rows)} 条）")
            if new_rows:
                print()
                for r in new_rows:
                    print(
                        f"  + id={r['id']:3d}  [{r['subject']}]"
                        f"  {r['predicate']} → {r['object_text']!r}"
                        f"  (source={r['source']!r})"
                    )
            else:
                print("  （未提取到新记忆）")
        else:
            _banner("§10  已跳过自动归档演示（--no-archiver 标志）")

    finally:
        if not args.keep and db_path:
            try:
                os.unlink(db_path)
                print(f"\n  [已清理临时 DB: {db_path}]")
            except OSError:
                pass


# ─────────────────────────────────────────────────────────────────────────────
# 入口
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="模拟生产环境：注入记忆三元组 → 展示 system prompt → 发出真实 LLM 调用",
    )
    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="只展示 system prompt，跳过 LLM 请求（节省 API 费用）",
    )
    parser.add_argument(
        "--keep",
        action="store_true",
        help="保留临时 DB（默认运行结束后自动删除）",
    )
    parser.add_argument(
        "--no-archiver",
        action="store_true",
        help="跳过 §10 自动归档演示（不发出额外 LLM 请求）",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="启用 DEBUG 日志",
    )
    args = parser.parse_args()
    # --no-llm 隐含 --no-archiver（archiver 也需要 LLM API）
    if args.no_llm:
        args.no_archiver = True

    log_level = logging.DEBUG if args.verbose else logging.WARNING
    logging.basicConfig(
        level=log_level,
        format="%(levelname)-8s  %(name)s: %(message)s",
    )

    asyncio.run(main(args))
