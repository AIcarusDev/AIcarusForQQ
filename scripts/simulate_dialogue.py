"""scripts/simulate_dialogue.py — 无 QQ 连接的对话仿真测试环境

读取一个 YAML 剧本文件，在隔离的临时数据库中完整跑通：
  1. 预填记忆（seed_memories）
  2. 逐 Turn 推进对话上下文
  3. bot_turn：召回记忆 → 主模型响应 → 归档记忆提取
  4. 输出结构化检查报告

用法（在项目根目录执行）:
    python scripts/simulate_dialogue.py scripts/dialogues/basic_memory.yaml
    python scripts/simulate_dialogue.py scripts/dialogues/basic_memory.yaml --keep-db
    python scripts/simulate_dialogue.py scripts/dialogues/basic_memory.yaml --archive-only
    python scripts/simulate_dialogue.py scripts/dialogues/basic_memory.yaml --output report.txt
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import textwrap
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# ── 路径 ──────────────────────────────────────────────────────────────────────
_SCRIPT_DIR = Path(__file__).resolve().parent
_ROOT = _SCRIPT_DIR.parent
_SRC = _ROOT / "src"
sys.path.insert(0, str(_SRC))

# dotenv
try:
    from dotenv import load_dotenv
    load_dotenv(_ROOT / ".env")
except ImportError:
    pass

import yaml

# ── 强制 stdout/stderr 使用 UTF-8，避免 GBK 终端崩溃 ─────────────────────────
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
if hasattr(sys.stderr, "reconfigure"):
    try:
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# ── 颜色输出 ──────────────────────────────────────────────────────────────────
_USE_COLOR = sys.stdout.isatty()


def _c(code: str, text: str) -> str:
    if not _USE_COLOR:
        return text
    return f"\033[{code}m{text}\033[0m"


def _h1(text: str) -> str:
    bar = "═" * 60
    return _c("1;36", f"\n{bar}\n  {text}\n{bar}")


def _h2(text: str) -> str:
    return _c("1;33", f"\n{'─' * 50}\n  {text}\n{'─' * 50}")


def _h3(text: str) -> str:
    return _c("1;34", f"  ── {text}")


def _ok(text: str) -> str:
    return _c("32", text)


def _warn(text: str) -> str:
    return _c("33", text)


def _err(text: str) -> str:
    return _c("31", text)


def _dim(text: str) -> str:
    return _c("2", text)


# ── 全局输出缓冲（同时写终端 + 可选文件）────────────────────────────────────────
_output_lines: list[str] = []
_output_file: str | None = None


def _print(*args, sep=" ", end="\n") -> None:
    line = sep.join(str(a) for a in args) + end
    sys.stdout.write(line)
    sys.stdout.flush()
    _output_lines.append(line)


def _flush_output_to_file(path: str) -> None:
    raw = "".join(_output_lines)
    # 去掉 ANSI 转义码
    import re
    clean = re.sub(r"\033\[[0-9;]*m", "", raw)
    Path(path).write_text(clean, encoding="utf-8")
    print(f"\n报告已保存至: {path}")


# ── 数据库隔离 ────────────────────────────────────────────────────────────────

def _create_sim_db_path() -> str:
    ts = int(time.time())
    name = f"sim_{ts}_{uuid.uuid4().hex[:6]}.db"
    data_dir = _ROOT / "data"
    data_dir.mkdir(exist_ok=True)
    return str(data_dir / name)


def _patch_db_path(db_path: str) -> None:
    """将 database 模块的 DB_PATH 替换为仿真专用路径。必须在 import database 后调用。"""
    import database
    database.DB_PATH = db_path
    # _connect() 内部直接引用 DB_PATH 模块变量，不需要其他补丁。


# ── app_state 初始化 ──────────────────────────────────────────────────────────

def _init_app_state(config: dict, scenario: dict) -> None:
    """将 app_state 初始化为仿真所需的最小可运行状态。"""
    from zoneinfo import ZoneInfo
    import app_state
    from consciousness import ConsciousnessFlow
    from llm.core.provider import (
        create_adapter,
        build_archiver_adapter_cfg,
        build_is_adapter_cfg,
    )
    from llm.core.profiles import normalize_profile_config_inplace
    from llm.core.rate_limiter import MinuteRateLimiter

    normalize_profile_config_inplace(config)

    bot_cfg = scenario.get("bot", {})
    bot_id = str(bot_cfg.get("id", "100000"))
    bot_name = str(bot_cfg.get("name", config.get("bot_name", "Bot")))

    app_state.config = config
    app_state.persona = _load_persona()
    app_state.style_prompt = _load_file("config/style.md", "")
    app_state.social_tips_private = _load_file("config/social_tips/private.md", "")
    app_state.social_tips_group = _load_file("config/social_tips/group.md", "")
    app_state.MODEL = config.get("model", "")
    app_state.MODEL_NAME = config.get("model_name", app_state.MODEL)
    app_state.GEN = config.get("generation", {})
    tz_str = (config.get("timezone") or "").strip() or "Asia/Shanghai"
    app_state.TIMEZONE = ZoneInfo(tz_str)
    app_state.MAX_CALLS_PER_MINUTE = config.get("max_calls_per_minute", 15)
    app_state.MAX_CONTEXT = int(config.get("max_context", 20))
    app_state.BOT_NAME = bot_name
    app_state.qq_adapter_client = None  # 仿真模式下无 QQ 连接
    app_state.consciousness_flow = ConsciousnessFlow()
    app_state.rate_limiter = MinuteRateLimiter(app_state.MAX_CALLS_PER_MINUTE)
    app_state.vision_bridge = None
    app_state.archive_tasks = set()

    # 主模型适配器
    try:
        app_state.adapter = create_adapter(config)
        _print(_ok(f"  [✓] 主模型适配器: {app_state.adapter.model} ({app_state.adapter.provider})"))
    except Exception as e:
        _print(_err(f"  [✗] 主模型适配器初始化失败: {e}"))
        app_state.adapter = None

    # archiver 适配器（config 里的键名是 memory.auto_archive）
    archiver_cfg = config.get("memory", {}).get("auto_archive", {})
    app_state.archiver_cfg = archiver_cfg
    if archiver_cfg.get("provider") and archiver_cfg.get("model"):
        try:
            app_state.archiver_adapter = create_adapter(
                build_archiver_adapter_cfg(config, archiver_cfg)
            )
            _print(_ok(f"  [✓] 记忆提取适配器: {app_state.archiver_adapter.model} ({app_state.archiver_adapter.provider})"))
        except Exception as e:
            _print(_warn(f"  [!] 记忆提取适配器初始化失败: {e}"))
            app_state.archiver_adapter = None
    else:
        _print(_warn("  [!] 记忆提取适配器未配置 (memory.auto_archive.provider/model)"))
        app_state.archiver_adapter = None

    # IS 适配器（可选）
    is_cfg = config.get("is", {})
    if is_cfg.get("enabled", True):
        try:
            app_state.is_adapter = create_adapter(build_is_adapter_cfg(config, is_cfg))
        except Exception:
            app_state.is_adapter = None
    else:
        app_state.is_adapter = None

    # session 全局默认值
    from llm.session import init_session_globals, update_bot_info
    init_session_globals(
        max_context=app_state.MAX_CONTEXT,
        timezone=app_state.TIMEZONE,
        persona=app_state.persona,
        model_name=app_state.MODEL_NAME,
        guardian_name=config.get("guardian_name", ""),
        guardian_id=config.get("guardian_id", ""),
        style_prompt=app_state.style_prompt,
        social_tips_private=app_state.social_tips_private,
        social_tips_group=app_state.social_tips_group,
    )
    update_bot_info(bot_id, bot_name)


def _load_persona() -> str:
    return _load_file("config/persona.md", "你是一个友好的 AI 助手。")


def _load_file(rel_path: str, default: str = "") -> str:
    full = _ROOT / rel_path
    if full.exists():
        return full.read_text(encoding="utf-8")
    return default


# ── 会话创建 ──────────────────────────────────────────────────────────────────

def _create_sim_session(scenario: dict):
    """根据剧本创建 ChatSession 并设置会话元信息。"""
    from llm.session import create_session

    session = create_session()
    conv = scenario.get("conversation", {})
    session.set_conversation_meta(
        conv_type=conv.get("type", "group"),
        conv_id=str(conv.get("id", "999999")),
        conv_name=conv.get("name", "仿真群"),
        member_count=len(scenario.get("members", [])),
    )
    return session


# ── 记忆预填 ─────────────────────────────────────────────────────────────────

async def _seed_memories(seed_list: list[dict]) -> list[int]:
    """将 seed_memories 写入数据库，返回写入的 event_id 列表。"""
    from memory.repo.events import write_event

    written_ids: list[int] = []
    for seed in seed_list:
        try:
            eid = await write_event(
                event_type=seed.get("event_type", "be"),
                summary=seed.get("summary", ""),
                modality=seed.get("modality", "actual"),
                confidence=float(seed.get("confidence", 0.9)),
                context_type=seed.get("context_type", "episodic"),
                recall_scope=seed.get("recall_scope", "global"),
                source="seed",
                reason="预填种子记忆",
                conv_type=seed.get("conv_type", ""),
                conv_id=seed.get("conv_id", ""),
                conv_name=seed.get("conv_name", ""),
                roles=seed.get("roles", []),
            )
            written_ids.append(eid)
        except Exception as e:
            _print(_warn(f"  [!] 预填记忆失败: {seed.get('summary','?')[:40]} — {e}"))
    return written_ids


# ── 上下文消息构建 ─────────────────────────────────────────────────────────────

_msg_counter = 0


def _next_msg_id(prefix: str = "sim") -> str:
    global _msg_counter
    _msg_counter += 1
    return f"{prefix}_{_msg_counter:04d}"


def _now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat()


def _build_user_entry(turn: dict, members: dict[str, dict]) -> dict:
    """构造一条用户消息 context entry。"""
    sender_id = str(turn.get("sender_id", "10001"))
    member = members.get(sender_id, {})
    nickname = turn.get("nickname") or member.get("nickname", f"user_{sender_id}")
    group_role = member.get("role", "member")
    content = str(turn.get("content", ""))

    return {
        "role": "user",
        "sender_id": sender_id,
        "sender_name": nickname,
        "group_role": group_role,
        "content_type": "text",
        "content": content,
        "content_segments": [{"type": "text", "text": content}],
        "timestamp": _now_iso(),
        "message_id": _next_msg_id("u"),
    }


def _build_bot_entry(text: str) -> dict:
    """构造一条 bot 发言的 context entry。"""
    return {
        "role": "bot",
        "content_type": "text",
        "content": text,
        "content_segments": [{"type": "text", "text": text}],
        "timestamp": _now_iso(),
        "message_id": _next_msg_id("b"),
    }


# ── Bot 响应 ──────────────────────────────────────────────────────────────────

def _extract_sent_messages(tool_calls_log: list[dict]) -> list[str]:
    """从 tool_calls_log 中提取 send_message / send_short_message 的文本内容。"""
    texts: list[str] = []
    for call in tool_calls_log:
        fn = call.get("function", "")
        if fn not in ("send_message", "send_short_message"):
            continue
        args = call.get("args", {})
        # send_message: args["messages"] = [{"command": "text", "params": {"content": "..."}}]
        for seg in args.get("messages", []):
            cmd = seg.get("command", "")
            params = seg.get("params", {})
            if cmd == "text":
                t = params.get("content", "")
                if t:
                    texts.append(t)
            elif cmd == "sticker":
                texts.append(f"[表情包 id={params.get('sticker_id', '?')}]")
            elif cmd == "at":
                texts.append(f"[@{params.get('user_id', '?')}]")
            elif cmd in ("image", "voice"):
                texts.append(f"[{cmd}]")
        # send_short_message: args["content"]
        if "content" in args:
            c = args.get("content", "")
            if c:
                texts.append(str(c))
    return texts


def _run_bot_round(session, archive_only: bool) -> dict:
    """同步执行一轮主模型调用，返回结果摘要。

    在 archive_only 模式下跳过主模型调用，仅做召回查询。
    返回字段:
      sent_texts   — bot 发出的文本列表
      tool_calls   — 所有工具调用摘要
      system_prompt — 本轮 system prompt
      cognition    — 思考链（如有）
      failed       — bool
    """
    import app_state
    from tools import build_tools
    from llm.prompt.user_prompt_builder import build_main_user_prompt

    if archive_only or app_state.adapter is None:
        return {
            "sent_texts": [],
            "tool_calls": [],
            "system_prompt": "",
            "cognition": "",
            "failed": True,
            "skipped": True,
        }

    tool_collection = build_tools(
        app_state.config,
        qq_adapter_client=None,  # 仿真模式无 QQ 连接
        group_id=session.conv_id if session.conv_type == "group" else None,
        user_id=(
            int(session.conv_id)
            if session.conv_type == "private" and session.conv_id.isdigit()
            else None
        ),
        session=session,
        vision_bridge=None,
        provider=app_state.adapter.provider if app_state.adapter else "unknown",
    )

    def system_prompt_builder(activated_names=None, latent_names=None):
        return session.build_system_prompt(
            activated_names=activated_names,
            latent_names=latent_names,
        )

    chat_log = build_main_user_prompt(session)

    result = app_state.adapter.call_one_round(
        system_prompt_builder,
        chat_log,
        app_state.GEN,
        tool_collection,
        app_state.consciousness_flow,
        None,
    )

    sent_texts = _extract_sent_messages(result.tool_calls_log)
    tool_calls = [
        {"function": c.get("function", "?"), "args": c.get("args", {})}
        for c in result.tool_calls_log
    ]

    return {
        "sent_texts": sent_texts,
        "tool_calls": tool_calls,
        "system_prompt": result.system_prompt,
        "cognition": result.cognition,
        "failed": result.failed,
        "skipped": False,
        "prompt_tokens": result.prompt_tokens,
        "output_tokens": result.output_tokens,
    }


# ── 归档结果查询 ──────────────────────────────────────────────────────────────

async def _get_all_events() -> list[dict]:
    """从数据库读取所有未删除的记忆事件（含角色边）。"""
    import aiosqlite
    import database

    events: list[dict] = []
    async with aiosqlite.connect(database.DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(
            """SELECT e.event_id, e.event_type, e.summary, e.modality,
                      e.confidence, e.context_type, e.recall_scope,
                      e.occurred_at, e.occurrences, e.source,
                      e.conv_type, e.conv_id, e.supersedes
               FROM MemoryEvents e
               WHERE e.is_deleted = 0
               ORDER BY e.event_id"""
        ) as cur:
            rows = await cur.fetchall()

        for row in rows:
            ev = dict(row)
            # 角色边
            async with db.execute(
                "SELECT role, entity, value_text, target_event FROM MemoryRoles WHERE event_id = ?",
                (ev["event_id"],),
            ) as rcur:
                ev["roles"] = [dict(r) for r in await rcur.fetchall()]
            events.append(ev)

    return events


async def _get_events_since(before_count: int) -> list[dict]:
    """获取当前 DB 中 event_id > before_count（通过计数逼近）的新事件。"""
    all_evs = await _get_all_events()
    # before_count 是写入前的总数，取最后几条
    return all_evs[before_count:]


# ── 报告打印 ──────────────────────────────────────────────────────────────────

def _print_recall(recalled: list[dict]) -> None:
    if not recalled:
        _print(_dim("  (无召回记忆)"))
        return
    for ev in recalled:
        conf = float(ev.get("confidence", 0))
        summary = ev.get("summary", "")
        eid = ev.get("event_id", "?")
        when_ms = ev.get("occurred_at", 0)
        _print(f"    #{eid}  conf={conf:.2f}  {summary}")
        roles = ev.get("roles") or []
        if roles:
            role_str = ", ".join(
                f"{r['role']}=" + (
                    r.get("entity") or f'"{r.get("value_text", "")}"'
                )
                for r in roles
            )
            _print(_dim(f"      roles: {role_str}"))


def _print_tool_calls(tool_calls: list[dict]) -> None:
    if not tool_calls:
        _print(_dim("  (无工具调用)"))
        return
    for tc in tool_calls:
        fn = tc.get("function", "?")
        args = tc.get("args", {})
        args_brief = json.dumps(args, ensure_ascii=False, separators=(",", ":"))
        if len(args_brief) > 120:
            args_brief = args_brief[:120] + "..."
        _print(f"    {_c('35', fn)}  {_dim(args_brief)}")


def _print_new_events(new_events: list[dict]) -> None:
    if not new_events:
        _print(_dim("  (本轮未提取到新记忆)"))
        return
    for ev in new_events:
        eid = ev["event_id"]
        etype = ev["event_type"]
        summary = ev["summary"]
        conf = float(ev.get("confidence", 0))
        source = ev.get("source", "")
        sup = ev.get("supersedes")
        mark = f" [supersedes #{sup}]" if sup else ""
        modality = ev.get("modality", "actual")
        mod_tag = f"[{modality}]" if modality != "actual" else ""
        _print(_ok(f'    +#{eid}  {etype}{mod_tag}  conf={conf:.2f}  \"{summary}\"{mark}'))
        roles = ev.get("roles") or []
        if roles:
            role_str = ", ".join(
                f"{r['role']}=" + (
                    r.get("entity") or f'"{r.get("value_text", "")}"'
                )
                for r in roles
            )
            _print(_dim(f"       roles: {role_str}"))


def _print_system_prompt_excerpt(system_prompt: str, max_lines: int = 8) -> None:
    if not system_prompt:
        return
    lines = system_prompt.splitlines()
    shown = lines[:max_lines]
    _print(_dim("  [system prompt 前 {} 行]".format(len(shown))))
    for ln in shown:
        _print(_dim(f"    {ln}"))
    if len(lines) > max_lines:
        _print(_dim(f"    ... ({len(lines) - max_lines} 行已省略)"))


# ── 主流程 ────────────────────────────────────────────────────────────────────

async def run_simulation(yaml_path: str, archive_only: bool, keep_db: bool) -> None:
    # ── 读取剧本 ──
    scenario_text = Path(yaml_path).read_text(encoding="utf-8")
    scenario: dict = yaml.safe_load(scenario_text)

    meta = scenario.get("meta", {})
    scenario_name = meta.get("name", Path(yaml_path).stem)
    scenario_desc = meta.get("description", "")

    # ── 创建隔离数据库 ──
    db_path = _create_sim_db_path()

    # !! 必须在所有 database/memory 相关 import 之前完成，否则无效
    import database
    _patch_db_path(db_path)
    await database.init_db()

    # ── 加载配置 & app_state ──
    from config_loader import load_config
    config, _prompt_docs = load_config()
    # 用剧本内的 memory 配置覆盖（如有）
    if "memory_config" in scenario:
        config.setdefault("memory", {}).update(scenario["memory_config"])
    # 确保 auto_archive 启用
    config.setdefault("memory", {}).setdefault("auto_archive", {})["enabled"] = True

    _print(_h1(f"模拟对话: {scenario_name}"))
    if scenario_desc:
        _print(f"  {_dim(scenario_desc)}")

    conv = scenario.get("conversation", {})
    bot_cfg = scenario.get("bot", {})
    _print(f"  会话  : {conv.get('name', '?')} ({conv.get('type', '?')}/{conv.get('id', '?')})")
    _print(f"  机器人: {bot_cfg.get('name', '?')} ({bot_cfg.get('id', '?')})")
    _print(f"  数据库: {db_path}")
    _print(f"  模式  : {'仅归档（跳过主模型）' if archive_only else '完整管线'}")

    _init_app_state(config, scenario)

    # 构建 members 字典 {id: {nickname, role}}
    members: dict[str, dict] = {}
    for m in scenario.get("members", []):
        mid = str(m.get("id", ""))
        members[mid] = {"nickname": m.get("nickname", mid), "role": m.get("role", "member")}

    # ── 创建会话 ──
    session = _create_sim_session(scenario)

    # ── 预填记忆 ──
    seed_list = scenario.get("seed_memories", [])
    if seed_list:
        _print(_h2(f"预填种子记忆 ({len(seed_list)} 条)"))
        seed_ids = await _seed_memories(seed_list)
        for i, (seed, eid) in enumerate(zip(seed_list, seed_ids)):
            _print(_ok(f"  #{eid}  {seed.get('summary', '?')}"))
    else:
        seed_ids = []

    seed_count = len(seed_ids)

    # ── 跑 Turns ──
    turns = scenario.get("turns", [])
    bot_turn_index = 0
    all_events_before_turns: list[dict] = await _get_all_events()
    events_before: int = len(all_events_before_turns)
    last_sender_id: str = ""

    for i, turn in enumerate(turns):
        turn_type = turn.get("type", "user")

        # ── 用户消息 ──────────────────────────────────────────────────────────
        if turn_type == "user":
            entry = _build_user_entry(turn, members)
            session.add_to_context(entry)
            sender_id = entry["sender_id"]
            nickname = entry["sender_name"]
            last_sender_id = sender_id
            content = entry["content"]
            _print(_h2(f"Turn {i + 1} | 用户消息"))
            _print(f"  {_c('36', nickname)} ({sender_id}): {content}")

        # ── 系统通知 ──────────────────────────────────────────────────────────
        elif turn_type == "note":
            text = str(turn.get("content", ""))
            note_entry = {
                "role": "note",
                "content": text,
                "content_type": "text",
                "timestamp": _now_iso(),
                "message_id": _next_msg_id("n"),
            }
            session.add_to_context(note_entry)
            _print(_h2(f"Turn {i + 1} | 系统通知"))
            _print(_dim(f"  {text}"))

        # ── Bot 处理 ──────────────────────────────────────────────────────────
        elif turn_type == "bot":
            bot_turn_index += 1
            _print(_h2(f"Turn {i + 1} | Bot 处理 (第 {bot_turn_index} 轮)"))

            # 1. 记忆召回
            _print(_h3("记忆召回"))
            try:
                await session.prepare_memory_recall()
                _print_recall(session.recalled_events)
            except Exception as e:
                _print(_warn(f"  [!] 记忆召回出错: {e}"))

            # 2. 主模型响应（同步，run in thread）
            if not archive_only:
                _print(_h3("主模型响应"))
                t0 = time.monotonic()
                try:
                    round_result = await asyncio.to_thread(_run_bot_round, session, False)
                except Exception as e:
                    _print(_err(f"  [✗] 主模型调用异常: {e}"))
                    round_result = {
                        "sent_texts": [], "tool_calls": [], "system_prompt": "",
                        "cognition": "", "failed": True, "skipped": False,
                    }
                elapsed = time.monotonic() - t0

                if round_result.get("skipped"):
                    _print(_warn("  [跳过] 未配置主模型适配器"))
                elif round_result.get("failed"):
                    _print(_err("  [✗] 主模型调用失败"))
                else:
                    tokens_info = (
                        f"  提示词: {round_result.get('prompt_tokens', '?')} tokens  "
                        f"输出: {round_result.get('output_tokens', '?')} tokens  "
                        f"耗时: {elapsed:.1f}s"
                    )
                    _print(_dim(tokens_info))

                # 思考链
                if round_result.get("cognition"):
                    _print(_h3("思考链 (摘录)"))
                    cog_lines = round_result["cognition"].splitlines()
                    for ln in cog_lines[:10]:
                        _print(_dim(f"  {ln}"))
                    if len(cog_lines) > 10:
                        _print(_dim(f"  ... ({len(cog_lines) - 10} 行已省略)"))

                # 工具调用
                _print(_h3("工具调用"))
                _print_tool_calls(round_result.get("tool_calls", []))

                # bot 发出的消息
                sent_texts = round_result.get("sent_texts", [])
                if sent_texts:
                    _print(_h3("Bot 发出的消息"))
                    for t in sent_texts:
                        _print(f"  {_c('32', '▶')} {t}")
                    # 把第一条（合并）加回 context 供后续归档
                    combined = " | ".join(sent_texts)
                    bot_entry = _build_bot_entry(combined)
                    session.add_to_context(bot_entry)
                else:
                    _print(_warn("  [!] Bot 本轮未发出可见消息（可能调用了 sleep/wait/shift 等）"))

            # 3. 记忆归档
            _print(_h3("记忆提取 (归档)"))
            events_cnt_before = len(await _get_all_events())
            try:
                from memory.archiver import archive_turn_memories
                await archive_turn_memories(
                    session,
                    sender_id=last_sender_id,
                    tool_calls_log=[],
                )
            except Exception as e:
                _print(_err(f"  [✗] 记忆归档出错: {type(e).__name__}: {e}"))
            new_events = await _get_events_since(events_cnt_before)
            _print_new_events(new_events)

    # ── 汇总报告 ──────────────────────────────────────────────────────────────
    _print(_h1("汇总报告"))

    all_events = await _get_all_events()
    new_this_run = [e for e in all_events if e["event_id"] > seed_count]

    _print(f"  种子记忆        : {seed_count} 条")
    _print(f"  本次提取新记忆  : {len(new_this_run)} 条")
    _print(f"  数据库记忆总数  : {len(all_events)} 条")

    if all_events:
        _print(_h2("所有记忆事件"))
        for ev in all_events:
            src_mark = _dim(f"[{ev.get('source','?')}]") if ev.get("source") else ""
            sup_mark = _warn(f" ← supersedes #{ev['supersedes']}") if ev.get("supersedes") else ""
            seed_mark = _c("33", " [seed]") if ev["event_id"] <= seed_count else _ok(" [new]")
            modality = ev.get("modality", "actual")
            mod_tag = f"[{modality}]" if modality != "actual" else ""
            _print(
                f"  #{ev['event_id']:3d} {ev['event_type']:12s}{mod_tag} conf={float(ev['confidence']):.2f} "
                f"{seed_mark} {src_mark}{sup_mark}"
            )
            _print(f"       {ev['summary']}")
            for r in ev.get("roles") or []:
                ent = r.get("entity") or ""
                val = r.get("value_text") or ""
                tgt = r.get("target_event")
                val_str = ent or (f'"{val}"' if val else "")
                if tgt:
                    val_str += f" →#{tgt}"
                _print(_dim(f"       · {r['role']:12s} {val_str}"))

    _print(f"\n{'─' * 60}")
    _print(f"  数据库文件: {db_path}")
    if not keep_db:
        try:
            os.unlink(db_path)
            _print(_dim(f"  (数据库已清理；使用 --keep-db 保留)"))
        except Exception:
            pass
    else:
        _print(_ok(f"  (数据库已保留，可用 sqlite3 或 DB Browser 查看)"))


# ── 入口 ──────────────────────────────────────────────────────────────────────

def _parse_args():
    p = argparse.ArgumentParser(
        description="AICQ 对话仿真测试工具（无需 QQ 连接）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("scenario", help="YAML 剧本文件路径")
    p.add_argument("--archive-only", action="store_true",
                   help="跳过主模型调用，仅测试记忆提取管线")
    p.add_argument("--keep-db", action="store_true",
                   help="运行结束后保留临时数据库（便于手动检查）")
    p.add_argument("--output", metavar="FILE",
                   help="将报告同时保存为纯文本文件")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    yaml_path = args.scenario

    if not Path(yaml_path).exists():
        print(f"错误: 找不到剧本文件 {yaml_path}", file=sys.stderr)
        sys.exit(1)

    try:
        asyncio.run(run_simulation(yaml_path, args.archive_only, args.keep_db))
    except KeyboardInterrupt:
        print("\n[中断]")
    except Exception as e:
        import traceback
        print(_err(f"\n[致命错误] {e}"), file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)

    if args.output:
        _flush_output_to_file(args.output)


if __name__ == "__main__":
    main()
