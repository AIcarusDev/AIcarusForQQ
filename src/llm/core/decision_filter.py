"""decision_filter.py — 模型输出 decision 结构规范化

对 json_repair 解析出的 decision dict 做进一步的结构处理：
  1. 拆分同一消息中相邻的 text segment，使每条消息只含一个 text
     （normalize_send_messages）
  2. 将 decision 子 action 对象里错误放置的 motivation 提升到 decision 层
     （hoist_decision_motivation）
  3. 对所有 schema required 的 motivation 字段，若整体缺失则填入占位文本
     （fill_missing_motivations）
  4. 移除由于误解 schema 错误生出的 "additionalProperties" 键
     （remove_additional_properties_key）

后续可在此添加更多过滤/规范化逻辑。
"""

# 窥屏模式 decision 中允许的 action 键
_WATCHER_ACTION_KEYS = {"engage", "wait", "shift", "pass", "hibernate"}

# loop_control 中的 action 键
_LOOP_CONTROL_ACTION_KEYS = {"continue", "idle", "wait", "shift"}


def hoist_decision_motivation(data: dict) -> tuple[dict, bool]:
    """将误放在 action 子对象里的 motivation 字段提升到 decision 层。

    LLM 有时会把 motivation 塞进 wait / shift 等子对象，而 schema 要求它
    直接位于 decision 层。本函数在 schema 校验前调用，避免触发 ValidationError。

    Returns
    -------
    (data, repaired)
      repaired=True 表示发生了结构修复
    """
    decision = data.get("decision")
    if not isinstance(decision, dict):
        return data, False
    if "motivation" in decision:
        return data, False  # 已在正确位置，无需处理

    for key, val in decision.items():
        if key in _WATCHER_ACTION_KEYS and isinstance(val, dict) and "motivation" in val:
            decision["motivation"] = val.pop("motivation")
            return data, True

    return data, False


def hoist_loop_control_motivation(data: dict) -> tuple[dict, bool]:
    """将误放在 loop_control 的 action 子对象里的 motivation 提升到 loop_control 层。

    LLM 有时会将 motivation 塞进 continue / idle / wait / shift 等子对象，
    而 schema 要求它直接位于 loop_control 层。
    例如 ``{"continue": {"motivation": "..."}``  →  ``{"motivation": "...", "continue": {}``

    Returns
    -------
    (data, repaired)
      repaired=True 表示发生了结构修复
    """
    loop_control = data.get("loop_control")
    if not isinstance(loop_control, dict):
        return data, False
    if "motivation" in loop_control:
        return data, False  # 已在正确位置，无需处理

    for key, val in loop_control.items():
        if key in _LOOP_CONTROL_ACTION_KEYS and isinstance(val, dict) and "motivation" in val:
            loop_control["motivation"] = val.pop("motivation")
            return data, True

    return data, False


# 所有 schema 中 required 的 motivation 字段位置（对象路径）
_REQUIRED_MOTIVATION_PATHS: list[list[str]] = [
    ["decision"],       # decision.motivation
    ["loop_control"],   # loop_control.motivation
]

_MOTIVATION_PLACEHOLDER = "忘了，好像走神了"


def fill_missing_motivations(data: dict) -> tuple[dict, bool]:
    """对 schema 中所有 required 的 motivation 字段，若缺失则自动填入占位文本。

    适用于模型输出时整体遗漏 motivation 导致 ValidationError 的情况。
    目前覆盖：decision.motivation、loop_control.motivation。

    Returns
    -------
    (data, repaired)
      repaired=True 表示至少有一个字段被填入了占位文本
    """
    repaired = False
    for path in _REQUIRED_MOTIVATION_PATHS:
        node = data
        for key in path:
            if not isinstance(node, dict):
                node = None
                break
            node = node.get(key)
        if isinstance(node, dict) and "motivation" not in node:
            node["motivation"] = _MOTIVATION_PLACEHOLDER
            repaired = True
    return data, repaired


def remove_additional_properties_key(data: dict) -> tuple[dict, bool]:
    """递归遍历整个字典，移除模型错误生出的 'additionalProperties' 键。

    有的模型会误把 schema 里的 additionalProperties: false 
    作为数据字段输出（例如在 idle, continue, wait 等对象中），导致 ValidationError。
    
    Returns
    -------
    (data, repaired)
      repaired=True 表示至少移除了一个这样的键
    """
    repaired = False
    
    def _traverse(node):
        nonlocal repaired
        if isinstance(node, dict):
            # 如果存在这个非法键，就直接删掉
            if "additionalProperties" in node:
                del node["additionalProperties"]
                repaired = True
            for val in node.values():
                _traverse(val)
        elif isinstance(node, list):
            for item in node:
                _traverse(item)

    _traverse(data)
    return data, repaired



def normalize_send_messages(send_messages: list[dict]) -> list[dict]:
    """将同一条消息中相邻的 text segment 拆分为各自独立的消息。

    例如 [{segments: [at, text1, text2]}]
    → [{segments: [at, text1]}, {segments: [text2]}]

    非 text 的 segment（at、sticker 等）归入其所在 buffer，
    quote 仅保留在拆分后的第一条消息上。
    """
    result: list[dict] = []
    for msg in send_messages:
        quote = msg.get("quote")
        segments = msg.get("segments", [])

        sub_segs_list: list[list[dict]] = []
        current: list[dict] = []
        has_text = False

        for seg in segments:
            if seg.get("command") == "text":
                if has_text:
                    sub_segs_list.append(current)
                    current = []
                    has_text = False
                current.append(seg)
                has_text = True
            else:
                current.append(seg)

        if current:
            sub_segs_list.append(current)

        # 没有发生实际拆分，保留原消息不变
        if len(sub_segs_list) <= 1:
            result.append(msg)
            continue

        for i, segs in enumerate(sub_segs_list):
            new_msg: dict = {"segments": segs}
            if i == 0 and quote:
                new_msg["quote"] = quote
            result.append(new_msg)

    return result
