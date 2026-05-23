"""decision_filter.py — 模型输出 decision 结构规范化

对 json_repair 解析出的 decision dict 做进一步的结构处理：
  1. 拆分同一消息中相邻的 text segment，使每条消息只含一个 text
     （normalize_send_messages）
  2. 移除由于误解 schema 错误生出的 "additionalProperties" 键
     （remove_additional_properties_key）

后续可在此添加更多过滤/规范化逻辑。
"""

_LOOP_CONTROL_WAIT_MAX_TIMEOUT = 300


def clamp_wait_timeout(data: dict) -> tuple[dict, bool]:
    """将 loop_control.wait.timeout 超出 schema 最大值的情况直接钳制，而非触发 ValidationError。

    Returns
    -------
    (data, repaired)
      repaired=True 表示发生了钳制
    """
    lc = data.get("loop_control")
    if not isinstance(lc, dict):
        return data, False
    wait = lc.get("wait")
    if not isinstance(wait, dict):
        return data, False
    timeout = wait.get("timeout")
    if isinstance(timeout, (int, float)) and timeout > _LOOP_CONTROL_WAIT_MAX_TIMEOUT:
        wait["timeout"] = _LOOP_CONTROL_WAIT_MAX_TIMEOUT
        return data, True
    return data, False


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
