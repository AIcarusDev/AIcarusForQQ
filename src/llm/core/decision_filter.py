"""decision_filter.py — 模型输出 decision 结构规范化

对 json_repair 解析出的 decision dict 做进一步的结构处理：
  1. 拆分同一消息中相邻的 text segment，使每条消息只含一个 text
     （normalize_send_messages）

后续可在此添加更多过滤/规范化逻辑。
"""


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
