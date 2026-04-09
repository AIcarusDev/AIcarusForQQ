"""json_repair.py — 模型输出 JSON 清洗与修复

对模型返回的文本进行以下清洗，尽可能得到可解析的 JSON 对象：
  1. 剥离 Markdown 代码块（```json ... ``` 或 ``` ... ```）
  2. 去除首尾空白
  3. 截取第一个 { 到最后一个 } 之间的内容（处理前后多余文字）
  4. 跳过前缀乱码（如 `{"m{ "mood":...}` — 从下一个合法 { 重新截取）
  5. 删除前置裸开括号行（如 `{\n{...}` 模式）
  6. 删除游离引号垃圾行
  7. 修复多余/缺失的尾部大括号 / 方括号
  8. 转义字符串值内的裸双引号（如 "他说"你好"真烦" → "他说\"你好\"真烦"）

若清洗后能被正常解析，返回 (result_dict, True) 表示经过了修复；
若清洗后仍无法解析，抛出 json.JSONDecodeError。
若原始文本直接可解析，返回 (result_dict, False) 表示无需修复。
"""

import json
import logging
import re

logger = logging.getLogger("AICQ.json_repair")


def _strip_markdown(text: str) -> str:
    """去掉 Markdown 代码块标记。"""
    # ```json\n...\n``` 或 ```\n...\n```
    m = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if m:
        return m.group(1).strip()
    return text.strip()


def _extract_object(text: str) -> str:
    """截取第一个 { 到最后一个 } 之间的内容。"""
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end >= start:
        return text[start : end + 1]
    return text


def _probe_inner_object(text: str) -> str | None:
    """逐一尝试文本中每个 '{' 作为 JSON 起点，返回第一个可解析的子串。

    处理模型在真实 JSON 对象前输出了少量乱码前缀的情况，例如：
      ``{"m{ "mood": ...}``  →  跳过 ``{"m`` 乱码，找到真正的 ``{ "mood": ...}``
    """
    end = text.rfind("}")
    if end == -1:
        return None
    pos = 1  # 第一个 { 已由 _extract_object 处理过，从第二个开始
    while True:
        start = text.find("{", pos)
        if start == -1 or start > end:
            break
        candidate = text[start : end + 1]
        try:
            json.loads(candidate)
            return candidate
        except json.JSONDecodeError:
            pass
        pos = start + 1
    return None


def _strip_leading_lone_braces(text: str) -> str:
    """删除开头多余的裸开括号行。

    如 ``{\n{"key": "val"}`` 这种模型在真实 JSON 前多吐了一个 ``{``，
    会导致后续解析出现嵌套对象但无键名的情况。
    规则：若某行（去除空白后）仅为 ``{`` 或 ``[``，且其后仍有完整的
    JSON 对象内容（下一个非空行以 ``{`` 或 ``[`` 开头），则删除该行。
    """
    lines = text.splitlines()
    result = []
    i = 0
    while i < len(lines):
        stripped = lines[i].strip()
        # 裸开括号行：仅含 `{` 或 `[`
        if stripped in ("{", "["):
            # 看后续是否还有一个新的完整对象/数组开头
            rest = "\n".join(lines[i + 1:]).lstrip()
            if rest.startswith(("{", "[")):
                # 跳过这行（多余的裸开括号）
                i += 1
                continue
        result.append(lines[i])
        i += 1
    return "\n".join(result)


def _remove_stray_lines(text: str) -> str:
    """删除 JSON 行内的游离垃圾行（以引号开头的残片）。

    模型有时会在 JSON body 里插入类似 ``"{"`` ``"{`` ````` 等残片行，
    这些行无法构成合法的键值对，直接删除。

    匹配规则：
    - 行必须以引号 ``"`` 开头（否则是合法的结构行如 ``{`` ``}``，不能删）
    - 去除首尾空白后，内容仅由引号、花括号、方括号、反引号、空格、逗号组成
    - 不包含冒号（有冒号说明是合法的 key: value 行）
    """
    cleaned = []
    for line in text.splitlines():
        stripped = line.strip()
        # 空行保留
        if not stripped:
            cleaned.append(line)
            continue
        # 含冒号 → 合法键值行，保留
        if ":" in stripped:
            cleaned.append(line)
            continue
        # 不以引号开头 → 合法结构行（`{` `}` `[` `]` 等），保留
        if not stripped.startswith('"'):
            cleaned.append(line)
            continue
        # 以引号开头，且内容仅由结构性符号/引号/反引号组成 → 游离垃圾行，删除
        if re.fullmatch(r'["\'`{}\[\],\s]*', stripped):
            continue
        cleaned.append(line)
    return "\n".join(cleaned)


def _fix_unescaped_string_quotes(text: str) -> str:
    """转义 JSON 字符串值内部的裸双引号。

    模型有时输出 ``"mood": "他说"你好"真烦"`` 这样内层引号未转义的文本。
    扫描时追踪字符串上下文，对字符串内部遇到的 ``"`` 进行前瞻判断：

    - 向后跳过空白后，下一字符是 ``,`` ``}`` ``]`` ``\\n`` ``\\r`` ``:`` → 合法终止，不处理
    - 其余情况 → 内层裸引号，补 ``\\``
    """
    result = []
    i = 0
    n = len(text)
    in_string = False

    while i < n:
        c = text[i]
        if not in_string:
            result.append(c)
            if c == '"':
                in_string = True
        else:
            if c == '\\':
                # 已转义字符，原样保留（含下一个字符）
                result.append(c)
                i += 1
                if i < n:
                    result.append(text[i])
            elif c == '"':
                # 前瞻：跳过行内空白，看下一个有效字符
                j = i + 1
                while j < n and text[j] in ' \t':
                    j += 1
                if j >= n or text[j] in (',', '}', ']', '\n', '\r', ':'):
                    # 合法终止引号
                    result.append(c)
                    in_string = False
                else:
                    # 内层裸引号，补转义
                    result.append('\\')
                    result.append('"')
            else:
                result.append(c)
        i += 1

    return ''.join(result)


def _fix_braces(text: str) -> str:
    """平衡大括号和方括号。

    - 多余的尾部闭合符号会被裁剪
    - 缺失的尾部闭合符号会被追加
    """
    stack = []
    close_map = {"}": "{", "]": "["}
    open_set = {"{", "["}
    close_set = {"}", "]"}
    pair_map = {"{": "}", "[": "]"}

    result = []
    for ch in text:
        if ch in open_set:
            stack.append(ch)
            result.append(ch)
        elif ch in close_set:
            if stack and stack[-1] == close_map[ch]:
                # 正常闭合
                stack.pop()
                result.append(ch)
            elif not stack:
                # 多余的闭合符号，跳过
                pass
            else:
                # 闭合符号类型不匹配，追加缺失的再处理当前
                result.append(pair_map[stack.pop()])
                # 重新处理当前字符（递归一步）
                if stack and stack[-1] == close_map[ch]:
                    stack.pop()
                    result.append(ch)
        else:
            result.append(ch)

    # 追加缺失的闭合符号（逆序）
    while stack:
        result.append(pair_map[stack.pop()])

    return "".join(result)


def clean_and_parse(text: str, source: str = "") -> tuple[dict, bool]:
    """清洗并解析模型输出文本。

    Parameters
    ----------
    text:   模型返回的原始字符串
    source: 调用方标识，用于日志（如 "[Gemini]" "[OpenAICompat]"）

    Returns
    -------
    (result_dict, repaired)
      repaired=False 表示原文即可直接解析，无需修复
      repaired=True  表示经过了至少一步清洗/修复

    Raises
    ------
    json.JSONDecodeError  若所有修复手段均告失败
    """
    # ── 第一步：尝试直接解析 ──────────────────────────────
    try:
        return json.loads(text), False
    except json.JSONDecodeError:
        pass

    prefix = f"{source} " if source else ""
    repaired_text = text

    # ── 第二步：剥离 Markdown 代码块 ─────────────────────
    stripped = _strip_markdown(text)
    try:
        result = json.loads(stripped)
        logger.warning(
            "%sJSON 含 Markdown 包裹，已自动剥离\n"
            "===== 原始文本（前200字）=====\n%s",
            prefix, text[:200],
        )
        return result, True
    except json.JSONDecodeError:
        repaired_text = stripped

    # ── 第三步：截取首尾大括号范围 ───────────────────────
    extracted = _extract_object(repaired_text)
    try:
        result = json.loads(extracted)
        logger.warning(
            "%sJSON 首尾存在多余文字，已截取对象范围\n"
            "===== 原始文本（前200字）=====\n%s",
            prefix, text[:200],
        )
        return result, True
    except json.JSONDecodeError:
        repaired_text = extracted

    # ── 第四步：跳过前缀乱码（如 `{"m{ "mood":...}` 模式）────
    probed = _probe_inner_object(repaired_text)
    if probed is not None:
        logger.warning(
            "%sJSON 前置乱码前缀，已跳过前缀从内层 { 截取\n"
            "===== 原始文本（前200字）=====\n%s",
            prefix, text[:200],
        )
        return json.loads(probed), True

    # ── 第五步：删除前置裸开括号行（如 `{\n{...}` 模式）────
    no_lead = _strip_leading_lone_braces(repaired_text)
    no_lead = _extract_object(no_lead)
    try:
        result = json.loads(no_lead)
        logger.warning(
            "%sJSON 前置多余裸开括号行，已删除\n"
            "===== 原始文本（前200字）=====\n%s",
            prefix, text[:200],
        )
        return result, True
    except json.JSONDecodeError:
        repaired_text = no_lead

    # ── 第六步：删除游离引号垃圾行 ──────────────────────
    destrayed = _remove_stray_lines(repaired_text)
    destrayed = _extract_object(destrayed)
    try:
        result = json.loads(destrayed)
        logger.warning(
            "%sJSON 含游离垃圾行（如模型输出 \"{\" 等残片），已清除\n"
            "===== 原始文本（前200字）=====\n%s",
            prefix, text[:200],
        )
        return result, True
    except json.JSONDecodeError:
        repaired_text = destrayed

    # ── 第七步：修复大括号不平衡 ─────────────────────────
    fixed = _fix_braces(repaired_text)
    try:
        result = json.loads(fixed)
        logger.warning(
            "%sJSON 括号不平衡，已自动修复\n"
            "===== 修复前（前200字）=====\n%s\n"
            "===== 修复后（前200字）=====\n%s",
            prefix, text[:200], fixed[:200],
        )
        return result, True
    except json.JSONDecodeError:
        repaired_text = fixed

    # ── 第八步：转义字符串值内的裸双引号 ─────────────────
    # 处理模型将引号直接嵌入字符串值，如 "他说"你好"真烦"
    unquoted = _fix_unescaped_string_quotes(repaired_text)
    try:
        result = json.loads(unquoted)
        logger.warning(
            "%sJSON 字符串内含未转义双引号，已自动修复\n"
            "===== 原始文本（前200字）=====\n%s",
            prefix, text[:200],
        )
        return result, True
    except json.JSONDecodeError:
        pass

    # ── 全部失败：记录错误并抛出 ─────────────────────────
    logger.error(
        "%sJSON 修复失败，所有清洗手段均无效\n"
        "===== 原始返回文本 =====\n%s",
        prefix, text,
    )
    # 抛出更有意义的错误（基于最终修复尝试）
    json.loads(unquoted)  # 一定会抛出
    raise AssertionError("unreachable")  # 仅为类型检查
