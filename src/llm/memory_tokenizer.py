"""memory_tokenizer.py — jieba 分词封装

负责：
  1. 对记忆文本进行分词（供 FTS5 索引使用）
  2. 管理 jieba 自定义词典（与记忆数据同源，重启时恢复）
  3. 构造 FTS5 查询串（精确项 + 前缀项）

设计原则：
  - jieba 负责语言学分词，理解中文词边界
  - FTS5 unicode61 接收空格分隔的 token 流，按空格切分建立倒排索引
  - 两层职责分离，MemoryTriples 同时存储原始文本 (object_text) 和分词版本 (object_text_tok)
"""

import jieba

# 常用停用词（高频虚词/语气词，过滤后不影响语义检索）
STOPWORDS: frozenset[str] = frozenset({
    "的", "了", "在", "是", "我", "你", "他", "她", "它",
    "和", "也", "都", "就", "吗", "呢", "啊", "哦", "嗯",
    "这", "那", "有", "不", "到", "说", "要", "会", "来",
    "与", "或", "但", "而", "于", "为", "以", "及", "其",
    "被", "从", "向", "对", "把", "让", "使", "等", "们",
    "什么", "怎么", "哪个", "一个", "这个", "那个",
})

_MIN_TOKEN_LEN: int = 2
_CUSTOM_WORD_FREQ: int = 100


def load_custom_dict_from_triples(triples: list[dict]) -> None:
    """启动时从已有记忆中恢复自定义词典。

    在 lifecycle.py 中的记忆恢复流程里调用，确保重启后分词精度与运行时一致。
    """
    for row in triples:
        for field_val in (row.get("object_text"), row.get("predicate")):
            if field_val and len(field_val) >= _MIN_TOKEN_LEN:
                # predicate 为 "[note]" 之类的结构标记，直接跳过
                if field_val.startswith("[") and field_val.endswith("]"):
                    continue
                jieba.add_word(field_val, freq=_CUSTOM_WORD_FREQ)


def register_word(text: str) -> None:
    """写入新记忆时实时注册词汇，形成"记忆越丰富、召回越精准"的正反馈。

    在 write_triple() 完成后调用。
    """
    if text and len(text) >= _MIN_TOKEN_LEN:
        jieba.add_word(text, freq=_CUSTOM_WORD_FREQ)


def tokenize(text: str) -> str:
    """对文本分词，返回空格分隔的 token 串，供写入 MemoryTriples.object_text_tok 使用。

    若分词后无有效 token（全为停用词或过短），回退返回原文（保留检索可能性）。
    """
    if not text:
        return ""
    tokens = [
        t for t in jieba.cut(text)
        if t.strip() and t not in STOPWORDS and len(t) >= _MIN_TOKEN_LEN
    ]
    return " ".join(tokens) if tokens else text


def build_fts_query(message: str) -> str:
    """从消息文本构造 FTS5 查询串。

    每个 token 同时生成精确项和前缀项，覆盖缩写召回（如「星际」命中「星际争霸2」）。
    返回空字符串表示消息无有效关键词，调用方应跳过 FTS5 查询。
    """
    tokens = [
        t for t in jieba.cut(message)
        if len(t) >= _MIN_TOKEN_LEN and t not in STOPWORDS
    ]
    if not tokens:
        return ""
    terms: list[str] = []
    for token in tokens:
        terms.append(f'"{token}"')    # 精确匹配
        terms.append(f'"{token}*"')   # 前缀匹配（覆盖缩写）
    return " OR ".join(terms)
