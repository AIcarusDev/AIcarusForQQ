"""jieba tokenizer helpers for memory indexing."""

import jieba

_MIN_TOKEN_LEN: int = 2
_CUSTOM_WORD_FREQ: int = 100


def configure(min_token_len: int = 2, custom_word_freq: int = 100) -> None:
    global _MIN_TOKEN_LEN, _CUSTOM_WORD_FREQ
    _MIN_TOKEN_LEN = min_token_len
    _CUSTOM_WORD_FREQ = custom_word_freq


def load_custom_dict_from_events(events: list[dict]) -> None:
    """从事件 summary 字段批量种子 jieba 自定义词典。"""
    for row in events:
        summary = row.get("summary") or ""
        if summary and len(summary) >= _MIN_TOKEN_LEN:
            if summary.startswith("[") and summary.endswith("]"):
                continue
            jieba.add_word(summary, freq=_CUSTOM_WORD_FREQ)


def register_word(text: str) -> None:
    if text and len(text) >= _MIN_TOKEN_LEN:
        jieba.add_word(text, freq=_CUSTOM_WORD_FREQ)


def tokenize(text: str) -> str:
    if not text:
        return ""
    tokens = [
        token for token in jieba.cut(text)
        if token.strip() and len(token) >= _MIN_TOKEN_LEN
    ]
    return " ".join(tokens) if tokens else text


def build_fts_query(message: str) -> str:
    tokens = [
        token for token in jieba.cut(message)
        if token.strip() and len(token) >= _MIN_TOKEN_LEN
    ]
    if not tokens:
        return ""
    terms: list[str] = []
    for token in tokens:
        terms.append(f'"{token}"')
        terms.append(f'"{token}*"')
    return " OR ".join(terms)
