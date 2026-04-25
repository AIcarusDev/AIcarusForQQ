"""jieba tokenizer helpers for memory indexing."""

import jieba

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


def configure(min_token_len: int = 2, custom_word_freq: int = 100) -> None:
    global _MIN_TOKEN_LEN, _CUSTOM_WORD_FREQ
    _MIN_TOKEN_LEN = min_token_len
    _CUSTOM_WORD_FREQ = custom_word_freq


def load_custom_dict_from_triples(triples: list[dict]) -> None:
    for row in triples:
        for field_val in (row.get("object_text"), row.get("predicate")):
            if field_val and len(field_val) >= _MIN_TOKEN_LEN:
                if field_val.startswith("[") and field_val.endswith("]"):
                    continue
                jieba.add_word(field_val, freq=_CUSTOM_WORD_FREQ)


def register_word(text: str) -> None:
    if text and len(text) >= _MIN_TOKEN_LEN:
        jieba.add_word(text, freq=_CUSTOM_WORD_FREQ)


def tokenize(text: str) -> str:
    if not text:
        return ""
    tokens = [
        token for token in jieba.cut(text)
        if token.strip() and token not in STOPWORDS and len(token) >= _MIN_TOKEN_LEN
    ]
    return " ".join(tokens) if tokens else text


def build_fts_query(message: str) -> str:
    tokens = [
        token for token in jieba.cut(message)
        if len(token) >= _MIN_TOKEN_LEN and token not in STOPWORDS
    ]
    if not tokens:
        return ""
    terms: list[str] = []
    for token in tokens:
        terms.append(f'"{token}"')
        terms.append(f'"{token}*"')
    return " OR ".join(terms)