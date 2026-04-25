"""High-level memory write and delete operations."""

from . import runtime
from .repo.triples import soft_delete_triple, write_triple
from .tokenizer import register_word, tokenize


async def add_memory(
    content: str,
    source: str,
    reason: str,
    conv_type: str = "",
    conv_id: str = "",
    conv_name: str = "",
    subject: str = "UnknownUser",
    predicate: str = "[note]",
    origin: str = "passive",
    recall_scope: str = "global",
    confidence: float = 0.6,
) -> int:
    object_text_tok = tokenize(content)
    register_word(content)

    created_at = runtime._ms()
    entry: dict = {
        "subject": subject,
        "predicate": predicate,
        "object_text": content,
        "origin": origin,
        "confidence": confidence,
        "context": "truth",
        "created_at": created_at,
        "last_accessed": created_at,
        "source": source,
        "reason": reason,
        "conv_type": conv_type,
        "conv_id": conv_id,
        "conv_name": conv_name,
        "recall_scope": recall_scope,
    }

    async with runtime._get_lock():
        target_pool, cap = runtime.choose_target_pool(subject, origin)
        if cap > 0:
            while len(target_pool) >= cap:
                target_pool.pop(0)

        triple_id = await write_triple(
            subject=subject,
            predicate=predicate,
            object_text=content,
            object_text_tok=object_text_tok,
            source=source,
            reason=reason,
            conv_type=conv_type,
            conv_id=conv_id,
            conv_name=conv_name,
            origin=origin,
            recall_scope=recall_scope,
            confidence=confidence,
        )
        entry["id"] = triple_id
        target_pool.append(entry)
    return triple_id


async def remove_memory(memory_id_str: str) -> bool:
    try:
        triple_id = int(memory_id_str)
    except (TypeError, ValueError):
        return False

    runtime.remove_cached_memory(triple_id)
    return await soft_delete_triple(triple_id)