"""Memory recall pipeline."""

from .repo.triples import search_triples
from .tokenizer import build_fts_query


async def recall_memories(
    message_text: str,
    sender_id: str = "",
    config: dict | None = None,
    context_scope: str = "",
) -> list[dict]:
    cfg = config or {}
    ranking = cfg.get("ranking", {})
    alpha = float(ranking.get("alpha", 0.5))
    beta = float(ranking.get("beta", 0.3))
    gamma = float(ranking.get("gamma", 0.2))
    recall_top_k = int(ranking.get("recall_top_k", 20))
    inject_top_n = int(ranking.get("inject_top_n", 8))

    subject_filter = f"User:qq_{sender_id}" if sender_id else ""
    fts_query = build_fts_query(message_text)

    user_results = await search_triples(
        fts_query=fts_query,
        subject_filter=subject_filter,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
        recall_top_k=recall_top_k,
        context_scope=context_scope,
    )
    top_user = user_results[:inject_top_n]

    self_top_k = max(3, recall_top_k // 3)
    self_results = await search_triples(
        fts_query=fts_query,
        subject_filter="Bot:self",
        alpha=alpha,
        beta=beta,
        gamma=gamma,
        recall_top_k=self_top_k,
        context_scope=context_scope,
    )
    seen_ids = {row["id"] for row in top_user if "id" in row}
    top_self = [row for row in self_results if row.get("id") not in seen_ids]
    return top_user + top_self