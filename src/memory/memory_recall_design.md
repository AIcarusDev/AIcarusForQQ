# Memory Recall Implementation Spec

## Non-Negotiable Constraints

- Use `prompt.py` directly as the only archive prompt source.
- Do not modify `prompt.py` prompt text in any way.
- Do not preserve backward compatibility with the old memory database schema.
- Assume the memory database can be deleted and rebuilt from scratch.
- Recall output shown to the model contains only event `summary` plus event time.
- All other event fields are internal signals for parsing, storage, recall, ranking, filtering, debugging, and backfill.
- `confidence` is stored, but it is not a primary ranking signal in Memory.

## Prompt And Archive Contract

- [ ] Replace the old archive prompt entry with a direct import of `prompt.ARCHIVE_SYSTEM_PROMPT`.
- [ ] Remove the forced-tool archive contract for memory archive calls.
- [ ] Call the archive model as normal text generation and parse the prompt-native output format.
- [ ] Keep old archive code only if isolated behind an explicitly disabled legacy path.

Expected output shape:

```xml
<analysis>
...
</analysis>
<extract>
<event>{...}</event>
<event>{...}</event>
</extract>
```

## Archive Parser Contract

Parser strictness is intentionally high.

- [ ] Require exactly one top-level `<extract>...</extract>` block.
- [ ] Ignore every `<event>` block outside `<extract>`.
- [ ] Allow an empty `<extract></extract>` and treat it as zero events.
- [ ] Forbid markdown fences inside `<event>`.
- [ ] Forbid free text before or after the JSON object inside `<event>`.
- [ ] Parse each `<event>` payload as one JSON object.
- [ ] Require each event JSON object to contain `summary`, `event_type`, and `roles`.
- [ ] Require `summary` and `event_type` to be non-empty strings after trim.
- [ ] Require `roles` to be a list.
- [ ] Partially accept a batch: valid events are stored, invalid events are rejected with structured logs.
- [ ] Reject a whole batch when `<extract>` is missing, duplicated, or structurally unparseable.
- [ ] Preserve the raw event JSON string for every accepted event.

Allowed optional-field defaults:

| Field | Default | Notes |
| --- | --- | --- |
| `is_negated` | `false` | Must remain separate from `event_type`. |
| `status` | `"actual"` | Unknown values are stored as raw JSON but normalized to `"actual"` for traversal until explicitly supported. |
| `confidence` | `0.5` | Stored only; not a primary ranking signal. |
| `source` | archive job source | System metadata, not model-derived if absent. |
| `reason` | empty string | Internal metadata. |
| `occurred_at` | cognition timestamp if available, else archive time | Must be absolute milliseconds internally. |

Unknown model-emitted fields:

- [ ] Keep them in `raw_event_json`.
- [ ] Do not invent table columns automatically.
- [ ] Do not feed them to the model during recall rendering.

## Predicate Normalization Boundary

Open predicates must remain open.

- [ ] `event_type` stores the model output after trim.
- [ ] `event_type_norm` is allowed only for light canonicalization:
  - trim surrounding whitespace
  - lowercase ASCII letters
  - collapse internal whitespace and separators to `_`
  - optionally strip trivial punctuation around the token
- [ ] `event_type_norm` must not map to a closed enum.
- [ ] `event_type_norm` must not use the old `_EVENT_TYPE_NORMALIZE` table.
- [ ] Lemmatization is allowed only if model-independent and does not map multiple semantic predicates into a fixed vocabulary.

Examples:

| Raw `event_type` | Allowed `event_type_norm` |
| --- | --- |
| `" Like "` | `"like"` |
| `"located at"` | `"located_at"` |
| `"depend-on"` | `"depend_on"` |

Forbidden:

| Raw `event_type` | Forbidden normalization |
| --- | --- |
| `"prefer"` | `"like"` |
| `"located_at"` | `"be"` |
| `"deny"` with `is_negated=true` | `"refuse"` |

## New Storage Model

Rebuild memory storage around prompt-native events.

### `MemoryEvents`

- [ ] `event_id INTEGER PRIMARY KEY`
- [ ] `summary TEXT NOT NULL`
- [ ] `summary_tok TEXT NOT NULL DEFAULT ''`
- [ ] `event_type TEXT NOT NULL`
- [ ] `event_type_norm TEXT NOT NULL`
- [ ] `is_negated INTEGER NOT NULL DEFAULT 0`
- [ ] `status TEXT NOT NULL DEFAULT 'actual'`
- [ ] `confidence REAL NOT NULL DEFAULT 0.5`
- [ ] `occurred_at INTEGER NOT NULL`
- [ ] `created_at INTEGER NOT NULL`
- [ ] `last_seen_at INTEGER NOT NULL DEFAULT 0`
- [ ] `last_accessed INTEGER NOT NULL DEFAULT 0`
- [ ] `occurrences INTEGER NOT NULL DEFAULT 1`
- [ ] `source TEXT NOT NULL DEFAULT ''`
- [ ] `reason TEXT NOT NULL DEFAULT ''`
- [ ] `conv_type TEXT NOT NULL DEFAULT ''`
- [ ] `conv_id TEXT NOT NULL DEFAULT ''`
- [ ] `conv_name TEXT NOT NULL DEFAULT ''`
- [ ] `raw_event_json TEXT NOT NULL`
- [ ] `is_deleted INTEGER NOT NULL DEFAULT 0`

Rules:

- [ ] Do not store canonical embedding status on the event row.
- [ ] Do not store cognition-source ids as an event-row scalar only; the normalized event-to-cognition links belong to `MemoryEventSources`.
- [ ] Per-vector status belongs to `MemoryEmbeddingJobs` and `MemoryVectors`.
- [ ] Event-level embedding readiness may be computed by query or exposed as a view, but must not become a second source of truth.

### `CognitionSources`

Core runtime table, not Memory. Store each persisted cognition-source block with a durable identity.
`source_uid` must not be derived from the prompt-local `prompt_source_id`; the local id may change when the same cognition block appears in a different extraction window.

- [ ] `source_uid TEXT PRIMARY KEY`
- [ ] `source_kind TEXT NOT NULL DEFAULT 'cognition'`
- [ ] `origin_type TEXT NOT NULL DEFAULT ''`
- [ ] `origin_id TEXT NOT NULL DEFAULT ''`
- [ ] `prompt_source_id TEXT NOT NULL DEFAULT ''`
- [ ] `source_seq INTEGER`
- [ ] `source_timestamp TEXT NOT NULL DEFAULT ''`
- [ ] `cognition_text TEXT NOT NULL DEFAULT ''`
- [ ] `cognition_hash TEXT NOT NULL DEFAULT ''`
- [ ] `created_at INTEGER NOT NULL`
- [ ] `last_seen_at INTEGER NOT NULL`
- [ ] `metadata_json TEXT NOT NULL DEFAULT '{}'`

### `MemoryEventSources`

Store the source cognition blocks that produced each extracted event.

- [ ] `event_source_id INTEGER PRIMARY KEY`
- [ ] `event_id INTEGER NOT NULL REFERENCES MemoryEvents(event_id) ON DELETE CASCADE`
- [ ] `source_kind TEXT NOT NULL DEFAULT 'cognition'`
- [ ] `source_uid TEXT NOT NULL`
- [ ] `source_id TEXT NOT NULL`
- [ ] `prompt_source_id TEXT NOT NULL DEFAULT ''`
- [ ] `source_seq INTEGER`
- [ ] `source_timestamp TEXT NOT NULL DEFAULT ''`
- [ ] `created_at INTEGER NOT NULL`

Rules:

- [ ] Prompt output must include string `source_id` on every event.
- [ ] `source_id` should copy ids from input `<cognition id="...">` blocks, using commas for multiple ids.
- [ ] The backend extracts contiguous digits from `source_id` and only stores ids that exist in the current task input.
- [ ] If no valid source id remains after filtering, the event is still written; it simply gets no source-link rows.
- [ ] Prompt-local ids are not durable identities; event-source links must store the durable `source_uid`.
- [ ] Exact event dedupe must still add any newly observed `source_id` links to the existing event.
- [ ] The current `source_kind='cognition'` links are intended to become the bridge from memory events to future world-slice anchors.

### `MemoryParticipants`

- [ ] `participant_id INTEGER PRIMARY KEY`
- [ ] `event_id INTEGER NOT NULL REFERENCES MemoryEvents(event_id) ON DELETE CASCADE`
- [ ] `role TEXT NOT NULL`
- [ ] `entity TEXT`
- [ ] `value_text TEXT`
- [ ] `value_tok TEXT NOT NULL DEFAULT ''`
- [ ] `raw_participant_json TEXT NOT NULL`

Rules:

- [ ] Every participant row must have at least one of `entity` or `value_text`.
- [ ] Entity identifiers are late-bound. Do not merge similar entities at write time.
- [ ] `value_text` may be indexed and embedded, but it is not shown in final recall output.
- [ ] Soft-deleted events keep participant rows; every normal query must filter `MemoryEvents.is_deleted=0`.
- [ ] Hard-deleted events cascade-delete participants.

### `MemoryPredicates`

- [ ] `predicate_id INTEGER PRIMARY KEY`
- [ ] `event_type_norm TEXT NOT NULL UNIQUE`
- [ ] `display_event_type TEXT NOT NULL`
- [ ] `created_at INTEGER NOT NULL`
- [ ] `last_seen_at INTEGER NOT NULL`
- [ ] `occurrences INTEGER NOT NULL DEFAULT 1`

Rules:

- [ ] Open predicates are first-class recall nodes.
- [ ] Predicate similarity is computed by cosine similarity.
- [ ] Similar predicate edges are computed on demand or cached top-K.
- [ ] Do not permanently materialize all-pairs predicate similarity edges.
- [ ] Do not store canonical embedding status on the predicate row.
- [ ] Predicate vector readiness is derived from `MemoryEmbeddingJobs` and `MemoryVectors`.

### `MemoryRelations`

Use a relation table for merge and supersession instead of overloading event rows.

- [ ] `relation_id INTEGER PRIMARY KEY`
- [ ] `src_event_id INTEGER NOT NULL REFERENCES MemoryEvents(event_id) ON DELETE CASCADE`
- [ ] `dst_event_id INTEGER NOT NULL REFERENCES MemoryEvents(event_id) ON DELETE CASCADE`
- [ ] `relation_type TEXT NOT NULL`
- [ ] `created_at INTEGER NOT NULL`
- [ ] `reason TEXT NOT NULL DEFAULT ''`

Allowed `relation_type` values:

- [ ] `merge_into`
- [ ] `supersedes`

Rules:

- [ ] Keep Memory merge/supersedes only if the prompt output or archive dedupe layer explicitly produces it.
- [ ] If memory archive does not produce these relations in the first implementation, leave the table empty.
- [ ] Do not carry old forced-tool merge fields into the memory parser.
- [ ] Soft-deleted events keep relation rows; every normal traversal must ignore relations whose source or target event is soft-deleted.
- [ ] Hard-deleted events cascade-delete relations.

## Memory Dedupe Policy

First implementation uses conservative write-time dedupe.

- [ ] Compute a dedupe signature from `event_type_norm`, `is_negated`, normalized `summary`, normalized participant roles, participant entities, and participant values.
- [ ] If the signature matches an existing non-deleted event in the same `conv_type` and `conv_id`, do not insert a new event.
- [ ] On exact signature match, increment `occurrences`, update `last_seen_at`, and create no `merge_into` relation.
- [ ] Do not dedupe by embedding similarity in Memory first implementation.
- [ ] Do not dedupe events with different `status` values.
- [ ] Do not dedupe when only summary vector or predicate vector similarity is high.
- [ ] `merge_into` and `supersedes` relations are reserved for explicit future archive/dedupe decisions; exact repeat dedupe uses `occurrences`.

## Vector Storage

First implementation should use SQLite BLOB vectors and application-layer cosine search.

Rationale:

- Avoid adding sqlite-vss or an external vector database before recall behavior stabilizes.
- Keep rebuild and delete-database workflows simple.
- Permit later replacement with a vector index behind the same repository API.

### `MemoryVectors`

- [ ] `vector_id INTEGER PRIMARY KEY`
- [ ] `owner_type TEXT NOT NULL`
- [ ] `owner_id INTEGER NOT NULL`
- [ ] `embedding_kind TEXT NOT NULL`
- [ ] `embedding BLOB NOT NULL`
- [ ] `dim INTEGER NOT NULL`
- [ ] `model TEXT NOT NULL`
- [ ] `model_version TEXT NOT NULL DEFAULT ''`
- [ ] `normalized INTEGER NOT NULL DEFAULT 1`
- [ ] `source_hash TEXT NOT NULL`
- [ ] `created_at INTEGER NOT NULL`

Ownership rules:

- [ ] `owner_type='event'` refers to `MemoryEvents.event_id`.
- [ ] `owner_type='predicate'` refers to `MemoryPredicates.predicate_id`.
- [ ] `owner_type='participant'` refers to `MemoryParticipants.participant_id`.
- [ ] SQLite cannot enforce a polymorphic foreign key directly; repository delete code must clean vectors for hard-deleted owners.
- [ ] Soft-deleted events keep vectors, but normal recall must ignore vectors owned by soft-deleted events.

Required `embedding_kind` values:

- [ ] `summary`
- [ ] `predicate`

Optional future values:

- [ ] `participant_value`
- [ ] `entity_profile`

Encoding:

- [ ] Store vectors as little-endian `float32` BLOB.
- [ ] Store L2-normalized vectors.
- [ ] Cosine similarity is dot product for normalized vectors.
- [ ] Validate `dim` on read.
- [ ] Treat a vector as stale when `model`, `model_version`, `dim`, or `source_hash` differs.

Performance boundary:

- [ ] Application-layer brute force is acceptable for the first implementation.
- [ ] Add a performance test before enabling large production memory sets.
- [ ] Revisit sqlite-vss or another vector index only after behavior is validated.

## Embedding Service And Backfill

- [ ] Add a new memory embedding client/service.
- [ ] Make embedding provider/model configurable.
- [ ] Batch embedding requests where possible.
- [ ] Store model name, model version, dimension, normalized flag, and source hash with each vector.
- [ ] Archive writes must not fail only because embedding generation failed.
- [ ] Failed embedding generation marks the corresponding `MemoryEmbeddingJobs` row as `failed` and records `last_error`.
- [ ] Successful vector writes create or replace the `MemoryVectors` row and complete the corresponding embedding job.
- [ ] Events/predicates without ready vectors remain recallable through FTS/entity/recent fallback.

Embedding client interface contract:

- [ ] Input is an ordered batch of UTF-8 strings.
- [ ] Output is an ordered batch of vectors with exactly the same length and order as input.
- [ ] The client must return `model`, `model_version`, `dim`, and `normalized`.
- [ ] If the provider returns unnormalized vectors, the memory embedding service normalizes them before storage.
- [ ] Empty input batch returns an empty output batch.
- [ ] Empty or whitespace-only text is rejected before provider call and becomes a failed embedding job.
- [ ] Error types must distinguish at least configuration error, provider/network error, invalid response, and unsupported dimension.
- [ ] Partial provider failure fails the whole batch unless the provider response can unambiguously map failures to individual inputs.

Embedding status values:

- [ ] `pending`
- [ ] `ready`
- [ ] `failed`
- [ ] `stale`

Backfill requirements:

- [ ] Provide a rebuild command for all memory embeddings.
- [ ] Provide an incremental backfill worker for `pending`, `failed`, and `stale` vectors.
- [ ] Persist enough state to resume after shutdown.
- [ ] Log retry count and last error.

Suggested table:

### `MemoryEmbeddingJobs`

- [ ] `job_id INTEGER PRIMARY KEY`
- [ ] `owner_type TEXT NOT NULL`
- [ ] `owner_id INTEGER NOT NULL`
- [ ] `embedding_kind TEXT NOT NULL`
- [ ] `status TEXT NOT NULL DEFAULT 'pending'`
- [ ] `retry_count INTEGER NOT NULL DEFAULT 0`
- [ ] `last_error TEXT NOT NULL DEFAULT ''`
- [ ] `created_at INTEGER NOT NULL`
- [ ] `updated_at INTEGER NOT NULL`

Rules:

- [ ] Jobs are the canonical status source for pending, failed, and stale work.
- [ ] Ready state is represented by a valid `MemoryVectors` row matching current model metadata and source hash.
- [ ] Failed jobs remain retryable until retry policy decides otherwise.

## Settings

Only expose high-value settings in WebUI for Memory. Keep the rest as internal constants until behavior is stable.

### WebUI Settings

- [ ] `memory_predicate_similarity_threshold`
  - Default: `0.8`
  - Range: `0.5` to `0.95`
  - Meaning: minimum cosine similarity for predicate-to-predicate traversal.

- [ ] `memory_recall_max_results`
  - Default: `8`
  - Range: `1` to `30`
  - Meaning: maximum rendered events.

- [ ] `memory_recall_recent_fallback`
  - Default: `true`
  - Meaning: allow recent-event fallback when semantic seeds are weak.

### Internal Constants For First Implementation

- [ ] `BFS_MAX_ENERGY = 5.0`
- [ ] `BFS_MAX_DEPTH = 3`
- [ ] `BFS_MAX_NODES = 256`
- [ ] `HUB_PENALTY_WEIGHT = 0.3`
- [ ] `TIME_DECAY_WEIGHT = 0.15`
- [ ] `SUMMARY_VECTOR_WEIGHT = 0.45`
- [ ] `PREDICATE_VECTOR_WEIGHT = 0.25`
- [ ] `ENTITY_EDGE_BASE_COST = 1.0`
- [ ] `PREDICATE_EDGE_BASE_COST = 1.0`
- [ ] `SESSION_EDGE_BASE_COST = 1.4`
- [ ] `HYPOTHETICAL_TO_ACTUAL_COST = infinity`

Move an internal constant to WebUI only after there is a demonstrated need to tune it.

## Recall Pipeline

The new recall pipeline is a hybrid graph and vector system.

### Stage 1: Seed Retrieval

Generate initial candidate events/nodes from:

- [ ] Summary FTS search.
- [ ] Summary embedding search.
- [ ] Participant entity exact/fuzzy match.
- [ ] Predicate embedding similarity.
- [ ] Recent events fallback, only when the above signals are weak or unavailable.

Seed outputs must include:

- [ ] Node id.
- [ ] Node type.
- [ ] Initial score.
- [ ] Signal source.
- [ ] Matched event id when directly available.

### Stage 2: Costed Graph Expansion

Use Dijkstra-style costed expansion, not plain unweighted BFS.

Supported node types:

- [ ] Event node.
- [ ] Entity node.
- [ ] Predicate node.
- [ ] Optional session/context node.

Supported edge types:

- [ ] Event -> participant entity.
- [ ] Event -> participant value.
- [ ] Event -> predicate.
- [ ] Predicate -> similar predicate.
- [ ] Event -> session/context.
- [ ] Event -> relation target for `merge_into` or `supersedes`.

Traversal rules:

- [ ] Use energy budget.
- [ ] Use hard depth cap.
- [ ] Use max expanded node cap.
- [ ] Keep parent pointers for path reconstruction.
- [ ] Penalize hub nodes by degree.
- [ ] Penalize old episodic edges by time decay.
- [ ] Block hypothetical-to-actual traversal by default.
- [ ] Allow actual-to-hypothetical traversal only when query seeds explicitly match hypothetical content.
- [ ] Do not merge similar nodes during traversal; only score paths.

Suggested edge cost shape:

```text
cost = base_edge_cost
     + HUB_PENALTY_WEIGHT * log10(degree(next_node) + 1)
     + TIME_DECAY_WEIGHT * age_penalty(edge_or_event)
     + context_penalty(status/context)
     + predicate_penalty(1 - cosine_similarity)
```

### Stage 3: Candidate Rerank

Final event ranking combines:

- [ ] Seed score.
- [ ] BFS path cost.
- [ ] Summary vector similarity.
- [ ] Predicate similarity, if relevant.
- [ ] Entity match strength.
- [ ] Time freshness.
- [ ] Occurrence count.
- [ ] Scope/session relevance.
- [ ] Summary quality penalty for too-short or context-dependent summaries.

Do not use `confidence` as a primary ranking signal in Memory.

### Stage 4: Render

Normal recall rendering shown to the model:

- [ ] Show event summary.
- [ ] Show relative event time.
- [ ] Show confidence.
- [ ] Do not show event id.
- [ ] Do not show event_type.
- [ ] Do not show participants.
- [ ] Do not show recall score.
- [ ] Do not show traversal path.
- [ ] Do not show internal context/status unless it is part of the summary itself.

Normal XML shape:

```xml
<memory>
  <mem when="1天前" confidence="0.95">...</mem>
  <mem when="3小时前" confidence="0.80">...</mem>
</memory>
```

Debug rendering:

- [ ] Debug render must be a separate code path.
- [ ] Debug render must never be injected into model context by default.
- [ ] Debug render may include event id, scores, paths, predicate matches, and seed sources.

## EntitySystem Ideas To Reuse

This project will not depend on `I:\github\FengM\entitySystem`.

Reusable ideas:

- [ ] Late binding: avoid write-time entity/predicate merging based on loose similarity.
- [ ] Spreading activation: use Dijkstra-style costed expansion.
- [ ] Hub penalty: high-degree generic nodes should be expensive to cross.
- [ ] Energy budget: traversal stops by accumulated cost, not only hop count.
- [ ] Parent-pointer paths: avoid copying full paths in the priority queue.
- [ ] Context penalty: status/context affects traversal cost.

Ideas not to copy directly:

- [ ] TypeDB dependency.
- [ ] ATOMIC closed predicate set.
- [ ] Aggressive ontology machinery.
- [ ] Cross-project runtime dependency.

## Explicit Conflicts With Current Code

These current-code behaviors must change for Memory:

- [ ] `archiver.py` currently imports `ARCHIVE_SYSTEM_PROMPT` from old `archive_prompt.py`.
- [ ] `archiver.py` currently has old event type normalization logic.
- [ ] `archive_memories.py` currently defines a forced-tool schema with closed `event_type` enum.
- [ ] Existing recall/rerank logic may use `confidence` as a scoring bonus.
- [ ] Existing render logic exposes confidence, modality, context, recall score, and recall path to the model.

## Migration Plan

- [ ] Add memory parser and schema behind a clear Memory boundary.
- [ ] Remove or bypass old archive tool schema for memory archive calls.
- [ ] Rebuild memory database.
- [ ] Re-run archive extraction from available cognition/dialogue sources.
- [ ] Generate summary and predicate embeddings.
- [ ] Enable new recall pipeline.
- [ ] Add WebUI settings.
- [ ] Add debug logging for seed sources, BFS expansion, and final rerank.
- [ ] Keep debug logs separate from model-visible render output.

## Test Plan

Parser tests:

- [ ] Extracts multiple `<event>{json}</event>` blocks inside `<extract>`.
- [ ] Ignores `<event>` blocks outside `<extract>`.
- [ ] Allows empty `<extract>`.
- [ ] Rejects missing or duplicated `<extract>`.
- [ ] Rejects markdown fenced event payloads.
- [ ] Rejects free text around event JSON.
- [ ] Accepts valid events and rejects invalid events in the same batch.
- [ ] Requires `summary`, `event_type`, and `roles`.

Storage tests:

- [ ] Stores every prompt-defined field.
- [ ] Applies documented defaults for missing optional fields.
- [ ] Preserves raw JSON.
- [ ] Stores `event_type_norm` without closed-enum mapping.
- [ ] Stores merge/supersedes relations only through the memory relation table.

Embedding tests:

- [ ] Predicates and summaries get vectors.
- [ ] Vectors are stored as normalized float32 BLOBs.
- [ ] Stale vectors are detected when model, version, dimension, or source hash changes.
- [ ] Embedding failure leaves archive write intact and creates retryable job state.
- [ ] Recall still works through FTS/entity/recent fallback when vectors are missing.

Recall tests:

- [ ] Exact summary query retrieves matching event.
- [ ] Semantically similar predicate crosses threshold and expands.
- [ ] Predicate below threshold does not expand.
- [ ] Hub node penalty prevents generic entities from dominating.
- [ ] Hypothetical events do not leak into actual recall without explicit query relevance.
- [ ] Repeated archive input dedupes or records occurrences according to Memory policy.
- [ ] Large predicate set top-K search stays within acceptable time.

Render tests:

- [ ] Normal render contains summary, relative time, and confidence only.
- [ ] Normal render does not contain id, event type, participants, score, or path.
- [ ] Debug render is isolated from model-visible context.

Determinism tests:

- [ ] Same memory state and same query produce stable top-K ordering.
- [ ] Tie-breaking is deterministic by score, time, then event id.

## Open Decisions

- [ ] Exact embedding provider/model.
- [ ] Exact memory table creation location.
- [ ] Whether participant value embeddings are required in the first implementation.
- [ ] Whether predicate top-K cache should be persisted or computed on demand.
- [ ] Exact archive re-run source and ordering.
