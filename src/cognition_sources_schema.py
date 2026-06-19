"""Schema for durable cognition-source identity storage."""

COGNITION_SOURCES_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS CognitionSources (
    source_uid TEXT PRIMARY KEY,
    source_kind TEXT NOT NULL DEFAULT 'cognition',
    origin_type TEXT NOT NULL DEFAULT '',
    origin_id TEXT NOT NULL DEFAULT '',
    prompt_source_id TEXT NOT NULL DEFAULT '',
    source_seq INTEGER,
    source_timestamp TEXT NOT NULL DEFAULT '',
    cognition_text TEXT NOT NULL DEFAULT '',
    cognition_hash TEXT NOT NULL DEFAULT '',
    created_at INTEGER NOT NULL,
    last_seen_at INTEGER NOT NULL,
    metadata_json TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_cognition_sources_origin
    ON CognitionSources(origin_type, origin_id);
CREATE INDEX IF NOT EXISTS idx_cognition_sources_hash
    ON CognitionSources(cognition_hash);
"""
