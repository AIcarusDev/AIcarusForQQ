from __future__ import annotations

import gzip
import json

from llm.discarded_response_log import save_cognition_prefill_discard


def _base_kwargs() -> dict:
    return {
        "provider": "local-qwen",
        "model": "Qwen3.6-27B",
        "feature": "main_round",
        "subfeature": "",
        "prompt_snapshot_id": "snap-1",
        "agent_run_id": "round-1",
        "context": {"focus": "private_1"},
        "retry_attempt": 2,
        "similarity": 0.98765,
        "matched_index": 3,
        "discarded_cognition": "重复 cognition",
        "matched_cognition": "历史 cognition",
        "chosen_prefill": "我先重新核对上下文",
        "visible_cognitions_count": 8,
        "prefill_exclusions": ("用过的 prefill",),
        "guard_config": {"similarity_threshold": 0.9, "min_chars": 60},
    }


def test_save_cognition_prefill_discard_writes_metadata(tmp_path):
    event_id = save_cognition_prefill_discard(
        {"root_dir": str(tmp_path), "maintenance_interval_seconds": 0},
        **_base_kwargs(),
    )

    assert event_id
    files = list(tmp_path.glob("*/cognition_prefill.jsonl"))
    assert len(files) == 1
    record = json.loads(files[0].read_text(encoding="utf-8").strip())

    assert record["event_id"] == event_id
    assert record["event_type"] == "cognition_prefill_discard"
    assert record["provider"] == "local-qwen"
    assert record["prompt_snapshot_id"] == "snap-1"
    assert record["context"] == {"focus": "private_1"}
    assert record["retry_attempt"] == 2
    assert record["guard"]["similarity"] == 0.9877
    assert record["guard"]["matched_index"] == 3
    assert record["guard"]["visible_cognitions_count"] == 8
    assert record["discarded"]["text"] == "重复 cognition"
    assert record["matched"]["text"] == "历史 cognition"
    assert record["chosen_prefill"]["text"] == "我先重新核对上下文"


def test_save_cognition_prefill_discard_rotates_large_jsonl(tmp_path):
    cfg = {
        "root_dir": str(tmp_path),
        "max_file_bytes": 1,
        "maintenance_interval_seconds": 0,
    }
    save_cognition_prefill_discard(cfg, **_base_kwargs())
    save_cognition_prefill_discard(
        cfg,
        **{**_base_kwargs(), "discarded_cognition": "第二条 cognition"},
    )

    gz_files = list(tmp_path.glob("*/*.jsonl.gz"))
    jsonl_files = list(tmp_path.glob("*/cognition_prefill.jsonl"))
    assert len(gz_files) == 1
    assert len(jsonl_files) == 1

    with gzip.open(gz_files[0], "rt", encoding="utf-8") as f:
        rotated_record = json.loads(f.readline())
    current_record = json.loads(jsonl_files[0].read_text(encoding="utf-8").strip())

    assert rotated_record["discarded"]["text"] == "重复 cognition"
    assert current_record["discarded"]["text"] == "第二条 cognition"
