import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def test_default_memory_cfg_includes_events_controls():
    from web.routes_settings import _default_memory_cfg

    cfg = _default_memory_cfg({"memory": {}})

    assert cfg["events"] == {
        "recall_limit": 6,
        "world_query_chunks": 6,
        "cognition_query_chunks": 3,
    }
    assert cfg["processing"] == {
        "enabled": False,
        "event_structuring_enabled": False,
        "algorithmic_storyline_enabled": False,
        "dry_run": True,
        "solidify": False,
        "max_candidate_storylines_per_maintenance": 100,
        "maintenance_timeout_seconds": 300,
        "storyline_synthesis_max_inputs_per_maintenance": 32,
        "storyline_synthesis_max_retries": 3,
        "provider": "",
        "model": "",
        "generation": {
            "temperature": 0.2,
            "max_output_tokens": 4000,
            "enable_thinking": False,
        },
    }


def test_default_memory_cfg_drops_retired_capacity_limits():
    from web.routes_settings import _default_memory_cfg

    cfg = _default_memory_cfg({"memory": {"max_active": 8, "max_passive": 15}})

    assert "max_active" not in cfg
    assert "max_passive" not in cfg


def test_default_memory_cfg_preserves_existing_events_controls():
    from web.routes_settings import _default_memory_cfg

    cfg = _default_memory_cfg({
        "memory": {
            "events": {
                "recall_limit": 9,
                "world_query_chunks": 12,
                "cognition_query_chunks": 4,
            }
        }
    })

    assert cfg["events"]["recall_limit"] == 9
    assert cfg["events"]["world_query_chunks"] == 12
    assert cfg["events"]["cognition_query_chunks"] == 4


def test_default_memory_cfg_preserves_existing_processing_controls():
    from web.routes_settings import _default_memory_cfg

    cfg = _default_memory_cfg(
        {
            "memory": {
                "processing": {
                    "enabled": True,
                    "event_structuring_enabled": True,
                    "algorithmic_storyline_enabled": True,
                    "dry_run": False,
                    "solidify": True,
                    "max_candidate_storylines_per_maintenance": 12,
                    "maintenance_timeout_seconds": 0,
                    "storyline_synthesis_max_inputs_per_maintenance": 7,
                    "storyline_synthesis_max_retries": 5,
                    "provider": "memory",
                    "model": "memory-model",
                    "generation": {
                        "temperature": 0.1,
                        "max_output_tokens": 2048,
                        "enable_thinking": True,
                    },
                }
            }
        }
    )

    assert cfg["processing"]["enabled"] is True
    assert cfg["processing"]["event_structuring_enabled"] is True
    assert cfg["processing"]["algorithmic_storyline_enabled"] is True
    assert cfg["processing"]["dry_run"] is False
    assert cfg["processing"]["solidify"] is True
    assert cfg["processing"]["max_candidate_storylines_per_maintenance"] == 12
    assert cfg["processing"]["maintenance_timeout_seconds"] == 0
    assert cfg["processing"]["storyline_synthesis_max_inputs_per_maintenance"] == 7
    assert cfg["processing"]["storyline_synthesis_max_retries"] == 5
    assert cfg["processing"]["provider"] == "memory"
    assert cfg["processing"]["model"] == "memory-model"
    assert cfg["processing"]["generation"] == {
        "temperature": 0.1,
        "max_output_tokens": 2048,
        "enable_thinking": True,
    }
