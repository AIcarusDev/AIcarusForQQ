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
    assert cfg["consolidation"] == {
        "enabled": False,
        "llm_tidy_enabled": False,
        "algorithmic_clustering_enabled": False,
        "dry_run": True,
        "solidify": False,
        "max_episode_candidates_per_sleep": 100,
        "sleep_maintenance_timeout_seconds": 300,
        "summary_max_inputs_per_sleep": 32,
        "summary_max_retries": 3,
        "provider": "",
        "model": "",
        "generation": {
            "temperature": 0.2,
            "max_output_tokens": 4000,
            "enable_thinking": False,
        },
    }


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


def test_default_memory_cfg_preserves_existing_consolidation_controls():
    from web.routes_settings import _default_memory_cfg

    cfg = _default_memory_cfg(
        {
            "memory": {
                "consolidation": {
                    "enabled": True,
                    "llm_tidy_enabled": True,
                    "algorithmic_clustering_enabled": True,
                    "dry_run": False,
                    "solidify": True,
                    "max_episode_candidates_per_sleep": 12,
                    "sleep_maintenance_timeout_seconds": 0,
                    "summary_max_inputs_per_sleep": 7,
                    "summary_max_retries": 5,
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

    assert cfg["consolidation"]["enabled"] is True
    assert cfg["consolidation"]["llm_tidy_enabled"] is True
    assert cfg["consolidation"]["algorithmic_clustering_enabled"] is True
    assert cfg["consolidation"]["dry_run"] is False
    assert cfg["consolidation"]["solidify"] is True
    assert cfg["consolidation"]["max_episode_candidates_per_sleep"] == 12
    assert cfg["consolidation"]["sleep_maintenance_timeout_seconds"] == 0
    assert cfg["consolidation"]["summary_max_inputs_per_sleep"] == 7
    assert cfg["consolidation"]["summary_max_retries"] == 5
    assert cfg["consolidation"]["provider"] == "memory"
    assert cfg["consolidation"]["model"] == "memory-model"
    assert cfg["consolidation"]["generation"] == {
        "temperature": 0.1,
        "max_output_tokens": 2048,
        "enable_thinking": True,
    }
