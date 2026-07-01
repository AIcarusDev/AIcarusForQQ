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
