from __future__ import annotations

from tools.core.wait import wait as wait_mod


def test_repair_schema_args_maps_legacy_wait_fields():
    repaired, changes = wait_mod.repair_schema_args(
        {
            "timeout": "3",
            "early_trigger": {"scope": "global", "condition": "any_message"},
        }
    )

    assert repaired == {
        "seconds": 3,
        "early_trigger": {"scope": "platforms", "condition": "any_change"},
    }
    assert changes == [
        "timeout -> seconds",
        "seconds: string -> int",
        "early_trigger.scope: global -> platforms",
        "early_trigger.condition: any_message -> any_change",
    ]


def test_sanitize_semantic_args_normalizes_trigger_values():
    args = {"seconds": 1, "early_trigger": {"scope": "WORLD", "condition": "ANY_MESSAGE"}}

    repaired, changes, error = wait_mod.sanitize_semantic_args(args)

    assert error is None
    assert changes == ["normalized early_trigger"]
    assert repaired["early_trigger"] == {"scope": "world", "condition": "any_change"}


def test_sanitize_semantic_args_rejects_invalid_browser_mention_trigger():
    _, changes, error = wait_mod.sanitize_semantic_args(
        {"seconds": 1, "early_trigger": {"scope": "browser", "condition": "mentioned"}}
    )

    assert changes == []
    assert error == "early_trigger condition 'mentioned' is not valid for browser scope"


def test_pending_trigger_matches_social_scope_only():
    assert wait_mod._pending_trigger_matches(
        {"scope": "session", "condition": "mentioned"},
        "mentioned",
    )
    assert not wait_mod._pending_trigger_matches(
        {"scope": "session", "condition": "mentioned"},
        "any_change",
    )
    assert not wait_mod._pending_trigger_matches(
        {"scope": "browser", "condition": "any_change"},
        "mentioned",
    )


def test_browser_signature_changed_compares_hashes():
    assert wait_mod._browser_signature_changed(None, {"hash": "a"}) is True
    assert wait_mod._browser_signature_changed({"hash": "a"}, {"hash": "b"}) is True
    assert wait_mod._browser_signature_changed({"hash": "a"}, {"hash": "a"}) is False
