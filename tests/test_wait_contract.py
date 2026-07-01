from __future__ import annotations

from tools.browser.browser_runtime import wait_browser_event as browser_wait_mod
from tools.core.wait import wait as core_wait_mod
from tools.qq.qq_runtime import wait_qq_event as qq_wait_mod


def test_core_wait_is_short_fuzzy_wait_only():
    declaration = core_wait_mod.DECLARATION

    assert declaration["name"] == "wait"
    assert declaration["parameters"]["properties"]["seconds"]["maximum"] == 15
    assert declaration["parameters"]["required"] == ["seconds"]
    assert "early_trigger" not in declaration["parameters"]["properties"]


def test_core_wait_repair_maps_timeout_to_seconds():
    repaired, changes = core_wait_mod.repair_schema_args({"timeout": "3"})

    assert repaired == {"seconds": 3}
    assert changes == ["timeout -> seconds", "seconds: string -> int"]


def test_qq_wait_repair_schema_args_maps_legacy_social_fields():
    repaired, changes = qq_wait_mod.repair_schema_args(
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


def test_qq_wait_sanitize_rejects_world_scope():
    _, changes, error = qq_wait_mod.sanitize_semantic_args(
        {"seconds": 1, "early_trigger": {"scope": "world", "condition": "any_change"}}
    )

    assert changes == []
    assert error == "invalid early_trigger.scope: 'world'"


def test_qq_wait_pending_trigger_matches_social_scope_only():
    assert qq_wait_mod._pending_trigger_matches(
        {"scope": "session", "condition": "mentioned"},
        "mentioned",
    )
    assert not qq_wait_mod._pending_trigger_matches(
        {"scope": "session", "condition": "mentioned"},
        "any_change",
    )


def test_browser_wait_sanitize_accepts_browser_any_change_only():
    args = {"seconds": 1, "early_trigger": {"scope": "BROWSER", "condition": "ANY_MESSAGE"}}

    repaired, changes, error = browser_wait_mod.sanitize_semantic_args(args)

    assert error is None
    assert changes == ["normalized early_trigger"]
    assert repaired["early_trigger"] == {"scope": "browser", "condition": "any_change"}


def test_browser_wait_sanitize_rejects_mentioned_and_world_scope():
    _, changes, error = browser_wait_mod.sanitize_semantic_args(
        {"seconds": 1, "early_trigger": {"scope": "browser", "condition": "mentioned"}}
    )
    assert changes == []
    assert error == "invalid early_trigger.condition: 'mentioned'"

    _, changes, error = browser_wait_mod.sanitize_semantic_args(
        {"seconds": 1, "early_trigger": {"scope": "world", "condition": "any_change"}}
    )
    assert changes == []
    assert error == "invalid early_trigger.scope: 'world'"


def test_browser_signature_changed_compares_hashes():
    assert browser_wait_mod._browser_signature_changed(None, {"hash": "a"}) is True
    assert browser_wait_mod._browser_signature_changed({"hash": "a"}, {"hash": "b"}) is True
    assert browser_wait_mod._browser_signature_changed({"hash": "a"}, {"hash": "a"}) is False
