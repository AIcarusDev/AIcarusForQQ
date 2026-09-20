from __future__ import annotations

from config_loader import (
    read_env_keys,
    read_env_proxies,
    read_env_values,
    save_env_key,
    save_env_proxy,
    save_env_value,
)


def test_env_key_read_write_masks_and_preserves_unrelated_lines(tmp_path):
    env_file = tmp_path / "settings.txt"
    env_file.write_text("API_ONE=abcdef\nOTHER=value\n", encoding="utf-8")

    assert read_env_keys(["API_ONE", "API_TWO"], env_path=str(env_file)) == {
        "API_ONE": "**cdef",
        "API_TWO": "",
    }

    save_env_key("API_ONE", "masked****", env_path=str(env_file))
    assert read_env_values(["API_ONE"], env_path=str(env_file)) == {"API_ONE": "abcdef"}

    save_env_key("API_ONE", "new-secret", env_path=str(env_file))
    save_env_key("API_TWO", "second-secret", env_path=str(env_file))

    assert read_env_values(["API_ONE", "API_TWO", "OTHER"], env_path=str(env_file)) == {
        "API_ONE": "new-secret",
        "API_TWO": "second-secret",
        "OTHER": "value",
    }


def test_save_env_value_deletes_empty_values(tmp_path):
    env_file = tmp_path / "settings.txt"
    env_file.write_text("PLAIN=keep\nREMOVE=gone\n", encoding="utf-8")

    save_env_value("REMOVE", "", env_path=str(env_file))
    save_env_value("ADDED", "value", env_path=str(env_file))

    assert read_env_values(["PLAIN", "REMOVE", "ADDED"], env_path=str(env_file)) == {
        "PLAIN": "keep",
        "REMOVE": "",
        "ADDED": "value",
    }


def test_browser_proxy_round_trips_through_legacy_env_helpers(tmp_path):
    env_file = tmp_path / "settings.txt"
    env_file.write_text("UNRELATED=keep\n", encoding="utf-8")

    save_env_proxy("BROWSER_PROXY", "http://127.0.0.1:7890", env_path=str(env_file))
    proxies = read_env_proxies(env_path=str(env_file))

    assert proxies["BROWSER_PROXY"].endswith("7890")
    assert "http://127.0.0.1:7890" not in str(proxies)
    assert "UNRELATED=keep" in env_file.read_text(encoding="utf-8")

    save_env_proxy("BROWSER_PROXY", "", env_path=str(env_file))
    assert "BROWSER_PROXY=" not in env_file.read_text(encoding="utf-8")
