from __future__ import annotations

from config_loader import (
    read_env_imap,
    read_env_keys,
    read_env_proxies,
    read_env_smtp,
    read_env_values,
    save_env_imap,
    save_env_key,
    save_env_proxy,
    save_env_smtp,
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


def test_smtp_and_imap_helpers_mask_passwords_and_skip_masked_updates(tmp_path):
    env_file = tmp_path / "settings.txt"
    env_file.write_text(
        "AICQ_SMTP_HOST=mail.local\n"
        "AICQ_SMTP_PASSWORD=secret\n"
        "AICQ_IMAP_HOST=imap.local\n"
        "AICQ_IMAP_PASSWORD=imap-secret\n",
        encoding="utf-8",
    )

    smtp = read_env_smtp(env_path=str(env_file))
    imap = read_env_imap(env_path=str(env_file))
    assert smtp["AICQ_SMTP_PASSWORD"] == "**cret"
    assert imap["AICQ_IMAP_PASSWORD"] == "*******cret"

    save_env_smtp({"AICQ_SMTP_PASSWORD": "****", "AICQ_SMTP_PORT": "465"}, env_path=str(env_file))
    save_env_imap({"AICQ_IMAP_PASSWORD": "", "AICQ_IMAP_PORT": "993"}, env_path=str(env_file))

    assert read_env_values(
        ["AICQ_SMTP_PASSWORD", "AICQ_SMTP_PORT", "AICQ_IMAP_PASSWORD", "AICQ_IMAP_PORT"],
        env_path=str(env_file),
    ) == {
        "AICQ_SMTP_PASSWORD": "secret",
        "AICQ_SMTP_PORT": "465",
        "AICQ_IMAP_PASSWORD": "",
        "AICQ_IMAP_PORT": "993",
    }
