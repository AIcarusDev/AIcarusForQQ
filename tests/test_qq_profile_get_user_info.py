from tools import build_tools
from tools.namespaces import NamespaceRuntimeState, load_namespace_registry
from platforms.qq.tools.qq_profile.get_user_info import _format_user_info


class FakeClient:
    connected = True
    bot_id = "10000"
    _loop = None


def test_get_user_info_is_registered_in_qq_profile():
    registry = load_namespace_registry()
    state = NamespaceRuntimeState()
    state.open("qq_profile", registry, 1)
    collection = build_tools(
        {"vision": False, "tts": {"enabled": False}},
        namespace_state=state,
        current_round=1,
        default_ttl_rounds=5,
        current_platform="qq",
        qq_client=FakeClient(),
    )

    spec = collection.all_specs["qq_profile.get_user_info"]
    assert spec.namespace == "qq_profile"
    assert "qq_profile.get_user_info" in collection.active_names()


def test_format_user_info_keeps_stable_profile_fields():
    result = _format_user_info(
        {
            "user_id": 12345,
            "nickname": "Alice",
            "sex": "female",
            "age": 20,
            "qid": "aliceqid",
            "level": 16,
            "login_days": 365,
            "longNick": "hello",
        },
        "12345",
    )

    assert result == {
        "qq_number": "12345",
        "nickname": "Alice",
        "qid": "aliceqid",
        "sex": "女",
        "age": 20,
        "level": 16,
        "login_days": 365,
        "signature": "hello",
    }



