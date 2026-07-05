from __future__ import annotations

from types import SimpleNamespace

from platforms.qq.tools.qq_group_info import query_group_members


class FakeLoop:
    def is_running(self) -> bool:
        return True


class FakeClient:
    connected = True

    def __init__(self, members: list[dict]):
        self._loop = FakeLoop()
        self.members = members
        self.calls: list[tuple[str, dict]] = []

    def send_api(self, action: str, params: dict):
        self.calls.append((action, params))
        return self.members


def _handler(monkeypatch, members: list[dict]):
    monkeypatch.setattr(query_group_members, "run_coroutine_sync", lambda coro, _loop, timeout=None: coro)
    client = FakeClient(members)
    session = SimpleNamespace(conv_type="group", conv_id="1234")
    return query_group_members.make_handler(client, lambda: session), client


def _members(count: int) -> list[dict]:
    rows = []
    for index in range(count):
        rows.append({
            "user_id": 10000 + index,
            "nickname": f"昵称{index}",
            "card": f"名片{index}" if index % 2 == 0 else "",
            "role": "member",
        })
    rows[0]["role"] = "owner"
    rows[1]["role"] = "admin"
    rows[2]["role"] = "admin"
    return rows


def test_query_group_members_list_admins_returns_owner_and_admins(monkeypatch):
    execute, client = _handler(monkeypatch, _members(25))

    result = execute(action="list_admins")

    assert client.calls == [("get_group_member_list", {"group_id": 1234})]
    assert result["action"] == "list_admins"
    assert result["total_in_group"] == 25
    assert result["returned"] == 3
    assert [member["role"] for member in result["members"]] == ["owner", "admin", "admin"]
    assert result["members"][0] == {
        "account": "10000",
        "name": "昵称0",
        "card": "名片0",
        "role": "owner",
    }


def test_query_group_members_list_members_paginates_by_member_order(monkeypatch):
    execute, _client = _handler(monkeypatch, _members(25))

    result = execute(action="list_members", page=2)

    assert result["action"] == "list_members"
    assert result["page"] == 2
    assert result["page_size"] == 20
    assert result["has_more"] is False
    assert result["returned"] == 5
    assert [member["account"] for member in result["members"]] == [
        "10020",
        "10021",
        "10022",
        "10023",
        "10024",
    ]
    assert result["members"][1]["card"] == ""


def test_query_group_members_search_matches_nickname_or_card_and_limits(monkeypatch):
    members = _members(18)
    for index, member in enumerate(members):
        member["nickname"] = f"成员{index}"
        member["card"] = ""
    members[0]["card"] = "小明-群名片"
    members[1]["nickname"] = "小明昵称"
    for index in range(2, 14):
        members[index]["card"] = f"第{index}个小明"
    execute, _client = _handler(monkeypatch, members)

    result = execute(action="search", query="小明")

    assert result["action"] == "search"
    assert result["query"] == "小明"
    assert result["total_matches"] == 14
    assert result["returned"] == 10
    assert result["truncated"] is True
    assert result["members"][0]["card"] == "小明-群名片"
    assert result["members"][1]["name"] == "小明昵称"


def test_query_group_members_rejects_wrong_action_fields():
    result = query_group_members.make_handler(None, lambda: None)(action="list_admins", page=1)

    assert result == {
        "error": "action=list_admins 时不能传其他参数",
        "extra": ["page"],
    }
