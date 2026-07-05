from __future__ import annotations

from llm.core.tool_calling.schema import (
    repair_arguments_by_declaration,
    validate_arguments_by_declaration,
)


DECLARATION = {
    "name": "demo",
    "parameters": {
        "type": "object",
        "properties": {
            "seconds": {"type": "integer", "minimum": 1, "maximum": 600},
            "quote": {"type": "string", "x-coerce-integer": True},
            "items": {"type": "array", "items": {"type": "integer"}},
        },
        "required": ["seconds"],
    },
}


def test_repair_arguments_by_declaration_coerces_safe_schema_values():
    repaired, changes = repair_arguments_by_declaration(
        {"seconds": "999", "quote": 42, "items": '["2",3]'},
        DECLARATION,
    )

    assert repaired == {"seconds": 600, "quote": "42", "items": [2, 3]}
    assert changes


def test_validate_arguments_by_declaration_accepts_repaired_args():
    repaired, _ = repair_arguments_by_declaration(
        {"seconds": "5", "quote": 42},
        DECLARATION,
    )

    ok, errors, summary = validate_arguments_by_declaration(repaired, DECLARATION)

    assert ok is True
    assert errors == []
    assert summary is None


def test_validate_arguments_by_declaration_reports_json_path_errors():
    ok, errors, summary = validate_arguments_by_declaration(
        {"seconds": "later"},
        DECLARATION,
    )

    assert ok is False
    assert summary == "arguments do not satisfy schema"
    assert errors == ["$.seconds: 'later' is not of type 'integer'"]


def test_tool_specific_schema_repairer_runs_after_generic_repair():
    def repairer(args):
        fixed = dict(args)
        fixed["extra"] = "added"
        return fixed, ["extra added"]

    repaired, changes = repair_arguments_by_declaration(
        {"seconds": "2"},
        DECLARATION,
        schema_repairer=repairer,
    )

    assert repaired == {"seconds": 2, "extra": "added"}
    assert changes[-1] == "extra added"


def test_discriminated_union_repair_uses_only_selected_branch():
    from tools.core import runtime_manage

    declaration = runtime_manage.TOOL_CONTRACT.declaration()

    repaired, changes = repair_arguments_by_declaration(
        {"action": "idle", "minutes": 15},
        declaration,
        runtime_manage.repair_schema_args,
    )

    assert repaired == {"action": "idle", "minutes": 15}
    assert changes == []

    repaired, changes = repair_arguments_by_declaration(
        {"action": "sleep", "minutes": 15},
        declaration,
        runtime_manage.repair_schema_args,
    )

    assert repaired == {"action": "sleep", "minutes": 30}
    assert changes == ["minutes: 15 -> 30 (range [30, 600])"]


def test_discriminated_union_repair_does_not_guess_unknown_branch():
    from tools.core import runtime_manage

    declaration = runtime_manage.TOOL_CONTRACT.declaration()

    repaired, changes = repair_arguments_by_declaration(
        {"action": "unknown", "minutes": 15},
        declaration,
        runtime_manage.repair_schema_args,
    )

    assert repaired == {"action": "unknown", "minutes": 15}
    assert changes == []


def test_goal_manage_schema_accepts_create_with_required_title():
    from tools.core import goal_manage

    declaration = goal_manage.TOOL_CONTRACT.declaration()

    ok, errors, summary = validate_arguments_by_declaration(
        {
            "action": "create",
            "goals": [
                {
                    "title": "整理目标管理",
                    "content": "修复 goal_manage create 参数声明",
                    "reason": "模型可见签名和后端校验必须一致",
                }
            ],
        },
        declaration,
    )
    assert ok is True
    assert errors == []
    assert summary is None

    ok, errors, summary = validate_arguments_by_declaration(
        {
            "action": "create",
            "goals": [
                {
                    "content": "修复 goal_manage create 参数声明",
                    "reason": "缺少 title 应该被拒绝",
                }
            ],
        },
        declaration,
    )
    assert ok is False
    assert summary == "arguments do not satisfy schema"
    assert any("not valid under any of the given schemas" in error for error in errors)


def test_browser_locator_schema_enforces_operation_specific_arguments():
    from tools.browser.browser_use import browser_locator

    declaration = browser_locator.TOOL_CONTRACT.declaration()

    accepted_args = [
        {"strategy": "css", "query": "input", "op": "fill", "input_text": "hello"},
        {"strategy": "css", "query": "button", "op": "press", "key": "Enter"},
        {"strategy": "css", "query": "a", "op": "read_attribute", "attribute": "href"},
        {"strategy": "css", "query": "select", "op": "select_option", "input_text": "a"},
        {"strategy": "css", "query": "select", "op": "select_option", "select_options": {"label": "A"}},
        {"strategy": "css", "query": "div", "op": "eval", "input_text": "el => el.textContent"},
    ]
    for args in accepted_args:
        ok, errors, summary = validate_arguments_by_declaration(args, declaration)
        assert ok is True, (args, errors)
        assert errors == []
        assert summary is None

    rejected_args = [
        {"strategy": "css", "query": "input", "op": "fill"},
        {"strategy": "css", "query": "button", "op": "press", "input_text": "Enter"},
        {"strategy": "css", "query": "a", "op": "read_attribute"},
        {"strategy": "css", "query": "select", "op": "select_option"},
        {"strategy": "css", "query": "select", "op": "select_option", "select_options": {}},
        {"strategy": "css", "query": "select", "op": "select_option", "options": {"exact": True}},
        {"strategy": "css", "query": "div", "op": "eval"},
    ]
    for args in rejected_args:
        ok, errors, summary = validate_arguments_by_declaration(args, declaration)
        assert ok is False, args
        assert summary == "arguments do not satisfy schema"
        assert any("not valid under any of the given schemas" in error for error in errors)


def test_group_notice_schema_enforces_action_specific_index_rules():
    from platforms.qq.tools.qq_group_info.get_group_notice import TOOL_CONTRACT

    group_notice_declaration = TOOL_CONTRACT.declaration()

    ok, errors, summary = validate_arguments_by_declaration(
        {"action": "list"},
        group_notice_declaration,
    )
    assert ok is True
    assert errors == []
    assert summary is None

    ok, errors, summary = validate_arguments_by_declaration(
        {"action": "read", "index": 0},
        group_notice_declaration,
    )
    assert ok is True
    assert errors == []
    assert summary is None

    ok, errors, summary = validate_arguments_by_declaration(
        {"action": "read"},
        group_notice_declaration,
    )
    assert ok is False
    assert summary == "arguments do not satisfy schema"
    assert any("not valid under any of the given schemas" in error for error in errors)

    ok, errors, summary = validate_arguments_by_declaration(
        {"action": "list", "index": 0},
        group_notice_declaration,
    )
    assert ok is False
    assert summary == "arguments do not satisfy schema"
    assert any("not valid under any of the given schemas" in error for error in errors)

