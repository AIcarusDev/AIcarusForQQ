from __future__ import annotations

import pytest

from llm.core.tool_calling.pipeline import process_tool_arguments
from tools.contract import get_contract_from_module
from tools.core import namespace_manage


@pytest.mark.parametrize("field", ["open", "close", "preview"])
def test_namespace_manage_accepts_single_namespace_string_for_list_fields(field):
    contract = get_contract_from_module(namespace_manage)

    assert contract is not None
    result = process_tool_arguments(
        f'{{"{field}":"qq_contacts"}}',
        "namespace_manage",
        "test",
        contract.declaration(),
        namespace_manage.repair_schema_args,
        namespace_manage.sanitize_semantic_args,
    )

    assert result.ok is True
    assert result.args == {field: ["qq_contacts"]}
    assert result.schema_changes == (f"{field}: string -> single-item array",)
