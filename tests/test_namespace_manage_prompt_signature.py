from __future__ import annotations

from tools.contract import get_contract_from_module
from tools.core import namespace_manage


def test_namespace_manage_prompt_signature_renders_fields_not_required_union():
    contract = get_contract_from_module(namespace_manage)

    assert contract is not None
    signature = contract.prompt_signature()

    assert "namespace_manage(args: {" in signature
    assert "open?: string[];" in signature
    assert "close?: string[];" in signature
    assert "preview?: string[];" in signature
    assert "search?: string;" in signature
    assert "unknown | unknown" not in signature
