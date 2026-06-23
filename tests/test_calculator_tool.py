from __future__ import annotations

import json

from llm.core.tool_calling.pipeline import process_tool_arguments
import tools.core.calculator as calculator


def test_calculator_handles_decimal_subtraction_exactly():
    result = calculator.execute("9.9 - 9.11")

    assert result["ok"] is True
    assert result["result"] == "0.79"
    assert result["approximate"] is False


def test_calculator_truncates_equals_suffix_before_evaluation():
    result = calculator.execute("9.9-9.11=?")

    assert result["ok"] is True
    assert result["expression"] == "9.9-9.11"
    assert result["result"] == "0.79"
    assert "truncated expression at equals sign" in result["normalization"]


def test_calculator_uses_decimal_for_common_float_trap():
    result = calculator.execute("0.1 + 0.2")

    assert result["ok"] is True
    assert result["result"] == "0.3"


def test_calculator_respects_parentheses_and_power():
    result = calculator.execute("(2 + 3) ^ 3 / 5")

    assert result["ok"] is True
    assert result["result"] == "25"


def test_calculator_reports_repeating_division_as_approximate():
    result = calculator.execute("1 / 3")

    assert result["ok"] is True
    assert result["result"].startswith("0.333333333333333333")
    assert result["approximate"] is True


def test_calculator_rounds_repeating_division_when_requested():
    result = calculator.execute("1 / 3", round_to=2)

    assert result["ok"] is True
    assert result["result"] == "0.33"


def test_calculator_rejects_unsafe_expression_text():
    result = calculator.execute('__import__("os").system("whoami")')

    assert result["ok"] is False
    assert "非法字符" in result["error"]


def test_repair_schema_args_maps_aliases_and_drops_extra_fields():
    repaired, changes = calculator.repair_schema_args(
        {
            "formula": "9.9-9.11=?",
            "answer": "0.79",
            "round_to": "2",
        }
    )

    assert repaired == {"expression": "9.9-9.11=?", "round_to": "2"}
    assert changes == [
        "formula -> expression",
        "removed unsupported calculator arguments",
    ]


def test_calculator_argument_pipeline_accepts_common_model_mistakes():
    result = process_tool_arguments(
        json.dumps(
            {
                "formula": "9.9-9.11=?",
                "answer": "0.79",
                "round_to": "2",
            },
            ensure_ascii=False,
        ),
        "calculator",
        "test",
        calculator.DECLARATION,
        calculator.repair_schema_args,
    )

    assert result.ok is True
    assert result.args == {"expression": "9.9-9.11=?", "round_to": 2}
    assert result.schema_changes == (
        "round_to: '2' -> 2 (int)",
        "formula -> expression",
        "removed unsupported calculator arguments",
    )
