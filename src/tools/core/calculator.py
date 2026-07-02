"""calculator.py - exact decimal arithmetic tool."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, DivisionByZero, Inexact, InvalidOperation, Rounded, localcontext
import re
from typing import Any

from pydantic import Field

from tools.contract import ToolArgsModel, ToolContract

class CalculatorArgs(ToolArgsModel):
    expression: str = Field(
        description=(
            "要计算的纯算术表达式。支持十进制数字、+、-、*、/、^ 或 **、括号；"
            "乘法须显式写 *。例如：9.9 - 9.11"
        ),
    )
    round_to: int | None = Field(
        default=None,
        ge=0,
        le=30,
        description="可选；当需要固定小数位时可填写。",
    )


TOOL_CONTRACT = ToolContract(
    name="calculator",
    description="语言模型不擅长计算，尤其是类似 9.9 - 9.11 的小数运算。如果遇到这类问题，该工具可以帮助进行精确的十进制算术计算。",
    args_model=CalculatorArgs,
)

_ARG_ALIASES: tuple[str, ...] = ("expr", "formula", "query", "input", "calculation")
_ALLOWED_ARG_KEYS = {"expression", "round_to"}
_NUMBER_RE = re.compile(
    r"(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
)
_TRANSLATION = str.maketrans(
    {
        "＋": "+",
        "－": "-",
        "−": "-",
        "﹣": "-",
        "＊": "*",
        "×": "*",
        "÷": "/",
        "／": "/",
        "（": "(",
        "）": ")",
        "＾": "^",
    }
)
_TRAILING_PUNCTUATION = "?？。；;"
_MAX_ABS_EXPONENT = 1000


class CalculatorError(ValueError):
    """Raised when a calculator expression cannot be evaluated safely."""


@dataclass(frozen=True)
class _Token:
    kind: str
    value: str
    position: int


class _Parser:
    def __init__(self, tokens: list[_Token]) -> None:
        self._tokens = tokens
        self._index = 0

    def parse(self) -> Decimal:
        if not self._tokens:
            raise CalculatorError("表达式为空")
        value = self._parse_expression()
        if self._peek() is not None:
            token = self._peek()
            assert token is not None
            raise CalculatorError(f"无法解析 token {token.value!r}，位置 {token.position}")
        return value

    def _peek(self) -> _Token | None:
        if self._index >= len(self._tokens):
            return None
        return self._tokens[self._index]

    def _take(self) -> _Token | None:
        token = self._peek()
        if token is not None:
            self._index += 1
        return token

    def _match_op(self, *operators: str) -> str | None:
        token = self._peek()
        if token is None or token.kind != "op" or token.value not in operators:
            return None
        self._index += 1
        return token.value

    def _parse_expression(self) -> Decimal:
        value = self._parse_term()
        while True:
            op = self._match_op("+", "-")
            if op is None:
                return value
            right = self._parse_term()
            if op == "+":
                value += right
            else:
                value -= right

    def _parse_term(self) -> Decimal:
        value = self._parse_power()
        while True:
            op = self._match_op("*", "/")
            if op is None:
                return value
            right = self._parse_power()
            if op == "*":
                value *= right
            else:
                if right == 0:
                    raise CalculatorError("除数不能为 0")
                value /= right

    def _parse_power(self) -> Decimal:
        value = self._parse_unary()
        op = self._match_op("**", "^")
        if op is None:
            return value
        exponent = self._parse_power()
        return _decimal_power(value, exponent)

    def _parse_unary(self) -> Decimal:
        op = self._match_op("+", "-")
        if op is None:
            return self._parse_primary()
        value = self._parse_unary()
        return value if op == "+" else -value

    def _parse_primary(self) -> Decimal:
        token = self._take()
        if token is None:
            raise CalculatorError("表达式不完整")
        if token.kind == "number":
            try:
                return Decimal(token.value)
            except InvalidOperation as exc:
                raise CalculatorError(f"非法数字: {token.value!r}") from exc
        if token.kind == "lparen":
            value = self._parse_expression()
            if self._match_op(")") is None:
                raise CalculatorError("缺少右括号")
            return value
        raise CalculatorError(f"位置 {token.position} 需要数字或左括号")


def repair_schema_args(args: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """Normalize common model mistakes before strict schema validation."""
    if not isinstance(args, dict):
        return args, []

    repaired = dict(args)
    changes: list[str] = []

    if "expression" not in repaired:
        for alias in _ARG_ALIASES:
            value = repaired.get(alias)
            if isinstance(value, str) and value.strip():
                repaired["expression"] = value
                changes.append(f"{alias} -> expression")
                break

    expression = repaired.get("expression")
    if isinstance(expression, (int, float)) and not isinstance(expression, bool):
        repaired["expression"] = repr(expression)
        changes.append("expression: number -> string")

    extra_keys = sorted(key for key in repaired if key not in _ALLOWED_ARG_KEYS)
    for key in extra_keys:
        repaired.pop(key, None)
    if extra_keys:
        changes.append("removed unsupported calculator arguments")

    return repaired, changes


def _normalize_expression(raw_expression: Any) -> tuple[str, list[str]]:
    text = str(raw_expression or "").strip()
    changes: list[str] = []
    if not text:
        return "", changes

    if (
        len(text) >= 2
        and text[0] == text[-1]
        and text[0] in {"'", '"', "`"}
    ):
        text = text[1:-1].strip()
        changes.append("stripped wrapping quotes")

    translated = text.translate(_TRANSLATION)
    if translated != text:
        text = translated
        changes.append("normalized unicode operators")

    equal_positions = [pos for marker in ("=", "＝") if (pos := text.find(marker)) >= 0]
    if equal_positions:
        text = text[:min(equal_positions)].strip()
        changes.append("truncated expression at equals sign")

    trimmed = text.rstrip(_TRAILING_PUNCTUATION).strip()
    if trimmed != text:
        text = trimmed
        changes.append("removed trailing punctuation")

    return text, changes


def _tokenize(expression: str) -> list[_Token]:
    tokens: list[_Token] = []
    index = 0
    length = len(expression)
    while index < length:
        char = expression[index]
        if char.isspace():
            index += 1
            continue

        number_match = _NUMBER_RE.match(expression, index)
        if number_match is not None:
            value = number_match.group(0)
            tokens.append(_Token("number", value, index))
            index = number_match.end()
            continue

        if expression.startswith("**", index):
            tokens.append(_Token("op", "**", index))
            index += 2
            continue

        if char in "+-*/^":
            tokens.append(_Token("op", char, index))
            index += 1
            continue

        if char == "(":
            tokens.append(_Token("lparen", char, index))
            index += 1
            continue

        if char == ")":
            tokens.append(_Token("op", char, index))
            index += 1
            continue

        raise CalculatorError(f"非法字符 {char!r}，位置 {index}")

    return tokens


def _decimal_power(base: Decimal, exponent: Decimal) -> Decimal:
    if exponent != exponent.to_integral_value():
        raise CalculatorError("指数必须是整数")
    exponent_int = int(exponent)
    if abs(exponent_int) > _MAX_ABS_EXPONENT:
        raise CalculatorError(f"指数绝对值不能超过 {_MAX_ABS_EXPONENT}")
    try:
        return base ** exponent_int
    except (InvalidOperation, OverflowError) as exc:
        raise CalculatorError(f"无法计算指数: {exc}") from exc


def _evaluate(expression: str) -> tuple[Decimal, bool]:
    try:
        tokens = _tokenize(expression)
        with localcontext() as ctx:
            ctx.prec = 80
            ctx.clear_flags()
            value = _Parser(tokens).parse()
            approximate = bool(ctx.flags[Inexact] or ctx.flags[Rounded])
            return +value, approximate
    except DivisionByZero as exc:
        raise CalculatorError("除数不能为 0") from exc
    except (InvalidOperation, OverflowError) as exc:
        raise CalculatorError(f"表达式无法计算: {exc}") from exc


def _format_decimal(value: Decimal, round_to: int | None = None) -> str:
    if round_to is not None:
        places = min(30, max(0, int(round_to)))
        precision = max(80, len(value.as_tuple().digits) + places + 5)
        with localcontext() as ctx:
            ctx.prec = precision
            value = value.quantize(Decimal(1).scaleb(-places))
        return format(value, "f")

    if value == 0:
        return "0"
    normalized = value.normalize()
    plain = format(normalized, "f")
    if len(plain) <= 200:
        return plain.rstrip("0").rstrip(".") if "." in plain else plain
    return format(normalized, "E")


def execute(expression: str, round_to: int | None = None, **kwargs) -> dict:
    normalized_expression, changes = _normalize_expression(expression)
    if not normalized_expression:
        return {
            "ok": False,
            "error": "表达式为空",
            "expression": normalized_expression,
        }

    try:
        normalized_round_to = None if round_to is None else int(round_to)
        value, approximate = _evaluate(normalized_expression)
        result = _format_decimal(value, normalized_round_to)
    except (CalculatorError, ValueError, InvalidOperation) as exc:
        return {
            "ok": False,
            "error": str(exc),
            "expression": normalized_expression,
        }

    payload: dict[str, Any] = {
        "ok": True,
        "expression": normalized_expression,
        "result": result,
        "approximate": approximate,
    }
    if changes:
        payload["normalization"] = changes
    return payload
