"""Standalone output requirements for the main model's trailing context."""


COGNITION_LANGUAGES = {
    "auto": ("自动", ""),
    "zh-CN": ("简体中文", "Simplified Chinese"),
    "zh-TW": ("繁體中文", "Traditional Chinese"),
    "en": ("English", "English"),
    "ja": ("日本語", "Japanese"),
    "ko": ("한국어", "Korean"),
    "es": ("Español", "Spanish"),
    "fr": ("Français", "French"),
    "de": ("Deutsch", "German"),
}

OUTPUT_REQUIREMENTS_PROMPT = """
# Output requirements

## Expression

{cognition_expression}Keep `<motive>` brief and state the reason for your next action.{language_instruction}

## Response structure

Output the following sections in this order:

{cognition_section}<motive>
...your motive...
</motive>
<action>
...one or more <tool_call> blocks in execution order...
</action>

Each `<tool_call>` must contain a strict JSON object. Its namespace, tool name, and arguments must follow the tool declarations in `<tools>`."""


def normalize_output_requirements_config(value: object, *, strict: bool = False) -> dict[str, str]:
    """Accept supported language codes, retaining automatic behavior for old configs."""
    if not isinstance(value, dict):
        if strict:
            raise ValueError("output_requirements 必须是对象")
        value = {}
    language = value.get("cognition_language", "auto")
    language = language.strip() if isinstance(language, str) else ""
    if language not in COGNITION_LANGUAGES:
        if strict:
            raise ValueError("不支持的认知语言")
        language = "auto"
    return {"cognition_language": language}


def build_output_requirements_prompt(
    *,
    native_reasoning_as_cognition: bool = False,
    cognition_language: str = "auto",
) -> str:
    """Build the terminal instructions with a blank line before their heading."""
    language = normalize_output_requirements_config(
        {"cognition_language": cognition_language}
    )["cognition_language"]
    language_name = COGNITION_LANGUAGES[language][1]
    language_instruction = ""
    if language_name:
        target = "internal reasoning" if native_reasoning_as_cognition else "`<cognition>`"
        language_instruction = (
            f"\n\nUse {language_name} by default for {target}, unless an explicit instruction "
            "requires another language. Preserve the original language of quotations, "
            "code, and proper names where appropriate."
        )
    return OUTPUT_REQUIREMENTS_PROMPT.format(
        cognition_expression=(
            "" if native_reasoning_as_cognition
            else "Write `<cognition>` as flowing, unstructured natural language.\n"
        ),
        cognition_section=(
            "" if native_reasoning_as_cognition
            else "<cognition>\n...your cognition...\n</cognition>\n"
        ),
        language_instruction=language_instruction,
    )
