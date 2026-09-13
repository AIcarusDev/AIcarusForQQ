"""Standalone output requirements for the main model's trailing context."""


OUTPUT_REQUIREMENTS_PROMPT = """
# Output requirements

## Expression

Write `<cognition>` as flowing, unstructured natural language.
Keep `<motive>` brief and state the reason for your next action.

## Response structure

Output the following three sections in this order:

<cognition>
...your cognition...
</cognition>
<motive>
...your motive...
</motive>
<action>
...one or more <tool_call> blocks in execution order...
</action>

Each `<tool_call>` must contain a strict JSON object. Its namespace, tool name, and arguments must follow the tool declarations in `<tools>`."""


def build_output_requirements_prompt() -> str:
    """Build the terminal instructions with a blank line before their heading."""
    return OUTPUT_REQUIREMENTS_PROMPT
