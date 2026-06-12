from .flow import ConsciousnessFlow, ToolCall, ToolResponse, FlowRound


def consciousness_main_loop(*args, **kwargs):
    from .main_loop import consciousness_main_loop as _consciousness_main_loop

    return _consciousness_main_loop(*args, **kwargs)


def trigger_first_activation(*args, **kwargs):
    from .main_loop import trigger_first_activation as _trigger_first_activation

    return _trigger_first_activation(*args, **kwargs)

__all__ = [
    "ConsciousnessFlow",
    "ToolCall",
    "ToolResponse",
    "FlowRound",
    "consciousness_main_loop",
    "trigger_first_activation",
]
