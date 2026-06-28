from types import SimpleNamespace

from llm.core.transport import (
    aggregate_chat_completion_stream,
    aggregate_chat_completion_stream_with_callbacks,
    create_streamed_chat_completion,
)


def _chunk(*, content=None, finish_reason=None, tool_calls=None, usage=None):
    choices = []
    if content is not None or finish_reason is not None or tool_calls is not None:
        choices.append(
            SimpleNamespace(
                delta=SimpleNamespace(content=content, tool_calls=tool_calls),
                finish_reason=finish_reason,
            )
        )
    return SimpleNamespace(
        id="chatcmpl-test",
        created=123,
        model="test-model",
        choices=choices,
        usage=usage,
    )


def test_aggregate_chat_completion_stream_collects_content_usage_and_finish_reason():
    usage = {
        "prompt_tokens": 3,
        "completion_tokens": 2,
        "total_tokens": 5,
    }

    response = aggregate_chat_completion_stream(
        [
            _chunk(content="你"),
            _chunk(content="好"),
            _chunk(finish_reason="stop"),
            _chunk(usage=usage),
        ]
    )

    assert response.choices
    assert response.choices[0].message.content == "你好"
    assert response.choices[0].finish_reason == "stop"
    assert response.usage == usage


def test_aggregate_chat_completion_stream_observer_errors_are_non_fatal():
    def broken_observer(_content):
        raise RuntimeError("observer failed")

    response = aggregate_chat_completion_stream_with_callbacks(
        [
            _chunk(content="你"),
            _chunk(content="好"),
            _chunk(finish_reason="stop"),
        ],
        on_text_delta=broken_observer,
    )

    assert response.choices[0].message.content == "你好"
    assert response.choices[0].finish_reason == "stop"


def test_aggregate_chat_completion_stream_propagates_marked_abort():
    class AbortStream(RuntimeError):
        stream_abort = True

    def aborting_observer(_content):
        raise AbortStream("stop streaming")

    try:
        aggregate_chat_completion_stream_with_callbacks(
            [
                _chunk(content="你"),
                _chunk(content="好"),
            ],
            on_text_delta=aborting_observer,
        )
    except AbortStream as exc:
        assert str(exc) == "stop streaming"
    else:
        raise AssertionError("marked stream abort was swallowed")


def test_aggregate_chat_completion_stream_collects_tool_call_chunks():
    response = aggregate_chat_completion_stream(
        [
            _chunk(
                tool_calls=[
                    SimpleNamespace(
                        index=0,
                        id="call_abc",
                        type="function",
                        function=SimpleNamespace(
                            name="memory_",
                            arguments='{"summary"',
                        ),
                    )
                ]
            ),
            _chunk(
                tool_calls=[
                    SimpleNamespace(
                        index=0,
                        id=None,
                        type=None,
                        function=SimpleNamespace(
                            name="write",
                            arguments=':"ok"}',
                        ),
                    )
                ],
                finish_reason="tool_calls",
            ),
        ]
    )

    tool_call = response.choices[0].message.tool_calls[0]
    assert tool_call.id == "call_abc"
    assert tool_call.type == "function"
    assert tool_call.function.name == "memory_write"
    assert tool_call.function.arguments == '{"summary":"ok"}'
    assert response.choices[0].finish_reason == "tool_calls"


def test_create_streamed_chat_completion_requests_streaming_with_usage():
    class FakeCompletions:
        def __init__(self):
            self.calls = []

        def create(self, **kwargs):
            self.calls.append(kwargs)
            return [_chunk(content="ok", finish_reason="stop")]

    completions = FakeCompletions()
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    messages = [{"role": "user", "content": "hello"}]

    response = create_streamed_chat_completion(
        client,
        provider="test",
        all_messages=messages,
        create_kwargs={"model": "test-model"},
    )

    assert response.choices[0].message.content == "ok"
    assert completions.calls == [
        {
            "messages": messages,
            "model": "test-model",
            "stream": True,
            "stream_options": {"include_usage": True},
        }
    ]


def test_create_streamed_chat_completion_accepts_non_stream_response():
    message = SimpleNamespace(content="ok", tool_calls=[])
    choice = SimpleNamespace(message=message, finish_reason="stop")
    whole_response = SimpleNamespace(choices=[choice], usage=None)

    class FakeCompletions:
        def create(self, **kwargs):
            assert kwargs["stream"] is True
            return whole_response

    client = SimpleNamespace(
        chat=SimpleNamespace(completions=FakeCompletions())
    )

    response = create_streamed_chat_completion(
        client,
        provider="test",
        all_messages=[{"role": "user", "content": "hello"}],
        create_kwargs={"model": "test-model"},
    )

    assert response is whole_response
