import os
import sys
import unittest
from types import SimpleNamespace

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from llm.core.provider import OpenAICompatAdapter
from tools.specs import ToolCollection, ToolSpec


class _FakeCompletions:
    def __init__(self, response) -> None:
        self.response = response
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return self.response


class _FakeClient:
    def __init__(self, response) -> None:
        self.completions = _FakeCompletions(response)
        self.chat = SimpleNamespace(completions=self.completions)


def _make_text_response(content: str | None):
    return SimpleNamespace(
        usage=None,
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=content, tool_calls=None)
            )
        ],
    )


class ProviderThinkingConfigTests(unittest.TestCase):
    def setUp(self) -> None:
        self.collection = ToolCollection(
            active_specs={
                "noop": ToolSpec(
                    name="noop",
                    declaration={"name": "noop", "parameters": {"type": "object"}},
                    handler=lambda **kwargs: {"ok": True},
                    module_name="tests.noop",
                )
            }
        )

    def _create_mock_adapter(self, response_text="我不需要再想想。") -> tuple[OpenAICompatAdapter, _FakeClient]:
        fake_client = _FakeClient(_make_text_response(response_text))
        adapter = object.__new__(OpenAICompatAdapter)
        adapter.client = fake_client
        adapter.provider = "test"
        adapter.model = "fake-model"
        adapter._vision_enabled = False
        return adapter, fake_client

    def test_enable_thinking_default_is_true(self) -> None:
        adapter, fake_client = self._create_mock_adapter()
        # 默认不传 enable_thinking
        result = adapter.call_one_round(
            lambda active, latent: "system",
            "user",
            {},
            self.collection,
        )
        calls = fake_client.completions.calls
        self.assertEqual(len(calls), 1)
        self.assertIn("extra_body", calls[0])
        self.assertEqual(calls[0]["extra_body"].get("enable_thinking"), True)

    def test_enable_thinking_explicit_true(self) -> None:
        adapter, fake_client = self._create_mock_adapter()
        result = adapter.call_one_round(
            lambda active, latent: "system",
            "user",
            {"enable_thinking": True},
            self.collection,
        )
        calls = fake_client.completions.calls
        self.assertEqual(len(calls), 1)
        self.assertIn("extra_body", calls[0])
        self.assertEqual(calls[0]["extra_body"].get("enable_thinking"), True)

    def test_enable_thinking_explicit_false(self) -> None:
        adapter, fake_client = self._create_mock_adapter()
        result = adapter.call_one_round(
            lambda active, latent: "system",
            "user",
            {"enable_thinking": False},
            self.collection,
        )
        calls = fake_client.completions.calls
        self.assertEqual(len(calls), 1)
        self.assertIn("extra_body", calls[0])
        self.assertEqual(calls[0]["extra_body"].get("enable_thinking"), False)


if __name__ == "__main__":
    unittest.main()
