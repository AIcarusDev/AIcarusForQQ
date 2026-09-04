"""Incrementally probe OpenAI-compatible Chat Completions or Responses APIs.

The probe reads the selected model/provider from ``config/config_user.yaml``
and its API key from the provider's configured environment variable. It never
prints the key or the Authorization header.

Examples:
    python scripts/probe_chat_completions.py
    python scripts/probe_chat_completions.py --api responses --model gpt-5.6-luna
    python scripts/probe_chat_completions.py --stage minimal,max_tokens,max_completion_tokens
    python scripts/probe_chat_completions.py --stage thinking_baseline,enable_thinking_true,enable_thinking_false,thinking_enabled,thinking_disabled,reasoning_effort_low,reasoning_effort_none --continue-on-error
    python scripts/probe_chat_completions.py --continue-on-error
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
import yaml
from dotenv import load_dotenv


ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "config" / "config_user.yaml"


@dataclass(frozen=True)
class ProbeCase:
    name: str
    description: str
    payload: dict[str, Any]


def _load_target() -> tuple[str, str, str]:
    load_dotenv(ROOT / ".env", override=False)
    cfg = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8-sig")) or {}
    provider_id = str(cfg.get("provider") or "").strip()
    provider = (cfg.get("model_providers") or {}).get(provider_id) or {}
    model = str(cfg.get("model") or "").strip()
    base_url = str(provider.get("base_url") or "").strip().rstrip("/")
    key_env = str(provider.get("api_key_env") or "").strip()

    if not provider_id or not model or not base_url or not key_env:
        raise ValueError("selected provider/model configuration is incomplete")
    api_key = os.environ.get(key_env, "").strip()
    if not api_key:
        raise ValueError(f"API key environment variable is empty: {key_env}")
    return base_url, model, api_key


def _chat_cases(model: str) -> list[ProbeCase]:
    user_only = [{"role": "user", "content": "Reply with exactly: OK"}]
    with_system = [
        {"role": "system", "content": "Follow the user's instruction exactly."},
        *user_only,
    ]
    multi_turn = [
        {"role": "system", "content": "Reply briefly."},
        {"role": "user", "content": "Reply with exactly: ONE"},
        {"role": "assistant", "content": "ONE"},
        {"role": "user", "content": "Reply with exactly: OK"},
    ]

    def payload(messages: list[dict[str, str]], **kwargs: Any) -> dict[str, Any]:
        return {"model": model, "messages": messages, **kwargs}

    return [
        ProbeCase("minimal", "user message only, non-streaming", payload(user_only)),
        ProbeCase(
            "max_tokens",
            "legacy output-limit field",
            payload(user_only, max_tokens=32),
        ),
        ProbeCase(
            "max_completion_tokens",
            "new output-limit field",
            payload(user_only, max_completion_tokens=32),
        ),
        ProbeCase(
            "temperature",
            "temperature only",
            payload(user_only, temperature=1),
        ),
        ProbeCase(
            "thinking_baseline",
            "thinking-control baseline with no thinking field",
            payload(user_only, max_tokens=32),
        ),
        ProbeCase(
            "enable_thinking_true",
            "non-standard enable_thinking=true",
            payload(user_only, max_tokens=32, enable_thinking=True),
        ),
        ProbeCase(
            "enable_thinking_false",
            "non-standard enable_thinking=false",
            payload(user_only, max_tokens=32, enable_thinking=False),
        ),
        ProbeCase(
            "thinking_enabled",
            "DeepSeek-style thinking.type=enabled",
            payload(user_only, max_tokens=32, thinking={"type": "enabled"}),
        ),
        ProbeCase(
            "thinking_disabled",
            "DeepSeek-style thinking.type=disabled",
            payload(user_only, max_tokens=32, thinking={"type": "disabled"}),
        ),
        ProbeCase(
            "reasoning_effort_low",
            "OpenAI-style reasoning_effort=low",
            payload(user_only, max_tokens=32, reasoning_effort="low"),
        ),
        ProbeCase(
            "reasoning_effort_none",
            "OpenAI-style reasoning_effort=none",
            payload(user_only, max_tokens=32, reasoning_effort="none"),
        ),
        ProbeCase("system", "system + user messages", payload(with_system)),
        ProbeCase("multi_turn", "assistant message in history", payload(multi_turn)),
        ProbeCase(
            "stream",
            "minimal SSE streaming",
            payload(user_only, stream=True),
        ),
        ProbeCase(
            "stream_options",
            "streaming with usage aggregation requested",
            payload(user_only, stream=True, stream_options={"include_usage": True}),
        ),
        ProbeCase(
            "app_like",
            "small request with the same transport fields as AIcarus",
            payload(
                with_system,
                temperature=1,
                max_tokens=32,
                stream=True,
                stream_options={"include_usage": True},
            ),
        ),
        ProbeCase(
            "app_like_new_limit",
            "AIcarus-like request using max_completion_tokens",
            payload(
                with_system,
                temperature=1,
                max_completion_tokens=32,
                stream=True,
                stream_options={"include_usage": True},
            ),
        ),
    ]


def _responses_cases(model: str) -> list[ProbeCase]:
    prompt = "Reply with exactly: OK"
    multi_turn = [
        {"role": "user", "content": "Reply with exactly: ONE"},
        {"role": "assistant", "content": "ONE"},
        {"role": "user", "content": prompt},
    ]

    def payload(input_value: Any, **kwargs: Any) -> dict[str, Any]:
        return {"model": model, "input": input_value, **kwargs}

    return [
        ProbeCase("minimal", "plain text input, non-streaming", payload(prompt)),
        ProbeCase(
            "max_output_tokens",
            "Responses output-limit field",
            payload(prompt, max_output_tokens=128),
        ),
        ProbeCase(
            "instructions",
            "instructions + plain text input",
            payload(prompt, instructions="Follow the user's instruction exactly."),
        ),
        ProbeCase(
            "multi_turn",
            "structured multi-turn input",
            payload(multi_turn),
        ),
        ProbeCase(
            "stream",
            "minimal Responses SSE streaming",
            payload(prompt, stream=True),
        ),
        ProbeCase(
            "app_like",
            "small Responses request matching AIcarus concepts",
            payload(
                multi_turn,
                instructions="Follow the user's instruction exactly.",
                max_output_tokens=128,
                stream=True,
            ),
        ),
    ]


def _compact_json_response(response: httpx.Response) -> str:
    try:
        body = response.json()
    except ValueError:
        return response.text[:1200].replace("\n", "\\n")
    return json.dumps(body, ensure_ascii=False, separators=(",", ":"))[:1200]


def _summarize_chat_success(response: httpx.Response, streamed: bool) -> str:
    if not streamed:
        try:
            body = response.json()
            choice = (body.get("choices") or [{}])[0]
            message = choice.get("message") or {}
            usage = body.get("usage") if isinstance(body.get("usage"), dict) else {}
            return json.dumps(
                {
                    "content": message.get("content"),
                    "reasoning_content": message.get("reasoning_content"),
                    "tool_calls": bool(message.get("tool_calls")),
                    "finish_reason": choice.get("finish_reason"),
                    "usage": usage,
                    "message_fields": sorted(message),
                },
                ensure_ascii=False,
            )
        except (ValueError, AttributeError, IndexError):
            return _compact_json_response(response)

    content_parts: list[str] = []
    finish_reason = None
    chunks = 0
    for line in response.text.splitlines():
        if not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if not data or data == "[DONE]":
            continue
        try:
            chunk = json.loads(data)
        except json.JSONDecodeError:
            continue
        chunks += 1
        choice = (chunk.get("choices") or [{}])[0]
        delta = choice.get("delta") or {}
        if isinstance(delta.get("content"), str):
            content_parts.append(delta["content"])
        if choice.get("finish_reason") is not None:
            finish_reason = choice["finish_reason"]
    return json.dumps(
        {
            "chunks": chunks,
            "content": "".join(content_parts)[:200],
            "finish_reason": finish_reason,
        },
        ensure_ascii=False,
    )


def _responses_output_text(body: dict[str, Any]) -> str:
    parts: list[str] = []
    for item in body.get("output") or []:
        if not isinstance(item, dict) or item.get("type") != "message":
            continue
        for content in item.get("content") or []:
            if not isinstance(content, dict):
                continue
            if content.get("type") in {"output_text", "text"}:
                text = content.get("text")
                if isinstance(text, str):
                    parts.append(text)
    return "".join(parts)


def _summarize_responses_success(response: httpx.Response, streamed: bool) -> str:
    if not streamed:
        try:
            body = response.json()
            return json.dumps(
                {
                    "status": body.get("status"),
                    "content": _responses_output_text(body)[:200],
                    "error": body.get("error"),
                },
                ensure_ascii=False,
            )
        except (ValueError, AttributeError):
            return _compact_json_response(response)

    content_parts: list[str] = []
    event_types: dict[str, int] = {}
    status = None
    for line in response.text.splitlines():
        if not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if not data or data == "[DONE]":
            continue
        try:
            event = json.loads(data)
        except json.JSONDecodeError:
            continue
        event_type = str(event.get("type") or "unknown")
        event_types[event_type] = event_types.get(event_type, 0) + 1
        if event_type == "response.output_text.delta" and isinstance(event.get("delta"), str):
            content_parts.append(event["delta"])
        if event_type in {"response.completed", "response.failed", "response.incomplete"}:
            response_body = event.get("response") or {}
            status = response_body.get("status")
            if not content_parts and isinstance(response_body, dict):
                content_parts.append(_responses_output_text(response_body))
    return json.dumps(
        {
            "events": event_types,
            "content": "".join(content_parts)[:200],
            "status": status,
        },
        ensure_ascii=False,
    )


def _run_case(
    client: httpx.Client,
    endpoint: str,
    case: ProbeCase,
    *,
    api: str,
) -> bool:
    response = client.post(endpoint, json=case.payload)
    request_id = (
        response.headers.get("x-request-id")
        or response.headers.get("request-id")
        or response.headers.get("cf-ray")
        or "-"
    )
    print(f"\n[{case.name}] {case.description}")
    print(f"  status={response.status_code} request_id={request_id}")
    if response.is_success:
        summarizer = (
            _summarize_responses_success if api == "responses" else _summarize_chat_success
        )
        print(f"  response={summarizer(response, bool(case.payload.get('stream')))}")
        return True
    print(f"  response={_compact_json_response(response)}")
    return False


def _probe_models(client: httpx.Client, base_url: str, selected_model: str) -> None:
    response = client.get(f"{base_url}/models")
    request_id = (
        response.headers.get("x-request-id")
        or response.headers.get("request-id")
        or response.headers.get("cf-ray")
        or "-"
    )
    print("\n[models] provider model catalog")
    print(f"  status={response.status_code} request_id={request_id}")
    if not response.is_success:
        print(f"  response={_compact_json_response(response)}")
        return
    try:
        data = response.json().get("data") or []
        model_ids = [str(item.get("id") or "") for item in data if isinstance(item, dict)]
    except (ValueError, AttributeError):
        print(f"  response={_compact_json_response(response)}")
        return
    selected_lower = selected_model.lower()
    related = [
        model_id
        for model_id in model_ids
        if "luna" in model_id.lower()
        or selected_lower in model_id.lower()
        or model_id.lower() in selected_lower
    ]
    print(
        "  result="
        + json.dumps(
            {
                "count": len(model_ids),
                "selected_model_listed": selected_model in model_ids,
                "related_models": related[:20],
            },
            ensure_ascii=False,
        )
    )


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--api",
        choices=("chat", "responses"),
        default="chat",
        help="API family to probe; default: chat",
    )
    parser.add_argument(
        "--stage",
        help="comma-separated stage names; default runs every stage",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="run later stages after a failure",
    )
    parser.add_argument("--model", help="override the configured model for comparison")
    parser.add_argument("--skip-model-check", action="store_true")
    parser.add_argument("--timeout", type=float, default=60.0)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv or sys.argv[1:])
    base_url, configured_model, api_key = _load_target()
    model = str(args.model or configured_model).strip()
    cases = _responses_cases(model) if args.api == "responses" else _chat_cases(model)
    available = {case.name: case for case in cases}
    if args.stage:
        requested = [name.strip() for name in args.stage.split(",") if name.strip()]
        unknown = [name for name in requested if name not in available]
        if unknown:
            raise ValueError(
                f"unknown stages: {', '.join(unknown)}; available: {', '.join(available)}"
            )
        cases = [available[name] for name in requested]

    endpoint_path = "responses" if args.api == "responses" else "chat/completions"
    endpoint = f"{base_url}/{endpoint_path}"
    print(f"endpoint={endpoint}")
    print(f"api={args.api}")
    print(f"model={model}")
    print(f"stages={','.join(case.name for case in cases)}")
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    failures = 0
    with httpx.Client(headers=headers, timeout=args.timeout) as client:
        if not args.skip_model_check:
            try:
                _probe_models(client, base_url, model)
            except httpx.HTTPError as exc:
                print(f"\n[models] transport_error={type(exc).__name__}: {exc}")
        for case in cases:
            try:
                succeeded = _run_case(client, endpoint, case, api=args.api)
            except httpx.HTTPError as exc:
                succeeded = False
                print(f"\n[{case.name}] transport_error={type(exc).__name__}: {exc}")
            if succeeded:
                continue
            failures += 1
            if not args.continue_on_error:
                print("\nStopped at first failure; use --continue-on-error to compare later stages.")
                break
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
