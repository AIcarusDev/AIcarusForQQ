"""Runtime registry for platform integrations."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class PlatformRegistry:
    runtimes: dict[str, Any] = field(default_factory=dict)

    def register(self, runtime: Any) -> None:
        platform = str(getattr(runtime, "platform", "") or "").strip()
        if not platform:
            raise ValueError("platform runtime must expose a non-empty platform id")
        self.runtimes[platform] = runtime

    def get(self, platform: str) -> Any | None:
        return self.runtimes.get(str(platform or "").strip())

    def require(self, platform: str) -> Any:
        runtime = self.get(platform)
        if runtime is None:
            raise KeyError(f"platform {platform!r} is not registered")
        return runtime

    def status_payload(self) -> dict[str, dict[str, Any]]:
        payload: dict[str, dict[str, Any]] = {}
        for name, runtime in self.runtimes.items():
            account = getattr(runtime, "account", None)
            payload[name] = {
                "enabled": bool(getattr(runtime, "enabled", False)),
                "connected": bool(getattr(runtime, "connected", False)),
                "state": str(getattr(runtime, "state", "connecting") or "connecting"),
                "account_id": getattr(account, "account_id", ""),
                "account_name": getattr(account, "account_name", ""),
            }
        return payload


def get_platform(platform: str) -> Any | None:
    import app_state

    registry = getattr(app_state, "platform_registry", None)
    if registry is None:
        return None
    return registry.get(platform)
