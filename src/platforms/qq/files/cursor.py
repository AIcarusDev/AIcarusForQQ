"""Authenticated stateless cursors for QQ file reads and pagination."""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
from pathlib import Path
from typing import Any


class CursorError(ValueError):
    pass


class CursorCodec:
    def __init__(self, project_root: Path) -> None:
        self.key_path = project_root / "data" / "qq_file_cursor.key"
        self._key: bytes | None = None

    def _load_key(self) -> bytes:
        if self._key is not None:
            return self._key
        self.key_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            key = self.key_path.read_bytes()
        except FileNotFoundError:
            key = os.urandom(32)
            try:
                descriptor = os.open(self.key_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            except FileExistsError:
                key = self.key_path.read_bytes()
            else:
                with os.fdopen(descriptor, "wb") as handle:
                    handle.write(key)
                    handle.flush()
                    os.fsync(handle.fileno())
        if len(key) < 32:
            raise CursorError("游标签名密钥不可用")
        self._key = key
        return key

    @staticmethod
    def _encode(value: bytes) -> str:
        return base64.urlsafe_b64encode(value).decode("ascii").rstrip("=")

    @staticmethod
    def _decode(value: str) -> bytes:
        return base64.urlsafe_b64decode(value + "=" * (-len(value) % 4))

    def dumps(self, kind: str, state: dict[str, Any]) -> str:
        body = json.dumps(
            {"v": 1, "kind": kind, "state": state},
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        signature = hmac.new(self._load_key(), body, hashlib.sha256).digest()
        return f"qfc_{self._encode(body)}.{self._encode(signature)}"

    def loads(self, token: object, kind: str) -> dict[str, Any]:
        raw = str(token or "")
        if not raw.startswith("qfc_") or "." not in raw or len(raw) > 2048:
            raise CursorError("游标无效或已损坏")
        body_part, signature_part = raw[4:].split(".", 1)
        try:
            body = self._decode(body_part)
            signature = self._decode(signature_part)
        except Exception as exc:
            raise CursorError("游标无效或已损坏") from exc
        expected = hmac.new(self._load_key(), body, hashlib.sha256).digest()
        if not hmac.compare_digest(signature, expected):
            raise CursorError("游标无效或已损坏")
        try:
            payload = json.loads(body.decode("utf-8"))
        except (UnicodeError, json.JSONDecodeError) as exc:
            raise CursorError("游标无效或已损坏") from exc
        if not isinstance(payload, dict) or payload.get("v") != 1 or payload.get("kind") != kind:
            raise CursorError("游标不属于此操作")
        state = payload.get("state")
        if not isinstance(state, dict):
            raise CursorError("游标无效或已损坏")
        return state
