"""Explicit, auditable cache maintenance operations."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import threading
from typing import Any


@dataclass(frozen=True)
class CacheTarget:
    id: str
    label: str
    relative_path: str
    description: str
    confirmation: str


CACHE_TARGETS: tuple[CacheTarget, ...] = (
    CacheTarget(
        id="image",
        label="图片缓存",
        relative_path="image",
        description="视觉理解与消息图片的可再生成缓存。",
        confirmation="CLEAR IMAGE CACHE",
    ),
    CacheTarget(
        id="tts",
        label="语音缓存",
        relative_path="tts",
        description="语音合成产生的可再生成音频缓存。",
        confirmation="CLEAR TTS CACHE",
    ),
    CacheTarget(
        id="stickers",
        label="表情缓存",
        relative_path="stickers",
        description="表情资源的派生缓存，不包含原始表情收藏记录。",
        confirmation="CLEAR STICKER CACHE",
    ),
)


class CacheMaintenanceError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        status_code: int = 400,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.details = details or {}


class CacheMaintenanceService:
    def __init__(self, cache_root: str | Path | None = None) -> None:
        repository_root = Path(__file__).resolve().parents[2]
        self.cache_root = Path(cache_root) if cache_root is not None else repository_root / "cache"
        self.targets = {target.id: target for target in CACHE_TARGETS}
        self._lock = threading.RLock()

    @property
    def paths(self) -> dict[str, Path]:
        return {
            target.id: self.cache_root / target.relative_path
            for target in CACHE_TARGETS
        }

    def expected_confirmation(self, target_id: str) -> str:
        return self._target(target_id).confirmation

    def describe_actions(
        self,
        *,
        overview: dict[str, dict[str, Any]] | None = None,
    ) -> list[dict[str, Any]]:
        overview = overview or self.overview()
        return [
            {
                "id": target.id,
                "label": f"清理{target.label}",
                "domain": "cache",
                "danger": "medium",
                "available": overview[target.id]["files"] > 0,
                "disabled_reason": "" if overview[target.id]["files"] > 0 else "当前缓存为空",
                "confirmation": target.confirmation,
                "expected_confirmation": target.confirmation,
                "confirmation_required": True,
                "target": target.label,
                "summary": target.description,
                "effects": [
                    f"删除 {target.label}目录中的 {overview[target.id]['files']} 个缓存文件",
                    "后续使用相关能力时，缓存可能重新生成",
                ],
                "preserves": ["配置", "数据库记录", "原始媒体资源"],
                "keeps": "保留配置、数据库记录与原始媒体资源。",
                "backup": {
                    "created": False,
                    "kind": "none",
                    "description": "缓存可重新生成，执行前不会创建备份。",
                },
                "metrics": overview[target.id],
            }
            for target in CACHE_TARGETS
        ]

    def overview(self) -> dict[str, dict[str, Any]]:
        return {
            target.id: {
                "label": target.label,
                "path": str(self._path(target)),
                **self._scan(self._path(target)),
            }
            for target in CACHE_TARGETS
        }

    def perform(self, target_id: str, *, confirmation: str) -> dict[str, Any]:
        target = self._target(target_id)
        if confirmation != target.confirmation:
            raise CacheMaintenanceError(
                "确认字符串不匹配",
                details={"expected_confirmation": target.confirmation},
            )
        with self._lock:
            before = self._scan(self._path(target))
            if before["files"] <= 0:
                raise CacheMaintenanceError(
                    "当前缓存为空",
                    status_code=409,
                    details={"target": target.id, "metrics": before},
                )
            return self._clear_target(target)

    def clear_target(self, target_id: str) -> dict[str, Any]:
        target = self._target(target_id)
        with self._lock:
            return self._clear_target(target)

    def _clear_target(self, target: CacheTarget) -> dict[str, Any]:
        path = self._path(target)
        before = self._scan(path)
        deleted = 0
        failures: list[str] = []
        if path.exists():
            for candidate in path.rglob("*"):
                if not candidate.is_file():
                    continue
                try:
                    candidate.unlink()
                    deleted += 1
                except OSError:
                    failures.append(candidate.name)
        after = self._scan(path)
        result = {
            "ok": not failures,
            "action": target.id,
            "message": f"{target.label}已清理" if not failures else f"{target.label}仅完成部分清理",
            "target": target.label,
            "deleted_files": deleted,
            "reclaimed_bytes": max(0, before["bytes"] - after["bytes"]),
            "remaining": after,
            "backup_created": False,
        }
        if failures:
            raise CacheMaintenanceError(
                f"有 {len(failures)} 个缓存文件无法删除",
                status_code=500,
                details={**result, "failed_files": len(failures)},
            )
        return result

    def _target(self, target_id: str) -> CacheTarget:
        target = self.targets.get(str(target_id or "").strip().lower())
        if target is None:
            raise CacheMaintenanceError(f"未知缓存目标: {target_id}", status_code=404)
        return target

    def _path(self, target: CacheTarget) -> Path:
        root = self.cache_root.resolve()
        path = (self.cache_root / target.relative_path).resolve()
        if path != root and root not in path.parents:
            raise CacheMaintenanceError("缓存路径超出受管目录", status_code=500)
        return path

    @staticmethod
    def _scan(path: Path) -> dict[str, int]:
        size = 0
        files = 0
        if not path.exists():
            return {"bytes": 0, "files": 0}
        for candidate in path.rglob("*"):
            if not candidate.is_file():
                continue
            try:
                size += candidate.stat().st_size
                files += 1
            except OSError:
                continue
        return {"bytes": size, "files": files}


cache_maintenance_service = CacheMaintenanceService()


__all__ = [
    "CACHE_TARGETS",
    "CacheMaintenanceError",
    "CacheMaintenanceService",
    "cache_maintenance_service",
]
