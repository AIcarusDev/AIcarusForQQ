from __future__ import annotations

import pytest

from runtime.cache_maintenance import CacheMaintenanceError, CacheMaintenanceService


def test_cache_actions_expose_exact_scope_and_require_server_confirmation(tmp_path) -> None:
    cache_root = tmp_path / "cache"
    image_cache = cache_root / "image"
    image_cache.mkdir(parents=True)
    (image_cache / "one.png").write_bytes(b"1234")
    (image_cache / "two.png").write_bytes(b"56")
    service = CacheMaintenanceService(cache_root)

    image_action = {item["id"]: item for item in service.describe_actions()}["image"]
    assert image_action["target"] == "图片缓存"
    assert image_action["metrics"]["bytes"] == 6
    assert image_action["metrics"]["files"] == 2
    assert image_action["expected_confirmation"] == "CLEAR IMAGE CACHE"
    assert image_action["backup"]["created"] is False

    with pytest.raises(CacheMaintenanceError):
        service.perform("image", confirmation="CLEAR CACHE")
    assert (image_cache / "one.png").exists()

    result = service.perform("image", confirmation=image_action["expected_confirmation"])
    assert result["ok"] is True
    assert result["deleted_files"] == 2
    assert result["reclaimed_bytes"] == 6
    assert list(image_cache.iterdir()) == []

    empty_action = {item["id"]: item for item in service.describe_actions()}["image"]
    assert empty_action["available"] is False
    with pytest.raises(CacheMaintenanceError) as unavailable:
        service.perform("image", confirmation=empty_action["expected_confirmation"])
    assert unavailable.value.status_code == 409
    assert unavailable.value.details["metrics"] == {"bytes": 0, "files": 0}

