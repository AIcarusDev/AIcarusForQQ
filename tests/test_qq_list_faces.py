"""The native QQ face catalog is paged from a validated local QQ resource."""

from __future__ import annotations

import json

from platforms.qq.adapter.config import normalize_qq_platform_config
from platforms.qq.tools.qq_social import list_faces
from tools import build_tools
from tools.namespaces import load_namespace_registry
from tools.prompt_signatures import build_prompt_signature


def _write_catalog(path, entries):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"sysface": entries}, ensure_ascii=False), encoding="utf-8")


def _config(path=None, transfer=None):
    adapter = {"type": "napcat"}
    if path is not None:
        adapter["face_config_path"] = str(path)
    if transfer is not None:
        adapter["file_transfer"] = {"host_directory": str(transfer)}
    return {"platforms": {"qq": {"enabled": True, "adapter": adapter}}}


def test_list_faces_pages_sorted_ids_and_filters_names(tmp_path):
    catalog = tmp_path / "face_config.json"
    entries = [
        {"QSid": str(face_id), "QDes": f"/表情{face_id}"}
        for face_id in range(30, -1, -1)
    ]
    entries.extend([
        {"QSid": "53", "QDes": "/流泪", "AniStickerType": 1, "AniStickerPackId": "1"},
        {"QSid": "358", "QDes": "/骰子", "AniStickerType": 2,
         "AniStickerPackId": "1", "QHide": "1"},
        {"QSid": "364", "QDes": "/超级赞", "AniStickerType": 1,
         "AniStickerPackId": "2", "QHide": "1"},
    ])
    _write_catalog(catalog, entries)
    config = _config(path=catalog)
    normalize_qq_platform_config(config)
    handler = list_faces.make_handler(config)
    assert list_faces.load_face_descriptions(config)[364] == "/超级赞"

    first = handler(kind="normal")
    assert (first["kind"], first["page"], first["page_size"], first["total"], first["next_page"]) == (
        "normal", 1, 30, 32, 2,
    )
    assert [item["id"] for item in first["faces"]] == list(range(30))
    assert all(set(item) == {"id", "des"} for item in first["faces"])
    assert handler(kind="normal", page=2)["faces"] == [
        {"id": 30, "des": "/表情30"},
        {"id": 53, "des": "/流泪"},
    ]
    assert handler(kind="normal", page=3) == {
        "kind": "normal",
        "page": 3, "page_size": 30, "total": 32,
        "next_page": None, "faces": [],
    }
    assert handler(kind="super", query=" /超级赞 ") == {
        "kind": "super",
        "page": 1, "page_size": 30, "total": 1,
        "next_page": None, "faces": [{"id": 364, "des": "/超级赞"}],
    }
    assert handler(kind="super")["faces"] == [
        {"id": 53, "des": "/流泪"},
        {"id": 358, "des": "/骰子"},
        {"id": 364, "des": "/超级赞"},
    ]
    assert handler(kind="normal", query="流泪")["faces"] == [{"id": 53, "des": "/流泪"}]
    assert handler(kind="normal", query="超级赞")["faces"] == []
    assert handler(kind="normal", query="表情", page=2)["faces"] == [{"id": 30, "des": "/表情30"}]
    assert handler(kind="normal", page=0)["code"] == "invalid_page"
    assert handler(kind="normal", query="/")["code"] == "invalid_query"
    assert handler(kind="all")["code"] == "invalid_kind"


def test_list_faces_uses_napcat_sibling_qq_mount_and_reports_unavailable(tmp_path):
    transfer = tmp_path / "data" / "transfer"
    transfer.mkdir(parents=True)
    catalog = (
        tmp_path / "data" / "qq" / "nt_qq" / "global" / "nt_data"
        / "Emoji" / "emoji-resource" / "face_config.json"
    )
    _write_catalog(catalog, [{"QSid": "14", "QDes": "/微笑"}])
    assert list_faces.make_handler(_config(transfer=transfer))(kind="normal")["faces"] == [
        {"id": 14, "des": "/微笑"},
    ]

    missing = list_faces.make_handler(_config(path=tmp_path / "missing.json"))(kind="normal")
    assert missing == {"error": "QQ 内置表情目录不可用", "code": "face_catalog_unavailable"}

    _write_catalog(catalog, [
        {"QSid": "14", "QDes": "/微笑"},
        {"QSid": "14", "QDes": "/重复"},
    ])
    assert list_faces.make_handler(_config(transfer=transfer))(kind="normal")["code"] == "face_catalog_unavailable"


def test_list_faces_is_one_qq_social_tool_with_ts_signature():
    registry = load_namespace_registry()
    assert registry.namespaces_for_tool("list_faces") == ("qq_social",)

    collection = build_tools(_config(), current_platform="qq")
    spec = collection.get_any("list_faces", namespace="qq_social")
    assert spec is not None
    assert spec.execution.parallel_safe is True
    signature = build_prompt_signature(list_faces.TOOL_CONTRACT.declaration())
    assert "list_faces(args: {" in signature
    assert "kind: \"normal\" | \"super\";" in signature
    assert "kind" in list_faces.ListFacesArgs.model_json_schema()["required"]
    assert "page?: number;" in signature
    assert "query?: string;" in signature
    assert "超级表情" in signature
