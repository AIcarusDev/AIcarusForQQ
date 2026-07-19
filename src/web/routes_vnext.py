"""Same-origin entry point for the production WebUI vNext bundle."""

from __future__ import annotations

from pathlib import Path

from quart import Blueprint, jsonify, redirect, send_from_directory


vnext_bp = Blueprint("vnext", __name__)
VNEXT_DIST_DIR = Path(__file__).resolve().parents[1] / "static" / "new"


@vnext_bp.route("/new")
async def vnext_redirect():
    return redirect("/new/", code=308)


@vnext_bp.route("/new/")
async def vnext_index():
    index_path = VNEXT_DIST_DIR / "index.html"
    if not index_path.is_file():
        return jsonify({
            "ok": False,
            "error": "vnext_build_missing",
            "message": "WebUI vNext 尚未构建。",
        }), 503
    return await send_from_directory(VNEXT_DIST_DIR, "index.html")


__all__ = ["vnext_bp"]
