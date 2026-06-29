"""WebUI password protection and first-run setup helpers."""

from __future__ import annotations

import hmac
import os
import secrets
import hashlib
from copy import deepcopy
from datetime import timedelta

from quart import Blueprint, abort, jsonify, redirect, render_template, request, session

import app_state
from config_loader import save_config


auth_bp = Blueprint("webui_auth", __name__)

_SESSION_KEY = "webui_authenticated"
_HASH_PREFIX = "pbkdf2_sha256"
_DEFAULT_SESSION_DAYS = 7


def default_auth_config(raw: object | None = None) -> dict:
    cfg = dict(raw or {}) if isinstance(raw, dict) else {}
    return {
        "enabled": bool(cfg.get("enabled", False)),
        "password_hash": str(cfg.get("password_hash") or ""),
        "skipped_setup": bool(cfg.get("skipped_setup", False)),
        "session_days": int(cfg.get("session_days") or _DEFAULT_SESSION_DAYS),
    }


def auth_config() -> dict:
    return default_auth_config(getattr(app_state, "config", {}).get("webui_auth"))


def session_secret(config: dict) -> str:
    env_secret = os.environ.get("AICQ_WEBUI_SESSION_SECRET", "").strip()
    if env_secret:
        return env_secret
    auth = default_auth_config(config.get("webui_auth"))
    if auth["password_hash"]:
        seed = auth["password_hash"]
    else:
        seed = os.path.abspath(os.getcwd())
    return hashlib.sha256(("aicq-webui-session:" + seed).encode("utf-8")).hexdigest()


def _hash_password(password: str) -> str:
    salt = secrets.token_hex(16)
    rounds = 260_000
    digest = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt.encode("ascii"),
        rounds,
    ).hex()
    return f"{_HASH_PREFIX}${rounds}${salt}${digest}"


def _verify_password(password: str, stored_hash: str) -> bool:
    try:
        prefix, rounds_raw, salt, digest = stored_hash.split("$", 3)
        if prefix != _HASH_PREFIX:
            return False
        rounds = int(rounds_raw)
        candidate = hashlib.pbkdf2_hmac(
            "sha256",
            password.encode("utf-8"),
            salt.encode("ascii"),
            rounds,
        ).hex()
        return hmac.compare_digest(candidate, digest)
    except Exception:
        return False


def _save_auth_config(next_auth: dict) -> None:
    new_cfg = deepcopy(app_state.config)
    new_cfg["webui_auth"] = default_auth_config(next_auth)
    save_config(new_cfg)
    app_state.config = new_cfg


def _is_local_host(host: str) -> bool:
    hostname = host.split(":", 1)[0].strip().lower()
    return hostname in {"127.0.0.1", "localhost", "::1", "[::1]"}


def _auth_status_payload() -> dict:
    cfg = auth_config()
    setup_required = not cfg["enabled"] and not cfg["password_hash"] and not cfg["skipped_setup"]
    host = request.host or ""
    return {
        "enabled": cfg["enabled"],
        "authenticated": bool(session.get(_SESSION_KEY)),
        "setup_required": setup_required,
        "skipped_setup": cfg["skipped_setup"],
        "session_days": cfg["session_days"],
        "external_access_hint": not _is_local_host(host),
    }


def _is_exempt_path(path: str) -> bool:
    if path.startswith("/static/"):
        return True
    return path in {
        "/login",
        "/api/auth/status",
        "/api/auth/login",
        "/api/auth/setup",
        "/api/auth/logout",
    }


def _wants_json(path: str) -> bool:
    if path.startswith("/api/") or path.startswith("/settings/") or path.startswith("/debug/api/"):
        return True
    accept = request.headers.get("Accept", "")
    return "application/json" in accept


async def require_webui_auth():
    path = request.path or "/"
    if _is_exempt_path(path):
        return None
    cfg = auth_config()
    if not cfg["enabled"] or not cfg["password_hash"]:
        return None
    if session.get(_SESSION_KEY):
        return None
    if _wants_json(path):
        return jsonify({"success": False, "error": "需要登录 WebUI"}), 401
    return redirect(f"/login?next={path}")


async def require_webui_auth_ws():
    cfg = auth_config()
    if not cfg["enabled"] or not cfg["password_hash"] or session.get(_SESSION_KEY):
        return None
    abort(401)


def install_auth(app) -> None:
    app.secret_key = session_secret(getattr(app_state, "config", {}))
    app.permanent_session_lifetime = timedelta(days=auth_config()["session_days"])
    app.before_request(require_webui_auth)
    if hasattr(app, "before_websocket"):
        app.before_websocket(require_webui_auth_ws)


@auth_bp.route("/login")
async def login_page():
    return await render_template("login.html")


@auth_bp.route("/api/auth/status")
async def auth_status():
    return jsonify(_auth_status_payload())


@auth_bp.route("/api/auth/login", methods=["POST"])
async def auth_login():
    data = await request.get_json(silent=True) or {}
    password = str(data.get("password") or "")
    cfg = auth_config()
    if not cfg["enabled"] or not cfg["password_hash"]:
        session[_SESSION_KEY] = True
        return jsonify({"success": True})
    if not _verify_password(password, cfg["password_hash"]):
        return jsonify({"success": False, "error": "密码不正确"}), 401
    session.permanent = True
    session[_SESSION_KEY] = True
    return jsonify({"success": True})


@auth_bp.route("/api/auth/logout", methods=["POST"])
async def auth_logout():
    session.pop(_SESSION_KEY, None)
    return jsonify({"success": True})


@auth_bp.route("/api/auth/setup", methods=["POST"])
async def auth_setup():
    data = await request.get_json(silent=True) or {}
    cfg = auth_config()
    if cfg["enabled"] and cfg["password_hash"] and not session.get(_SESSION_KEY):
        return jsonify({"success": False, "error": "需要先登录"}), 401
    action = str(data.get("action") or "set")
    if action == "skip":
        _save_auth_config({
            **cfg,
            "enabled": False,
            "password_hash": "",
            "skipped_setup": True,
        })
        session.pop(_SESSION_KEY, None)
        return jsonify({"success": True, "skipped": True})

    password = str(data.get("password") or "")
    if len(password) < 6:
        return jsonify({"success": False, "error": "密码至少需要 6 位"}), 400
    _save_auth_config({
        **cfg,
        "enabled": True,
        "password_hash": _hash_password(password),
        "skipped_setup": False,
    })
    session.permanent = True
    session[_SESSION_KEY] = True
    return jsonify({"success": True, "enabled": True})


@auth_bp.route("/api/auth/password", methods=["POST"])
async def auth_password():
    cfg = auth_config()
    if cfg["enabled"] and not session.get(_SESSION_KEY):
        return jsonify({"success": False, "error": "需要先登录"}), 401
    data = await request.get_json(silent=True) or {}
    action = str(data.get("action") or "set")
    if action == "disable":
        _save_auth_config({
            **cfg,
            "enabled": False,
            "password_hash": "",
            "skipped_setup": True,
        })
        session.pop(_SESSION_KEY, None)
        return jsonify({"success": True, "enabled": False})

    password = str(data.get("password") or "")
    if len(password) < 6:
        return jsonify({"success": False, "error": "密码至少需要 6 位"}), 400
    _save_auth_config({
        **cfg,
        "enabled": True,
        "password_hash": _hash_password(password),
        "skipped_setup": False,
    })
    session.permanent = True
    session[_SESSION_KEY] = True
    return jsonify({"success": True, "enabled": True})
