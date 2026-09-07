"""Startup gate for the one-time image import; never deletes legacy files."""
from __future__ import annotations

import json
import logging
import sqlite3
import subprocess
import sys
from pathlib import Path


def is_verified(root: Path) -> bool:
    path = root / 'data/AICQ.db'
    if not path.exists():
        return False
    with sqlite3.connect(path.resolve().as_uri() + '?mode=ro', uri=True) as db:
        if not db.execute("SELECT 1 FROM sqlite_master WHERE name='media_state'").fetchone():
            return False
        return db.execute(
            "SELECT 1 FROM media_state WHERE key='legacy_import_verified' AND value='1'"
        ).fetchone() is not None


def upgrade(root: Path) -> None:
    """Worker entry point. Run in a separate process to isolate store configuration."""
    from . import image_migration as migration

    root = root.resolve()
    output = root / 'data/migrations/unified-images-v1'
    output.mkdir(parents=True, exist_ok=True)
    # SQLite releases this lock after crashes as well as normal exit. A second
    # startup waits here, then checks the committed marker again.
    with sqlite3.connect(output / 'startup-lock.sqlite3', timeout=3600) as lock:
        lock.execute('BEGIN EXCLUSIVE')
        if is_verified(root):
            return
        db_path = root / 'data/AICQ.db'
        db_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(db_path):
            pass
        manifest_path = output / 'manifest.json'
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
            if Path(manifest['source_root']).resolve() != root:
                raise ValueError('Migration belongs to another installation')
        else:
            manifest = migration.scan(root)
            migration._json(manifest_path, manifest)
        migration.apply(manifest, root, output)
        migration.verify(manifest, output)


def ensure_images_ready(root: Path) -> None:
    """Complete import before WebUI or bot writers are started. Fail closed."""
    root = root.resolve()
    if is_verified(root):
        return
    logger = logging.getLogger('AICQ')
    output = root / 'data/migrations/unified-images-v1'
    logger.warning('正在自动迁移历史图片；业务将在验证完成后启动。备份与进度：%s', output)
    script = Path(__file__).resolve().parents[3] / 'scripts/migrate_unified_images.py'
    try:
        subprocess.run(
            [sys.executable, '-B', str(script), 'startup', '--root', str(root)],
            check=True,
        )
        if not is_verified(root):
            raise RuntimeError('Image migration did not publish a verified marker')
    except Exception as exc:
        raise RuntimeError(
            f'图片自动迁移未完成，已阻止业务启动。备份和进度保留在 {output}；'
            '修复报错后重新启动可续跑，旧图片未自动清理。'
        ) from exc
    logger.info('历史图片迁移验证完成；旧文件和独立备份已保留：%s', output)
