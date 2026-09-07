from __future__ import annotations

import json
import sqlite3
from concurrent.futures import ThreadPoolExecutor

import pytest

from llm.media import image_migration as migration
from llm.media.image_upgrade import ensure_images_ready, is_verified, upgrade
from test_unified_image_migration import legacy
from test_sticker_collection import png


def test_automatic_upgrade_preserves_old_refs_files_and_backup(tmp_path):
    root = tmp_path / 'installation'
    original, duplicate = legacy(root)
    ensure_images_ready(root)
    assert is_verified(root)
    output = root / 'data/migrations/unified-images-v1'
    assert (output / 'backup/data/stickers/images/first.png').read_bytes() == png()
    assert original.exists() and duplicate.exists()
    with sqlite3.connect(root / 'data/AICQ.db') as db:
        refs = db.execute('SELECT canonical_ref FROM media_refs WHERE ref IN (?,?)',
                          ('chat_old', 'browser_old')).fetchall()
        assert refs == [('sticker_old',), ('sticker_old',)]
    before = (output / 'manifest.json').read_bytes()
    ensure_images_ready(root)
    assert (output / 'manifest.json').read_bytes() == before


def test_empty_install_and_concurrent_startups(tmp_path):
    root = tmp_path / 'new'
    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(ensure_images_ready, [root, root]))
    assert is_verified(root)
    with sqlite3.connect(root / 'data/AICQ.db') as db:
        assert db.execute('SELECT count(*) FROM media_images').fetchone()[0] == 0


def test_conflict_blocks_startup_without_cleanup(tmp_path):
    root = tmp_path / 'conflict'
    original, duplicate = legacy(root)
    duplicate.rename(duplicate.with_name('chat_old.png'))
    duplicate.with_name('chat_old.png').write_bytes(png('blue'))
    with pytest.raises(RuntimeError):
        ensure_images_ready(root)
    assert not is_verified(root)
    assert original.exists()
    manifest = json.loads((root / 'data/migrations/unified-images-v1/manifest.json').read_text())
    assert 'chat_old' in manifest['conflicts']


def test_failed_verification_retries_from_saved_progress(tmp_path, monkeypatch):
    root = tmp_path / 'retry'
    original, duplicate = legacy(root)
    verify = migration.verify
    def fail(*args):
        raise OSError('interrupted verification')
    monkeypatch.setattr(migration, 'verify', fail)
    with pytest.raises(OSError):
        upgrade(root)
    assert not is_verified(root)
    output = root / 'data/migrations/unified-images-v1'
    backup = (output / 'backup/complete.json').read_bytes()
    monkeypatch.setattr(migration, 'verify', verify)
    upgrade(root)
    assert is_verified(root)
    assert (output / 'backup/complete.json').read_bytes() == backup
    assert original.exists() and duplicate.exists()
