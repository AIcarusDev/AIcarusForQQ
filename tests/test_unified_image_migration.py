from __future__ import annotations

import base64
import hashlib
import json
import sqlite3
from pathlib import Path

import pytest

from llm.media import image_migration as migration
from llm.media.image_store import read_image
from test_sticker_collection import png


def legacy(root):
    (root / 'data/stickers/images').mkdir(parents=True)
    raw = png()
    digest = hashlib.sha256(raw).hexdigest()
    original = root / 'data/stickers/images/first.png'
    original.write_bytes(raw)
    duplicate = root / 'cache/browser_image/browser_old.png'
    duplicate.parent.mkdir(parents=True)
    duplicate.write_bytes(raw)
    document = {'version': 3, 'ref_hashes': {'sticker_old': digest}, 'stickers': {
        'sticker_old': {'filename': 'first.png', 'mime': 'image/png', 'sha256': digest,
            'description': 'collection use', 'created_at': '2020-01-01', 'aliases': []}}}
    (root / 'data/stickers/index.json').write_text(json.dumps(document))
    with sqlite3.connect(root / 'data/AICQ.db') as db:
        db.execute('CREATE TABLE chat_messages(id INTEGER PRIMARY KEY,session_key TEXT,message_id TEXT,images TEXT,content_segments TEXT,created_at INTEGER)')
        db.execute('INSERT INTO chat_messages VALUES (1,?,?,?,?,?)', ('chat','1',
            json.dumps({'chat_old': {'base64': base64.b64encode(raw).decode(), 'description': 'image description'}}),
            json.dumps([{'type':'image','image_ref':'chat_old'},{'type':'image','image_ref':'chat_old'}]), 1))
    return original, duplicate


def test_migration_preserves_aliases_metadata_and_cleans_only_backed_up_copies(tmp_path):
    root, output = tmp_path / 'root', tmp_path / 'migration'
    original, duplicate = legacy(root)
    manifest = migration.scan(root)
    migration.apply(manifest, root, output)
    migration.verify(manifest, output)
    assert read_image('chat_old')['image_ref'] == 'sticker_old'
    assert read_image('browser_old')['data'] == png()
    assert read_image('sticker_old')['description'] == 'image description'
    migration.cleanup(manifest, output)
    assert not original.exists() and not duplicate.exists()
    assert (output / 'backup/data/stickers/images/first.png').read_bytes() == png()
    assert read_image('chat_old')['data'] == png()
    migration.cleanup(manifest, output)
    assert read_image('browser_old')['data'] == png()
    with sqlite3.connect(root / 'data/AICQ.db') as db:
        row = db.execute('SELECT images,content_segments FROM chat_messages').fetchone()
    assert 'base64' not in row[0]
    assert [i['image_ref'] for i in json.loads(row[1])] == ['sticker_old', 'sticker_old']


def test_migration_conflict_never_rebinds_or_cleans(tmp_path):
    root, output = tmp_path / 'root', tmp_path / 'migration'
    original, duplicate = legacy(root)
    duplicate.rename(duplicate.with_name('chat_old.png'))
    duplicate = duplicate.with_name('chat_old.png')
    duplicate.write_bytes(png('blue'))
    manifest = migration.scan(root)
    assert manifest['conflicts']
    with pytest.raises(ValueError, match='Conflicting'):
        migration.apply(manifest, root, output)
    assert original.exists() and duplicate.exists()


def test_cleanup_refuses_changed_source_and_keeps_original(tmp_path):
    root, output = tmp_path / 'root', tmp_path / 'migration'
    _, duplicate = legacy(root)
    manifest = migration.scan(root)
    migration.apply(manifest, root, output)
    migration.verify(manifest, output)
    duplicate.write_bytes(png('blue'))
    with pytest.raises(ValueError, match='Source changed'):
        migration.cleanup(manifest, output)
    assert duplicate.read_bytes() == png('blue')
    assert read_image('chat_old')['data'] == png()


def test_rehearsal_does_not_delete_source(tmp_path):
    root, output, target = tmp_path/'root', tmp_path/'migration', tmp_path/'target'
    original, duplicate = legacy(root)
    manifest = migration.scan(root)
    migration.apply(manifest, target, output)
    migration.verify(manifest, output)
    migration.cleanup(manifest, output)
    assert original.exists() and duplicate.exists()
    assert Path(read_image('chat_old')['locator']).is_relative_to(target)


def test_interrupted_apply_is_idempotent_and_missing_data_is_retained(tmp_path, monkeypatch):
    root, output = tmp_path/'root', tmp_path/'migration'
    legacy(root)
    (root/'cache/browser_image/second.png').write_bytes(png('blue'))
    with sqlite3.connect(root/'data/AICQ.db') as db:
        db.execute('INSERT INTO chat_messages VALUES (2,?,?,?,?,?)', ('chat','2',json.dumps({'missing_ref':{'expired':True}}),'[]',2))
    manifest = migration.scan(root)
    save = migration._json
    def interrupt(path, value):
        if value.get('phase') == 'registering':
            raise OSError('interrupted')
        save(path,value)
    with monkeypatch.context() as patch:
        patch.setattr(migration,'_json',interrupt)
        with pytest.raises(OSError):
            migration.apply(manifest, root, output)
    migration.apply(manifest, root, output)
    migration.verify(manifest, output)
    before = dict(manifest['mapping'])
    migration.apply(manifest, root, output)
    migration.verify(manifest, output)
    assert manifest['mapping'] == before
    assert read_image('missing_ref').get('unavailable_status')


def test_manifest_replace_retries_temporary_windows_file_lock(tmp_path, monkeypatch):
    destination = tmp_path/'manifest.json'
    destination.write_text('{"phase":"old"}')
    replace = Path.replace
    attempts = []
    def locked(path, target):
        attempts.append(True)
        if len(attempts) < 3:
            raise PermissionError('temporary reader lock')
        return replace(path,target)
    monkeypatch.setattr(Path, 'replace', locked)
    migration._json(destination, {'phase':'new'})
    assert json.loads(destination.read_text()) == {'phase':'new'}
    assert len(attempts) == 3


def test_verified_store_does_not_depend_on_collection_metadata(tmp_path):
    from llm.media.image_store import register_image
    root, output = tmp_path/'root', tmp_path/'migration'
    legacy(root)
    manifest = migration.scan(root)
    migration.apply(manifest, root, output)
    migration.verify(manifest, output)
    (root/'data/stickers/index.json').write_text('{broken collection metadata')
    ref = register_image(png('blue'), 'chat')['image_ref']
    assert read_image(ref)['data'] == png('blue')
