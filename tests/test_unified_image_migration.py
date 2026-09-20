from __future__ import annotations

import base64
import hashlib
import json
import sqlite3
import tracemalloc
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


@pytest.mark.parametrize('container', ['object', 'array'])
def test_inline_sources_survive_manifest_reload_without_retaining_image_data(tmp_path, container):
    root, output = tmp_path/'root', tmp_path/'migration'
    legacy(root)
    raw = png('blue')
    info = {'ref': 'inline_only', 'base64': base64.b64encode(raw).decode(),
            'description': 'preserved description', 'examinations': [{'focus': 'detail', 'result': 'blue'}]}
    payload = {'inline_only': info} if container == 'object' else [None, info]
    with sqlite3.connect(root/'data/AICQ.db') as db:
        db.execute('INSERT INTO chat_messages VALUES (2,?,?,?,?,?)',
                   ('chat', '2', json.dumps(payload), '[]', 2))
    manifest = migration.scan(root)
    serialized = json.dumps(manifest)
    assert info['base64'] not in serialized
    assert 'base64' not in serialized
    migration._json(output/'manifest.json', manifest)
    restored = json.loads((output/'manifest.json').read_text())
    migration.apply(restored, root, output)
    migration.verify(restored, output)
    record = read_image('inline_only')
    assert record['data'] == raw
    assert record['description'] == info['description']
    assert record['examinations'] == info['examinations']


def test_migration_memory_does_not_grow_with_total_inline_bytes(tmp_path):
    root = tmp_path/'root'
    legacy(root)
    # Trailing bytes are valid for this PNG and exercise storage size without
    # expanding pixels or requiring a large image decoder allocation.
    raw = png('blue') + b'x' * (1024 * 1024)
    encoded = base64.b64encode(raw).decode()
    with sqlite3.connect(root/'data/AICQ.db') as db:
        for row_id in range(2, 26):
            db.execute('INSERT INTO chat_messages VALUES (?,?,?,?,?,?)',
                       (row_id, 'chat', str(row_id),
                        json.dumps({f'inline_{row_id}': {'base64': encoded}}), '[]', row_id))
    tracemalloc.start()
    try:
        manifest = migration.scan(root)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert len([i for i in manifest['candidates'] if i.get('chat_source')]) == 25
    assert peak < 12 * 1024 * 1024
    output = tmp_path/'migration'
    migration.apply(manifest, root, output)
    migration.verify(manifest, output)
    tracemalloc.start()
    try:
        migration.cleanup(manifest, output)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert peak < 12 * 1024 * 1024
    with sqlite3.connect(root/'data/AICQ.db') as db:
        assert db.execute("SELECT count(*) FROM chat_messages WHERE images LIKE '%base64%'").fetchone()[0] == 0


def test_apply_rejects_changed_inline_original(tmp_path):
    root, output = tmp_path/'root', tmp_path/'migration'
    legacy(root)
    manifest = migration.scan(root)
    # Force the inline source to be selected over the identical sticker copy.
    next(i for i in manifest['candidates'] if i['ref'] == 'chat_old')['priority'] = -1
    with sqlite3.connect(root/'data/AICQ.db') as db:
        db.execute('UPDATE chat_messages SET images=? WHERE id=1',
                   (json.dumps({'chat_old': {'base64': base64.b64encode(png('blue')).decode()}}),))
    with pytest.raises(ValueError, match='Source changed since scan'):
        migration.apply(manifest, root, output)


def test_original_inline_manifest_can_still_resume(tmp_path):
    root, output = tmp_path/'root', tmp_path/'migration'
    legacy(root)
    manifest = migration.scan(root)
    manifest['version'] = 1
    for item in manifest['candidates']:
        item.pop('chat_source', None)
        if not item['path']:
            item['base64'] = base64.b64encode(png()).decode()
            item['priority'] = -1
    migration.apply(manifest, root, output)
    migration.verify(manifest, output)
    assert read_image('chat_old')['data'] == png()


def test_backup_and_verification_hash_files_without_reading_whole_files(tmp_path, monkeypatch):
    root, output = tmp_path/'root', tmp_path/'migration'
    legacy(root)
    manifest = migration.scan(root)
    read_bytes = Path.read_bytes

    def reject_database_buffer(path):
        if path.suffix == '.db':
            raise AssertionError('Database must be hashed in chunks')
        return read_bytes(path)

    monkeypatch.setattr(Path, 'read_bytes', reject_database_buffer)
    migration.apply(manifest, root, output)

    def reject_image_buffer(path):
        raise AssertionError('Verification must not fill the runtime image cache')

    monkeypatch.setattr(Path, 'read_bytes', reject_image_buffer)
    migration.verify(manifest, output)
    hashes = json.loads((output/'backup/complete.json').read_text())
    assert hashes[str(Path('data/AICQ.db'))] == hashlib.sha256(
        read_bytes(output/'backup/data/AICQ.db')).hexdigest()


def test_avif_original_is_migrated_without_conversion(tmp_path):
    from PIL import Image

    root, output = tmp_path/'root', tmp_path/'migration'
    legacy(root)
    path = root/'cache/browser_image/avif_old.avif'
    Image.new('RGB', (8, 8), 'blue').save(path, format='AVIF')
    original = path.read_bytes()
    manifest = migration.scan(root)
    assert not manifest['unavailable']
    migration.apply(manifest, root, output)
    migration.verify(manifest, output)
    record = read_image('avif_old')
    assert record['mime'] == 'image/avif'
    assert record['data'] == original


def test_cleanup_resumes_after_a_committed_message_without_deleting_originals(tmp_path):
    root, output = tmp_path/'root', tmp_path/'migration'
    original, duplicate = legacy(root)
    with sqlite3.connect(root/'data/AICQ.db') as db:
        db.execute('INSERT INTO chat_messages SELECT 2,session_key,?,images,content_segments,2 FROM chat_messages WHERE id=1', ('2',))
        db.execute("""CREATE TRIGGER interrupt_cleanup BEFORE UPDATE ON chat_messages
                      WHEN NEW.id=2 BEGIN SELECT RAISE(FAIL, 'interrupted cleanup'); END""")
    manifest = migration.scan(root)
    migration.apply(manifest, root, output)
    migration.verify(manifest, output)
    with pytest.raises(sqlite3.IntegrityError, match='interrupted cleanup'):
        migration.cleanup(manifest, output)
    with sqlite3.connect(root/'data/AICQ.db') as db:
        rows = db.execute('SELECT images FROM chat_messages ORDER BY id').fetchall()
        assert 'base64' not in rows[0][0]
        assert 'base64' in rows[1][0]
        db.execute('DROP TRIGGER interrupt_cleanup')
    assert original.exists() and duplicate.exists()
    migration.cleanup(manifest, output)
    assert read_image('chat_old')['data'] == png()
    with sqlite3.connect(root/'data/AICQ.db') as db:
        assert all('base64' not in row[0] for row in db.execute('SELECT images FROM chat_messages'))


def test_cleanup_rejects_damaged_database_backup_before_rewriting_inline_data(tmp_path):
    root, output = tmp_path/'root', tmp_path/'migration'
    original, duplicate = legacy(root)
    manifest = migration.scan(root)
    migration.apply(manifest, root, output)
    migration.verify(manifest, output)
    (output/'backup/data/AICQ.db').write_bytes(b'damaged backup')
    with pytest.raises(ValueError, match='Database backup verification failed'):
        migration.cleanup(manifest, output)
    with sqlite3.connect(root/'data/AICQ.db') as db:
        assert 'base64' in db.execute('SELECT images FROM chat_messages').fetchone()[0]
    assert original.exists() and duplicate.exists()
