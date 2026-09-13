"""Resumable legacy image import and explicit, separately requested cleanup."""
from __future__ import annotations

import base64
import hashlib
import json
import shutil
import sqlite3
import time
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path


def _hash(raw):
    return hashlib.sha256(raw).hexdigest()


def _hash_file(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    with temporary.open('w', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
    for attempt in range(20):
        try:
            temporary.replace(path)
            return
        except PermissionError:
            if attempt == 19:
                raise
            time.sleep(.05)


@contextmanager
def _open(path):
    db = sqlite3.connect(path.resolve().as_uri() + '?mode=ro', uri=True)
    db.row_factory = sqlite3.Row
    try:
        yield db
    finally:
        db.close()


def _image_entries(payload):
    """Retain the JSON key/index so inline originals can be read again on demand."""
    if isinstance(payload, dict):
        for key, info in payload.items():
            if isinstance(info, dict):
                yield key, key, info
    elif isinstance(payload, list):
        for index, info in enumerate(payload):
            if isinstance(info, dict):
                yield index, info.get('image_ref') or info.get('ref'), info


def _candidate_bytes(item, source_db):
    if item['path']:
        return Path(item['path']).read_bytes()
    if 'chat_source' not in item:
        # Resume manifests saved by the original importer.
        return base64.b64decode(item['base64'])
    locator = item['chat_source']
    row = source_db.execute('SELECT images FROM chat_messages WHERE id=?',
                            (locator['row_id'],)).fetchone()
    if row is None:
        raise ValueError('Source changed since scan: chat message missing')
    try:
        info = json.loads(row['images'])[locator['key']]
        return base64.b64decode(info['base64'], validate=True)
    except (ValueError, TypeError, KeyError, IndexError) as exc:
        raise ValueError('Source changed since scan: inline image missing or invalid') from exc


def scan(root: Path):
    """Inspect bytes without importing runtime services or modifying input stores."""
    from .image_resolver import inspect_image_payload
    from .media_storage import _SAFE_REF_PATTERN
    root = root.resolve()
    candidates, unavailable, files, bindings, browser = [], [], {}, {}, []
    def add(ref, *, raw=None, path=None, source='chat', priority=1, created=0, metadata=None, chat_source=None):
        location = str(path) if path else None
        try:
            if path:
                if path.stat().st_size > 20 * 1024 * 1024:
                    raise ValueError('image_too_large')
                raw = path.read_bytes()
            if raw is None:
                raise ValueError('no_original')
            info = inspect_image_payload(raw, max_bytes=20*1024*1024, max_pixels=100_000_000)
            digest = _hash(raw)
            if path:
                files[str(path.resolve())] = {'sha256': digest, 'size': len(raw)}
            candidates.append({'ref': ref, 'sha256': digest, 'size': len(raw), 'path': location,
                'chat_source': chat_source,
                'mime': info.mime_type, 'source': source, 'priority': priority,
                'created': str(created or ''), 'metadata': {
                    key: metadata[key] for key in ('description', 'examinations', 'phash')
                    if metadata and key in metadata
                }})
        except (OSError, ValueError) as exc:
            status = next((key for key in ('expired', 'failed') if (metadata or {}).get(key)), 'unavailable')
            unavailable.append({'ref': ref, 'path': location, 'reason': str(exc), 'status': status})

    db_path = root / 'data/AICQ.db'
    with _open(db_path) as db:
        tables = {r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        if 'chat_messages' in tables:
            for row in db.execute("SELECT id,images,created_at FROM chat_messages WHERE images IS NOT NULL"):
                try:
                    payload = json.loads(row['images'])
                except (ValueError, TypeError):
                    continue
                for key, ref, info in _image_entries(payload):
                    path = Path(info['file_path']) if info.get('file_path') else None
                    if path and not path.is_absolute():
                        path = root / path
                    try:
                        raw = base64.b64decode(info['base64'], validate=True) if not path and info.get('base64') else None
                    except ValueError:
                        raw = None
                    add(ref, raw=raw, path=path, created=row['created_at'], metadata=info,
                        chat_source={'row_id': row['id'], 'key': key} if not path else None)
        for table in ('media_registry', 'media_images'):
            if table in tables:
                for row in db.execute(f'SELECT * FROM {table}'):
                    path = Path(row['locator'])
                    if not path.is_absolute():
                        path = root / path
                    if path.is_file():
                        add(row['image_ref'], path=path, source=row['source_type'], created=row['created_at'])
                    if row['sha256']:
                        bindings.setdefault(row['image_ref'], set()).add(row['sha256'])
        if 'media_refs' in tables:
            for row in db.execute('SELECT ref,sha256 FROM media_refs'):
                bindings.setdefault(row['ref'], set()).update([row['sha256']] if row['sha256'] else [])
    ledger = root / 'data/media/references.sqlite3'
    if ledger.exists():
        with _open(ledger) as db:
            for row in db.execute('SELECT ref,sha256 FROM refs'):
                bindings.setdefault(row['ref'], set()).update([row['sha256']] if row['sha256'] else [])
    index = root / 'data/stickers/index.json'
    collection = json.loads(index.read_text(encoding='utf-8')) if index.exists() else {'version': 4, 'stickers': {}, 'ref_hashes': {}}
    if 'version' not in collection:
        collection = {'version': 1, 'stickers': collection, 'ref_hashes': {}}
    for ref, digest in collection.get('ref_hashes', {}).items():
        bindings.setdefault(ref, set()).add(digest)
    for ref, info in collection.get('stickers', {}).items():
        if info.get('filename'):
            path = root / 'data/stickers/images' / info['filename']
            add(ref if _SAFE_REF_PATTERN.fullmatch(ref) else None, path=path, source='sticker', priority=0, created=info.get('created_at'))
            for alias in info.get('aliases', []):
                add(alias, path=path, source='sticker', created=info.get('created_at'))
        else:
            matching = [item for item in candidates if item['ref'] == ref]
            for item in matching:
                item['priority'] = 0
            if not matching:
                unavailable.append({'ref': ref, 'reason': 'no_original', 'status': 'unavailable'})
    extensions = {'.png', '.jpg', '.jpeg', '.gif', '.webp', '.bmp', '.avif', '.ico'}
    for relative, source in [('data/media', 'chat'), ('data/stickers/images', 'sticker'), ('cache/browser_image', 'browser'), ('cache/image', 'legacy_cache')]:
        for path in sorted((root / relative).rglob('*')):
            if path.suffix.lower() not in extensions or not path.is_file():
                continue
            if str(path.resolve()) in files:
                continue
            ref = None if source == 'legacy_cache' or not _SAFE_REF_PATTERN.fullmatch(path.stem) else path.stem
            add(ref, path=path, source=source, priority=2)
    for path in (root / 'cache/browser_image/sendable').glob('*.json'):
        try:
            item = json.loads(path.read_text(encoding='utf-8'))
            if item.get('image_ref') and item.get('sha256'):
                browser.append(item)
        except ValueError:
            unavailable.append({'path': str(path), 'reason': 'invalid_browser_metadata'})
    by_ref = defaultdict(set)
    for item in candidates:
        if item['ref']:
            by_ref[item['ref']].add(item['sha256'])
    for ref, values in bindings.items():
        by_ref[ref].update(values)
    conflicts = {r: sorted(v) for r, v in by_ref.items() if len(v)>1}
    return {'version': 2, 'source_root': str(root), 'candidates': candidates,
        'files': files, 'unavailable': unavailable, 'conflicts': conflicts,
        'bindings': {r: sorted(v) for r,v in bindings.items()}, 'collection': collection,
        'browser': browser, 'phase': 'scanned'}


def summary(manifest):
    items = manifest['candidates']
    return {'phase': manifest['phase'], 'refs': len({i['ref'] for i in items if i['ref']}),
        'unique_contents': len({i['sha256'] for i in items}),
        'source_files': len(manifest['files']), 'unavailable': len(manifest['unavailable']),
        'conflicts': len(manifest['conflicts']),
        **{k: manifest[k] for k in ('verified_refs', 'removed_files', 'removed_bytes') if k in manifest}}


def backup(root, output):
    """Copy DB using SQLite backup API, plus a hash-verifiable legacy file snapshot."""
    target = output / 'backup'
    if (target / 'complete.json').exists():
        return
    target.mkdir(parents=True, exist_ok=True)
    with _open(root / 'data/AICQ.db') as source:
        dest_path = target / 'data/AICQ.db'
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(dest_path) as dest:
            source.backup(dest)
    for relative in ('data/media', 'data/stickers', 'cache/browser_image', 'cache/image'):
        if (root / relative).exists():
            shutil.copytree(root / relative, target / relative, dirs_exist_ok=True)
    hashes = {str(p.relative_to(target)): _hash_file(p) for p in target.rglob('*') if p.is_file()}
    _json(target / 'complete.json', hashes)


def configure(root):
    import database
    from . import media_storage, sticker_collection
    database.DB_PATH = str(root / 'data/AICQ.db')
    media_storage.MEDIA_ROOT = root / 'data/media'
    sticker_collection._STICKER_DIR = root / 'data/stickers'
    sticker_collection._IMAGES_DIR = root / 'data/stickers/images'
    sticker_collection._INDEX_PATH = root / 'data/stickers/index.json'


def apply(manifest, target, output):
    from .image_store import connection, _bind
    configure(target)
    if manifest['conflicts']:
        raise ValueError('Conflicting historical ref bindings; migration and cleanup refused')
    source = Path(manifest['source_root'])
    backup(source, output)
    if target != source and not (target / 'data/AICQ.db').exists():
        (target / 'data').mkdir(parents=True, exist_ok=True)
        shutil.copy2(output / 'backup/data/AICQ.db', target / 'data/AICQ.db')
    with _open(source / 'data/AICQ.db') as source_db:
        _register_candidates(manifest, output, source_db)
    mapping = manifest['mapping']
    content_mapping = manifest['content_mapping']
    with connection(write=True) as db:
        for ref, digests in manifest['bindings'].items():
            digest = next(iter(digests), None)
            canonical = mapping.get(ref) or content_mapping.get(digest)
            _bind(db, ref, digest, canonical)
            if canonical:
                mapping[ref] = canonical
        for item in manifest['unavailable']:
            if item.get('ref') and item['ref'] not in mapping:
                _bind(db, item['ref'], None)
                db.execute('UPDATE media_refs SET status=? WHERE ref=? AND canonical_ref IS NULL',
                           (item.get('status', 'unavailable'), item['ref']))
        for artifact in manifest['browser']:
            canonical = mapping.get(artifact['image_ref'])
            if not canonical:
                continue
            artifact = {**artifact, 'image_ref': canonical}
            old = db.execute('SELECT metadata FROM media_browser_artifacts WHERE image_ref=?', (canonical,)).fetchone()
            if old:
                artifact['confirmation_reasons'] = list(dict.fromkeys([*json.loads(old[0]).get('confirmation_reasons', []), *artifact.get('confirmation_reasons', [])]))
            db.execute('INSERT OR REPLACE INTO media_browser_artifacts VALUES (?,?)', (canonical, json.dumps(artifact)))
    collection = manifest['collection']
    merged = {}
    for ref, info in sorted(collection.get('stickers', {}).items(), key=lambda i: (i[1].get('created_at', ''), i[0])):
        canonical = mapping.get(ref) or content_mapping.get(info.get('sha256')) or ref
        if canonical not in merged:
            merged[canonical] = {k:v for k,v in info.items() if k != 'filename'}
            merged[canonical]['aliases'] = []
        for alias in [ref, *info.get('aliases', [])]:
            from .media_storage import _SAFE_REF_PATTERN
            if _SAFE_REF_PATTERN.fullmatch(alias) and alias != canonical and alias not in merged[canonical]['aliases']:
                merged[canonical]['aliases'].append(alias)
    collection = {'version': 4, 'stickers': {
        ref: {'description': info['description'], 'created_at': info['created_at']}
        for ref, info in merged.items()
    }}
    _json(target / 'data/stickers/index.json', collection)
    manifest['target_root'] = str(target)
    manifest['phase'] = 'applied'
    _json(output / 'manifest.json', manifest)


def _register_candidates(manifest, output, source_db):
    from .image_store import register_image, connection, _bind

    groups = defaultdict(list)
    for item in manifest['candidates']:
        groups[item['sha256']].append(item)
    mapping = manifest.setdefault('mapping', {})
    content_mapping = manifest.setdefault('content_mapping', {})
    for group_index, (digest, items) in enumerate(sorted(groups.items())):
        items.sort(key=lambda i: (i['priority'], i['created'], i['ref'] or '~'))
        chosen = items[0]
        raw = _candidate_bytes(chosen, source_db)
        if _hash(raw) != digest:
            raise ValueError('Source changed since scan')
        record = register_image(raw, chosen['source'], chosen['ref'])
        canonical = record['image_ref']
        content_mapping[digest] = canonical
        with connection(write=True) as db:
            for item in items:
                if item['ref']:
                    _bind(db, item['ref'], digest, canonical)
                    mapping[item['ref']] = canonical
            descriptions = [i['metadata'].get('description') for i in items if i['metadata'].get('description') and not i['metadata'].get('phash')]
            examinations = []
            for item in items:
                if item['metadata'].get('phash'):
                    continue
                for examination in item['metadata'].get('examinations', []):
                    if examination not in examinations:
                        examinations.append(examination)
            old = db.execute('SELECT description,examinations FROM media_images WHERE image_ref=?', (canonical,)).fetchone()
            for examination in json.loads(old['examinations']):
                if examination not in examinations:
                    examinations.append(examination)
            db.execute('UPDATE media_images SET description=?,examinations=? WHERE image_ref=?',
                (old['description'] or next(iter(descriptions), None), json.dumps(examinations, ensure_ascii=False), canonical))
        manifest['phase'] = 'registering'
        if group_index % 100 == 0:
            _json(output / 'manifest.json', manifest)


def _registered_hash_matches(ref, digest):
    from .image_store import lookup_image
    from .media_storage import _inside_media_root

    record = lookup_image(ref)
    if not record or record.get('unavailable_status') or record['sha256'] != digest:
        return False
    path = Path(record['locator'])
    if not _inside_media_root(path):
        return False
    try:
        return _hash_file(path) == digest
    except OSError:
        return False


def verify(manifest, output):
    from .image_store import connection
    configure(Path(manifest['target_root']))
    expected = {i['ref']: i['sha256'] for i in manifest['candidates'] if i['ref']}
    for ref, digest in expected.items():
        if not _registered_hash_matches(ref, digest):
            raise ValueError(f'Old ref failed verification: {ref}')
    for digest, ref in manifest.get('content_mapping', {}).items():
        if not _registered_hash_matches(ref, digest):
            raise ValueError(f'Canonical content failed verification: {ref}')
    manifest['verified_refs'] = len(expected)
    with connection(write=True) as db:
        db.execute("INSERT OR REPLACE INTO media_state VALUES ('legacy_import_verified','1')")
    manifest['phase'] = 'verified'
    _json(output / 'manifest.json', manifest)


def _compact_chat_messages(root, mapping):
    """Commit one message at a time, bounding both memory and the rollback journal."""
    from .image_store import connection

    last_id = None
    while True:
        # Keyset pagination releases the read lock before opening a write transaction.
        query = 'SELECT id,images,content_segments FROM chat_messages'
        params = ()
        if last_id is not None:
            query += ' WHERE id>?'
            params = (last_id,)
        with _open(root / 'data/AICQ.db') as db:
            row = db.execute(query + ' ORDER BY id LIMIT 1', params).fetchone()
        if row is None:
            return
        last_id = row['id']
        try:
            images = json.loads(row['images'] or '{}')
            segments = json.loads(row['content_segments'] or '[]')
        except (ValueError, TypeError):
            continue
        if not isinstance(images, (dict, list)) or not isinstance(segments, list):
            continue
        items = images.items() if isinstance(images, dict) else (
            (ref, info) for _, ref, info in _image_entries(images)
        )
        new = {}
        for ref, info in items:
            canonical = mapping.get(ref) if isinstance(info, dict) else None
            new[canonical or ref] = (
                {'image_ref': canonical, 'label': info.get('label', '图片')}
                if canonical and isinstance(info, dict) else info
            )
        updated_segments = []
        for segment in segments:
            if isinstance(segment, dict):
                ref = segment.get('image_ref') or segment.get('ref')
                if ref in mapping:
                    segment = {**segment, 'image_ref': mapping[ref]}
                    segment.pop('ref', None)
            updated_segments.append(segment)
        if new == images and updated_segments == segments:
            continue
        with connection(write=True) as db:
            db.execute('UPDATE chat_messages SET images=?,content_segments=? WHERE id=?',
                (json.dumps(new, ensure_ascii=False), json.dumps(updated_segments, ensure_ascii=False), last_id))


def cleanup(manifest, output):
    """Remove only manifest-listed duplicates after revalidating originals and backup."""
    from .image_store import connection
    if manifest['phase'] not in ('verified', 'cleaned'):
        raise ValueError('Verification is required before cleanup')
    verify(manifest, output)
    target, source = Path(manifest['target_root']), Path(manifest['source_root'])
    if target != source:
        manifest['phase'] = 'verified'  # Rehearsal must never delete source files.
        return
    backup_root = output / 'backup'
    backup_hashes = json.loads((backup_root / 'complete.json').read_text(encoding='utf-8'))
    database_backup = Path('data/AICQ.db')
    if _hash_file(backup_root / database_backup) != backup_hashes.get(str(database_backup)):
        raise ValueError('Database backup verification failed')
    _compact_chat_messages(target, manifest['mapping'])
    with connection(write=True) as db:
        formal = {Path(row[0]).resolve() for row in db.execute('SELECT locator FROM media_images')}
        old_registry = db.execute("SELECT type FROM sqlite_master WHERE name='media_registry'").fetchone()
        if old_registry and old_registry[0] == 'table':
            db.execute('DROP TABLE media_registry')
            db.execute('''CREATE VIEW media_registry AS
              SELECT r.ref AS image_ref,i.source_type,i.locator,i.mime,r.sha256,i.created_at
              FROM media_refs r JOIN media_images i ON r.canonical_ref=i.image_ref''')
    allowed = [source / rel for rel in ('data/media', 'data/stickers/images', 'cache/browser_image', 'cache/image')]
    count, size = 0, 0
    for file_index, (name, info) in enumerate(manifest['files'].items()):
        path = Path(name).resolve()
        if path in formal or not path.exists():
            continue
        if not any(path.is_relative_to(root.resolve()) for root in allowed):
            continue  # User workspace/export copies never belong to cache cleanup.
        relative = str(path.relative_to(source))
        backup_path = backup_root / relative
        if backup_hashes.get(relative) != info['sha256'] or _hash_file(backup_path) != info['sha256']:
            raise ValueError('Backup verification failed')
        if _hash_file(path) != info['sha256']:
            raise ValueError('Source changed before cleanup')
        with connection() as db:
            row = db.execute('SELECT image_ref FROM media_images WHERE sha256=?', (info['sha256'],)).fetchone()
        if not row or not _registered_hash_matches(row[0], info['sha256']):
            raise ValueError('Canonical original unavailable during cleanup')
        path.unlink()
        count += 1
        size += info['size']
        manifest['removed_files'] = manifest.get('removed_files', 0) + 1
        manifest['removed_bytes'] = manifest.get('removed_bytes', 0) + info['size']
        if file_index % 100 == 0:
            _json(output / 'manifest.json', manifest)
    manifest['phase'] = 'cleaned'
    _json(output / 'manifest.json', manifest)
