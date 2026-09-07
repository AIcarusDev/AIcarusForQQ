"""Content-addressed originals and reference aliases in the application database.

All producers register bytes here. Source-specific metadata never owns originals.
"""
from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import tempfile
import threading
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from .media_identity import MediaRefConflict, MediaIdentityUnavailable

SCHEMA = """
CREATE TABLE IF NOT EXISTS media_refs (
 ref TEXT PRIMARY KEY, sha256 TEXT, canonical_ref TEXT, locator TEXT,
 status TEXT NOT NULL DEFAULT 'pending'
);
CREATE TABLE IF NOT EXISTS media_images (
 image_ref TEXT PRIMARY KEY, sha256 TEXT NOT NULL UNIQUE,
 locator TEXT NOT NULL, mime TEXT NOT NULL, width INTEGER NOT NULL,
 height INTEGER NOT NULL, frame_count INTEGER NOT NULL,
 source_type TEXT NOT NULL, created_at INTEGER NOT NULL,
 description TEXT, examinations TEXT NOT NULL DEFAULT '[]',
 description_lease TEXT, description_lease_until REAL NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS media_browser_artifacts (
 image_ref TEXT PRIMARY KEY, metadata TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS media_state (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE VIEW IF NOT EXISTS media_registry AS
 SELECT r.ref AS image_ref,i.source_type,i.locator,i.mime,r.sha256,i.created_at
 FROM media_refs r JOIN media_images i ON r.canonical_ref=i.image_ref;
"""


@contextmanager
def connection(*, write=False):
    import database
    path = Path(database.DB_PATH)
    if write:
        path.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(str(path) if write else path.resolve().as_uri()+'?mode=ro' if path.exists() else ':memory:',
                         uri=not write, timeout=30)
    db.row_factory = sqlite3.Row
    try:
        if write:
            db.executescript(SCHEMA)
            db.execute('PRAGMA synchronous=FULL')
            db.execute('BEGIN IMMEDIATE')
        yield db
        if write:
            db.commit()
    except BaseException:
        db.rollback()
        raise
    finally:
        db.close()


def _reserve(db, dt=None, initial_length=5):
    from .media_identity import _historical_binding
    if not 4 <= initial_length <= 12:
        raise ValueError('Reference suffix must start at 4..12 digits')
    prefix = (dt or datetime.now(timezone.utc)).strftime('%y%m%d')
    for length in range(initial_length, 13):
        for _ in range(8):
            ref = f'{prefix}_{uuid.uuid4().hex[:length]}'
            if db.execute('SELECT 1 FROM media_refs WHERE ref=?', (ref,)).fetchone():
                continue
            occupied, _ = _historical_binding(ref)
            db.execute('INSERT INTO media_refs(ref) VALUES (?)', (ref,))
            if not occupied:
                return ref
    raise MediaRefConflict('Media reference allocation exhausted')


def reserve_ref(dt=None, *, initial_length=5):
    try:
        with connection(write=True) as db:
            return _reserve(db, dt, initial_length)
    except (OSError, sqlite3.Error) as exc:
        raise MediaIdentityUnavailable('Media identity persistence failed') from exc


def _bind(db, ref, digest, canonical=None):
    from .media_storage import _SAFE_REF_PATTERN
    from .media_identity import _historical_binding
    if not _SAFE_REF_PATTERN.fullmatch(ref):
        raise ValueError('Invalid media reference')
    old = db.execute('SELECT sha256 FROM media_refs WHERE ref=?', (ref,)).fetchone()
    if old and old[0]:
        if digest and old[0] != digest:
            raise MediaRefConflict('Media reference is bound to different content')
        digest = old[0]
    else:
        _, historical = _historical_binding(ref)
        if len(historical) > 1 or (digest and historical and digest not in historical):
            raise MediaRefConflict('Historical media reference has conflicting content')
        digest = digest or next(iter(historical), None)
    db.execute('''INSERT INTO media_refs(ref,sha256,canonical_ref) VALUES (?,?,?)
      ON CONFLICT(ref) DO UPDATE SET sha256=excluded.sha256,
      canonical_ref=COALESCE(excluded.canonical_ref,media_refs.canonical_ref)''',
      (ref, digest, canonical))


def bind_ref(ref, digest):
    with connection(write=True) as db:
        _bind(db, ref, digest)


def _atomic_original(path, raw):
    from .media_storage import _inside_media_root
    if not _inside_media_root(path):
        raise ValueError('Media file is outside media root')
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != raw:
            raise MediaRefConflict('Media file contains different content')
        return
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix='.image-', delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            if path.read_bytes() != raw:
                raise MediaRefConflict('Media file contains different content')
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def register_image(raw: bytes, source: str, reserved_ref: str | None = None, *, dt=None):
    """Commit an exact original and return canonical metadata plus ``reused``."""
    from .image_resolver import inspect_image_payload
    from .media_storage import get_media_dir, parse_time_ref_date, _MIME_TO_EXT
    info = inspect_image_payload(raw, max_bytes=20 * 1024 * 1024, max_pixels=100_000_000)
    digest = hashlib.sha256(raw).hexdigest()
    # A published reservation remains bound even when the subsequent file write fails.
    if reserved_ref:
        bind_ref(reserved_ref, digest)
    with connection(write=True) as db:
        if reserved_ref:
            _bind(db, reserved_ref, digest)
        existing = db.execute('SELECT * FROM media_images WHERE sha256=?', (digest,)).fetchone()
        if existing:
            record = dict(existing)
            path = Path(record['locator'])
            # Missing copies can be restored from identical verified input; changed files cannot.
            _atomic_original(path, raw)
            if reserved_ref:
                _bind(db, reserved_ref, digest, record['image_ref'])
            record['reused'] = True
        else:
            ref = reserved_ref or _reserve(db, dt)
            date = parse_time_ref_date(ref)
            now = dt or datetime.now(timezone.utc)
            year, month = date[:2] if date else (now.year, now.month)
            path = (get_media_dir(year, month) / (ref + _MIME_TO_EXT[info.mime_type])).resolve()
            _atomic_original(path, raw)
            _bind(db, ref, digest, ref)
            db.execute('''INSERT INTO media_images
              (image_ref,sha256,locator,mime,width,height,frame_count,source_type,created_at)
              VALUES (?,?,?,?,?,?,?,?,?)''', (ref, digest, str(path), info.mime_type,
              info.width, info.height, info.frame_count, source, int(time.time()*1000)))
            record = dict(db.execute('SELECT * FROM media_images WHERE image_ref=?', (ref,)).fetchone())
            record['reused'] = False
    return _record(record)


def _record(record):
    record['examinations'] = json.loads(record.get('examinations') or '[]')
    record['file_path'] = record['locator']
    return record


def lookup_image(ref):
    with connection() as db:
        try:
            row = db.execute('''SELECT i.* FROM media_refs r JOIN media_images i
                ON i.image_ref=r.canonical_ref WHERE r.ref=?''', (str(ref),)).fetchone()
            if row:
                return _record(dict(row))
            row = db.execute('SELECT sha256,status FROM media_refs WHERE ref=?', (str(ref),)).fetchone()
            if row:
                return {'image_ref': str(ref), 'unavailable_status': row[1] if row[1] != 'pending' or not row[0] else 'unavailable'}
        except sqlite3.OperationalError as exc:
            if 'no such table' not in str(exc):
                raise
    return None


def set_ref_status(ref, status):
    if status not in ('pending', 'failed', 'expired', 'unavailable'):
        raise ValueError('Invalid image status')
    with connection(write=True) as db:
        db.execute('UPDATE media_refs SET status=? WHERE ref=? AND canonical_ref IS NULL', (status, ref))


def read_image(ref):
    record = lookup_image(ref)
    if not record or record.get('unavailable_status'):
        return record
    from .media_storage import _inside_media_root
    path = Path(record['locator'])
    if not _inside_media_root(path):
        return {**record, 'unavailable_status': 'invalid_path'}
    try:
        stat = path.stat()
        fingerprint = (stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
        from .media_cache import get_recent_image
        cached = get_recent_image(record['image_ref'])
        if cached and cached[0].get('_file_stat') == fingerprint and cached[0].get('sha256') == record['sha256']:
            raw = cached[0]['data']
        else:
            raw = path.read_bytes()
    except OSError:
        return {**record, 'unavailable_status': 'missing_image'}
    if hashlib.sha256(raw).hexdigest() != record['sha256']:
        return {**record, 'unavailable_status': 'image_changed'}
    record['data'] = raw
    record['_file_stat'] = fingerprint
    from .media_cache import cache_recent_image
    cache_recent_image(record['image_ref'], record, record['source_type'])
    return record


def image_link(record, *, label="图片"):
    return {'image_ref': record['image_ref'], 'mime': record['mime'], 'label': label}


def replace_entry_ref(entry, old, record):
    """Replace an occurrence reference without filtering or reordering segments."""
    ref = record['image_ref']
    images = entry.setdefault('images', {})
    previous = images.pop(old, {})
    images[ref] = image_link(record, label=previous.get('label', '图片'))
    for segment in entry.get('content_segments', []):
        if segment.get('image_ref') == old:
            segment['image_ref'] = ref


def register_entry(entry, source='chat'):
    """Normalize inbound legacy payloads before opening a chat write transaction."""
    import base64
    import database
    items = database._media_payload_items(entry.get('images'))
    known = {ref for ref, _ in items}
    for segment in entry.get('content_segments', []):
        ref = segment.get('image_ref')
        if ref and ref not in known:
            current = lookup_image(ref)
            if current and not current.get('unavailable_status'):
                items.append((ref, image_link(current)))
                known.add(ref)
    if not items:
        return
    entry['images'] = dict(items)
    for ref, payload in items:
        current = lookup_image(ref)
        if current and not current.get('unavailable_status') and not any(payload.get(k) for k in ('base64', 'data', 'file_path')):
            replace_entry_ref(entry, ref, current)
            continue
        if any(payload.get(k) for k in ('pending', 'failed', 'expired', 'unavailable_status')):
            status = next((key for key in ('expired', 'failed', 'pending') if payload.get(key)), 'unavailable')
            set_ref_status(ref, status)
            continue
        raw = payload.get('data')
        if not isinstance(raw, bytes):
            raw = None
        if raw is None and payload.get('base64'):
            raw = base64.b64decode(payload['base64'], validate=True)
        if raw is None and payload.get('file_path'):
            raw = Path(payload['file_path']).read_bytes()
        if raw is not None:
            from .image_resolver import ImagePayloadError
            try:
                record = register_image(raw, source, ref)
            except ImagePayloadError as exc:
                bind_ref(ref, hashlib.sha256(raw).hexdigest())
                set_ref_status(ref, 'unavailable')
                entry['images'][ref] = {'image_ref': ref, 'unavailable_status': exc.code,
                                        'label': payload.get('label', '图片')}
                continue
            replace_entry_ref(entry, ref, record)


def update_description(ref, description):
    with connection(write=True) as db:
        db.execute('''UPDATE media_images SET description=? WHERE image_ref=
          (SELECT canonical_ref FROM media_refs WHERE ref=?)''', (description, ref))
    from .media_cache import clear_recent_media_cache
    clear_recent_media_cache()


def append_examination(ref, focus, result_text):
    with connection(write=True) as db:
        row = db.execute('''SELECT i.image_ref,i.examinations FROM media_images i JOIN media_refs r
          ON i.image_ref=r.canonical_ref WHERE r.ref=?''', (ref,)).fetchone()
        if row is None:
            raise MediaIdentityUnavailable('Image is not registered')
        entries = json.loads(row[1])
        entries.append({'focus': focus, 'result': result_text, 'examined_at': datetime.now(timezone.utc).isoformat()})
        db.execute('UPDATE media_images SET examinations=? WHERE image_ref=?', (json.dumps(entries, ensure_ascii=False), row[0]))
    from .media_cache import clear_recent_media_cache
    clear_recent_media_cache()


@contextmanager
def description_claim(ref):
    """Cross-process lease; the winner describes, waiters consume its result."""
    token = uuid.uuid4().hex
    acquired = False
    stopped = threading.Event()
    heartbeat = None
    try:
        deadline = time.monotonic() + 120
        while time.monotonic() < deadline:
            with connection(write=True) as db:
                row = db.execute('''SELECT i.* FROM media_images i JOIN media_refs r
                  ON i.image_ref=r.canonical_ref WHERE r.ref=?''', (ref,)).fetchone()
                if row is None or row['description']:
                    break
                if row['description_lease_until'] < time.time():
                    db.execute('UPDATE media_images SET description_lease=?,description_lease_until=? WHERE image_ref=?',
                               (token, time.time()+300, row['image_ref']))
                    acquired = True
                    break
            time.sleep(.1)
        if acquired:
            def renew():
                while not stopped.wait(30):
                    try:
                        with connection(write=True) as db:
                            db.execute('UPDATE media_images SET description_lease_until=? WHERE description_lease=?',
                                       (time.time()+300, token))
                    except sqlite3.Error:
                        return
            heartbeat = threading.Thread(target=renew, name='image-description-lease', daemon=True)
            heartbeat.start()
        yield acquired
    finally:
        stopped.set()
        if heartbeat:
            heartbeat.join(timeout=1)
        if acquired:
            with connection(write=True) as db:
                db.execute('UPDATE media_images SET description_lease=NULL,description_lease_until=0 WHERE description_lease=?', (token,))
