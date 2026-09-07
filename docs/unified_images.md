# Unified images

All producers call `llm.media.image_store.register_image(raw, source, reserved_ref=None)`.
It returns canonical metadata (`image_ref`, `sha256`, `mime`, dimensions, frame count,
file location, description, examinations) and `reused`. Equality means identical
original bytes, not perceptual similarity. Actual MIME is detected from the image.

`reserve_ref()` is for downloads whose bytes are not available yet. Registration
binds that reservation permanently; an existing identical image makes it an alias.
`read_image(ref)` accepts canonical refs and aliases and returns bytes plus current
metadata, `None` for an unknown ref, or an `unavailable_status`. Callers must not
replace unavailable data with context bytes, a source URL, or a similar image.

The application SQLite database owns `media_images` (unique full SHA-256),
`media_refs` (reservations and aliases), and browser send-confirmation metadata.
`media_registry` is a compatibility read view over those records after migration.
The old registry table and `references.sqlite3` are migration inputs only.
Originals live in `data/media/YYYY/MM/`; the LRU holds canonical image payloads.
File integrity is checked before serving; cache hits reuse verified bytes while
reading current shared descriptions/examinations. Missing originals can be repaired
by registering the same bytes again; changed files cause an error.

Collection metadata owns the usage description and membership, not another image
file. Removing a collection item or recalling a chat does not remove the shared
original. Browser resource references still identify download candidates. Browser
confirmation requirements survive aliases and content deduplication. Exporting an
image into the user's workspace remains an independent, no-overwrite copy.

Vision descriptions are shared by canonical identity. A cross-process lease
coalesces concurrent first descriptions; failures permit retries. Explicit focused
examinations still run on each request and append to shared history. pHash is not an
image identity and its previous payload/description fallback has been removed.

## Migration

Both normal and WebUI-only startup automatically import historical images before
database initialization and before accepting requests. A separate worker backs up,
applies and verifies the import, under a cross-process SQLite lock. Its checkpoint
and independent backup live in `data/migrations/unified-images-v1/`. A verified
database skips this work on later launches, including installations migrated manually.
New installations initialize the same schema without needing manual commands.

Failed imports block startup and retain progress for retry. Resolve the reported
error before restarting. Historical ref conflicts require reviewing the manifest;
if no import has started, correct the source records and archive the scanned
manifest before retrying so they can be rescanned. Never discard an in-progress
manifest or the backup. Stop the old application before upgrading; the startup
lock coordinates new launchers, not already-running older versions.

Startup does **not** delete old files or compact historical chat payloads. After
accepting the migration, stop the application and explicitly run `cleanup` with
`--output data/migrations/unified-images-v1` to reclaim duplicates. The desktop
launcher allows up to one hour for first startup, and displays the WebUI when ready.

Use the same Python environment as the bot, including its Pillow codecs. Stop all
application writers before production `apply` and `cleanup`. The CLI never stops
processes itself and never runs destructive migration during normal startup.

```powershell
python -B scripts/migrate_unified_images.py dry-run --output output/unified-images-production
python -B scripts/migrate_unified_images.py apply --output output/unified-images-production
python -B scripts/migrate_unified_images.py verify --output output/unified-images-production
python -B scripts/migrate_unified_images.py cleanup --output output/unified-images-production
```

`apply --target-root <empty-directory>` rehearses registration against a separate
database and image root, without deleting source files. To rehearse actual deletion,
copy the source database and stores into an isolated root, remap historical absolute
locators into that root, then run all four phases with `--root <isolated-root>`.

The migration keeps `manifest.json` and `backup/complete.json`, including hashes.
It preserves collection primary refs first, otherwise the earliest usable ref.
Aliases are flattened. Invalid/missing originals stay unavailable; conflicts stop
migration rather than selecting one image. Only exact, attributable legacy image
metadata is imported; pHash-associated descriptions remain in the backup.

Cleanup verifies all old refs and backed-up bytes again. It rewrites historical
inline payloads to links and removes only listed duplicate files under managed
image directories. It never deletes workspace/export files. Completed groups and
aliases can be replayed safely after interruption. Retain the backup separately
until the migration is accepted.

To roll back: stop writers, restore the pre-migration application revision and
`backup/data/AICQ.db`, restore the backed-up `data/media`, `data/stickers`,
`cache/browser_image`, and `cache/image` trees to their original locations, and
start in the prior mode. Do not restore an old database while an application
process has it open. Newly created unified originals may be retained as inert
files until rollback validation is complete.
