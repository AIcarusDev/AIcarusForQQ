"""Explicit image migration: dry-run, apply, verify, cleanup. Stop writers first."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from llm.media import image_migration as migration


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=['dry-run', 'apply', 'verify', 'cleanup', 'startup'])
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--target-root', type=Path, help='Separate root for a non-destructive rehearsal')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.phase == 'startup':
        from llm.media.image_upgrade import upgrade
        upgrade(args.root)
        return
    if args.output is None:
        parser.error('--output is required for manual migration phases')
    root, output = args.root.resolve(), args.output.resolve()
    if args.phase == 'dry-run':
        manifest = migration.scan(root)
        migration._json(output / 'manifest.json', manifest)
    else:
        manifest = json.loads((output / 'manifest.json').read_text(encoding='utf-8'))
        if args.phase == 'apply':
            migration.apply(manifest, (args.target_root or root).resolve(), output)
        elif args.phase == 'verify':
            migration.verify(manifest, output)
        else:
            migration.cleanup(manifest, output)
    print(json.dumps(migration.summary(manifest), ensure_ascii=False))


if __name__ == '__main__':
    main()
