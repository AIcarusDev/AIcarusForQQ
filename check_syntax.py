import ast
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
CHECK_DIRS = ("src", "scripts", "tests")
CHECK_FILES = ("run.py", "test_weather.py")


def iter_python_files():
    for filename in CHECK_FILES:
        path = ROOT / filename
        if path.exists():
            yield path
    for dirname in CHECK_DIRS:
        directory = ROOT / dirname
        if directory.exists():
            yield from sorted(directory.rglob("*.py"))


ok = True
for path in iter_python_files():
    try:
        ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
        print(f"OK: {path}")
    except SyntaxError as exc:
        print(f"ERROR: {path}: {exc}")
        ok = False

sys.exit(0 if ok else 1)
