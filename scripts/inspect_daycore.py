import sqlite3, json, sys

conn = sqlite3.connect("day_core.db")
c = conn.cursor()

c.execute("SELECT name FROM sqlite_master WHERE type='table'")
tables = [r[0] for r in c.fetchall()]
print("Tables:", tables)

for t in tables:
    c.execute(f"PRAGMA table_info({t})")
    cols = [r[1] for r in c.fetchall()]
    print(f"\n[{t}]:", cols)
    c.execute(f"SELECT * FROM {t} LIMIT 3")
    rows = c.fetchall()
    for row in rows:
        # truncate long fields
        short = tuple(str(v)[:120] if isinstance(v, str) and len(str(v)) > 120 else v for v in row)
        print(" ", short)
