import sqlite3

db = sqlite3.connect(r'F:\github\Feng\AIcarusForQQ\data\AICQ.db')
db.row_factory = sqlite3.Row

print('=== 1. 事件类型分布 ===')
for r in db.execute('SELECT event_type, COUNT(*) as n FROM MemoryEvents WHERE is_deleted=0 GROUP BY event_type ORDER BY n DESC'):
    print(f'  {r["event_type"]:40s} {r["n"]}')

print()
print('=== 2. context_type 分布 ===')
for r in db.execute('SELECT context_type, COUNT(*) as n FROM MemoryEvents WHERE is_deleted=0 GROUP BY context_type ORDER BY n DESC'):
    print(f'  {r["context_type"]:20s} {r["n"]}')

print()
print('=== 3. 合并/取代使用情况 ===')
r = db.execute('SELECT COUNT(*) FROM MemoryEvents WHERE merge_into IS NOT NULL').fetchone()
print(f'  merge_into 非空(被合并/软删): {r[0]}')
r = db.execute('SELECT COUNT(*) FROM MemoryEvents WHERE supersedes IS NOT NULL AND is_deleted=0').fetchone()
print(f'  supersedes 非空(主动取代): {r[0]}')
r = db.execute('SELECT COUNT(*) FROM MemoryEvents WHERE is_deleted=1').fetchone()
print(f'  软删除事件总数: {r[0]}')
r = db.execute('SELECT MAX(occurrences), AVG(occurrences) FROM MemoryEvents WHERE is_deleted=0').fetchone()
print(f'  occurrences  max={r[0]}  avg={r[1]:.2f}')

print()
print('=== 4. 实体命名空间合规检查 (前30违规) ===')
rows = db.execute('''
  SELECT entity, COUNT(*) as n FROM MemoryRoles
  WHERE entity NOT LIKE 'Bot:%'
    AND entity NOT LIKE 'User:qq_%'
    AND entity NOT LIKE 'Tool:%'
    AND entity NOT LIKE 'Person:%'
    AND entity NOT LIKE 'Org:%'
    AND entity NOT LIKE 'Group:qq_%'
    AND entity NOT LIKE 'Concept:%'
    AND entity NOT LIKE 'Place:%'
    AND entity NOT LIKE 'Event:%'
  GROUP BY entity ORDER BY n DESC LIMIT 30
''').fetchall()
if rows:
    for r in rows:
        print(f'  [{r["entity"]}] x{r["n"]}')
else:
    print('  (全部合规)')

print()
print('=== 5. 置信度分布 ===')
for r in db.execute("""
  SELECT CASE
    WHEN confidence<0.3 THEN '<0.3'
    WHEN confidence<0.5 THEN '0.3-0.5'
    WHEN confidence<0.7 THEN '0.5-0.7'
    WHEN confidence<0.9 THEN '0.7-0.9'
    ELSE '>=0.9'
  END as bucket, COUNT(*) as n
  FROM MemoryEvents WHERE is_deleted=0
  GROUP BY bucket ORDER BY bucket
"""):
    print(f'  {r["bucket"]:10s} {r["n"]}')

print()
print('=== 6. 角色类型分布 (Top 20) ===')
for r in db.execute('SELECT role, COUNT(*) as n FROM MemoryRoles GROUP BY role ORDER BY n DESC LIMIT 20'):
    print(f'  {r["role"]:30s} {r["n"]}')

print()
print('=== 7. 孤立事件 (无命名实体角色) ===')
r = db.execute('''
  SELECT COUNT(*) FROM MemoryEvents e WHERE is_deleted=0
  AND NOT EXISTS (
    SELECT 1 FROM MemoryRoles r WHERE r.event_id=e.event_id
    AND r.entity IS NOT NULL AND r.entity != ''
  )
''').fetchone()
print(f'  无实体角色的事件: {r[0]}')

print()
print('=== 8. 高频实体 Top 20 ===')
for r in db.execute('''
  SELECT entity, COUNT(DISTINCT event_id) as events
  FROM MemoryRoles WHERE entity IS NOT NULL AND entity != ''
  GROUP BY entity ORDER BY events DESC LIMIT 20
'''):
    print(f'  {r["entity"]:35s} {r["events"]} events')

print()
print('=== 9. 随机抽样 25 条 ===')
for r in db.execute('''
  SELECT event_id, event_type, context_type, confidence, occurrences, summary
  FROM MemoryEvents WHERE is_deleted=0 ORDER BY RANDOM() LIMIT 25
'''):
    print(f'  [{r["event_id"]:4}] {r["event_type"]:30s} {r["context_type"]:12s} conf={r["confidence"]:.2f} occ={r["occurrences"]}')
    print(f'        {r["summary"]}')

print()
print('=== 10. summary 异常检查 (过短/过长/含JSON/含英文占比高) ===')
rows = db.execute('''
  SELECT event_id, event_type, length(summary) as l, summary
  FROM MemoryEvents WHERE is_deleted=0
  ORDER BY l DESC LIMIT 10
''').fetchall()
print('  最长 summary TOP10:')
for r in rows:
    preview = r["summary"][:80].replace('\n', ' ')
    print(f'    [{r["event_id"]}] len={r["l"]} {preview}...')

rows = db.execute('''
  SELECT event_id, event_type, length(summary) as l, summary
  FROM MemoryEvents WHERE is_deleted=0
  ORDER BY l ASC LIMIT 10
''').fetchall()
print('  最短 summary TOP10:')
for r in rows:
    print(f'    [{r["event_id"]}] len={r["l"]} {r["summary"]}')

db.close()
print('\n=== 检查完成 ===')
