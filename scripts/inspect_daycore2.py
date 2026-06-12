import sqlite3, json

conn = sqlite3.connect("day_core.db")
c = conn.cursor()

# 查找群消息
c.execute("""
    SELECT time, user_id, group_id, summary, details
    FROM event_records
    WHERE post_type='message' AND event_type='group'
    ORDER BY id
    LIMIT 60
""")
rows = c.fetchall()
print(f"群消息总行: {len(rows)}")
for row in rows[:60]:
    time_, uid, gid, summary, details = row
    try:
        d = json.loads(details)
        text = ""
        raw = d.get("raw_message") or d.get("message","")
        if isinstance(raw, list):
            for seg in raw:
                if seg.get("type") == "text":
                    text += seg["data"].get("text","")
        else:
            text = str(raw)[:100]
        sender = d.get("sender",{})
        nick = sender.get("nickname","?") if isinstance(sender,dict) else "?"
    except:
        text = summary[:80]
        nick = str(uid)
    print(f"  [{time_[11:16]}] {nick}({uid}): {text[:80]}")
