import sys
sys.stdout.reconfigure(encoding='utf-8')

raw = open('logs/real_chat_result2.txt', encoding='utf-8-sig').read()
lines = raw.splitlines()

out = open('logs/real_chat_summary2.txt', 'w', encoding='utf-8')
for line in lines:
    out.write(line + '\n')
out.close()
print(f"Total lines: {len(lines)}")
