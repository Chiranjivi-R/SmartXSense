import sqlite3
from pathlib import Path

conn = sqlite3.connect('backend/instance/smartxsense.db')
cur = conn.cursor()
cur.execute('SELECT dataset_id, filename, domain, content_hash, file_path, created_at FROM datasets ORDER BY created_at')
rows = cur.fetchall()
print(f'Total datasets: {len(rows)}')
print()
for r in rows:
    dataset_id, filename, domain, content_hash, file_path, created_at = r
    hash_display = content_hash[:12] if content_hash else 'NULL'
    path_exists = Path(file_path).exists() if file_path else False
    print(f'  ID={dataset_id[:8]} file={filename} domain={domain} hash={hash_display} path_ok={path_exists}')

print()
cur.execute('SELECT content_hash, COUNT(*) as cnt FROM datasets GROUP BY content_hash HAVING cnt > 1')
dups = cur.fetchall()
print(f'Duplicate content_hash groups: {len(dups)}')
for d in dups:
    hash_val = d[0][:16] if d[0] else 'NULL'
    print(f'  hash={hash_val} count={d[1]}')

cur.execute('SELECT content_hash, COUNT(*) as cnt FROM datasets WHERE content_hash IS NULL GROUP BY content_hash')
nulls = cur.fetchall()
print(f'NULL content_hash rows: {len(nulls)}')

conn.close()
