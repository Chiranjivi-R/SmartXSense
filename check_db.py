"""Diagnostic script: inspect SQLite schema and check for missing columns."""
import sqlite3
import os

DB_PATHS = [
    "smartxsense.db",
    "backend/instance/smartxsense.db",
    "instance/smartxsense.db",
]

for db_path in DB_PATHS:
    if os.path.exists(db_path):
        print(f"\n=== DATABASE: {db_path} ===")
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()

        # List all tables
        cur.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name;")
        tables = [row[0] for row in cur.fetchall()]
        print(f"Tables: {tables}")

        # Show datasets table schema
        cur.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name='datasets';")
        result = cur.fetchone()
        if result:
            print(f"\ndatasets CREATE SQL:\n{result[0]}")
        else:
            print("\nNo 'datasets' table found!")

        # Show PRAGMA info (columns)
        cur.execute("PRAGMA table_info(datasets);")
        cols = cur.fetchall()
        print(f"\ndatasets columns (cid, name, type, notnull, dflt_value, pk):")
        for col in cols:
            print(f"  {col}")

        # Check if content_hash exists
        col_names = [col[1] for col in cols]
        print(f"\ncontent_hash present: {'content_hash' in col_names}")
        print(f"All column names: {col_names}")

        # Show row count
        try:
            cur.execute("SELECT COUNT(*) FROM datasets;")
            print(f"\ndatasets row count: {cur.fetchone()[0]}")
        except Exception as e:
            print(f"Error querying datasets: {e}")

        conn.close()
    else:
        print(f"DB not found: {db_path}")
