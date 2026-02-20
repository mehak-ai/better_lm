"""
Setup script: creates the database and enables pgvector extension.
Run once before migrate: venv\Scripts\python.exe setup_db.py
"""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(dotenv_path=Path(__file__).parent / ".env")

import psycopg2
from psycopg2 import sql
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT

DB_NAME = os.getenv("DB_NAME", "research_agent_db")
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "")
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")

print(f"Connecting to postgres@{DB_HOST}:{DB_PORT} as {DB_USER}...")

# Connect to the default 'postgres' db to run CREATE DATABASE
conn = psycopg2.connect(
    dbname="postgres",
    user=DB_USER,
    password=DB_PASSWORD,
    host=DB_HOST,
    port=DB_PORT,
)
conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
cur = conn.cursor()

# Create db if not exists
cur.execute("SELECT 1 FROM pg_database WHERE datname = %s", (DB_NAME,))
if cur.fetchone():
    print(f"Database '{DB_NAME}' already exists — skipping creation.")
else:
    cur.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(DB_NAME)))
    print(f"Database '{DB_NAME}' created.")

cur.close()
conn.close()

# Now connect to the new db and enable pgvector
conn2 = psycopg2.connect(
    dbname=DB_NAME,
    user=DB_USER,
    password=DB_PASSWORD,
    host=DB_HOST,
    port=DB_PORT,
)
conn2.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
cur2 = conn2.cursor()
try:
    cur2.execute("CREATE EXTENSION IF NOT EXISTS vector;")
    print("pgvector extension enabled.")
except psycopg2.errors.FeatureNotSupported:
    print("\n" + "="*60)
    print("ACTION REQUIRED — pgvector is not installed on PostgreSQL 17.")
    print("="*60)
    print("1. Download from: https://github.com/pgvector/pgvector/releases")
    print("   (pick the Windows / pg17 build)")
    print("2. Copy vector.dll  → C:\\Program Files\\PostgreSQL\\17\\lib\\")
    print("   Copy vector.control + vector--*.sql")
    print("         → C:\\Program Files\\PostgreSQL\\17\\share\\extension\\")
    print("3. Restart PostgreSQL 17 in Windows Services")
    print("4. Re-run this script")
    print("="*60 + "\n")
cur2.close()
conn2.close()

print("Done! Now run: venv\\Scripts\\python.exe manage.py migrate")
