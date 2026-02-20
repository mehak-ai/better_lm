import os
import sys
import psycopg2
from sentence_transformers import SentenceTransformer

# 1. Check DB
print("Checking DB connection...")
try:
    conn = psycopg2.connect(
        dbname="research_agent_db",
        user="postgres",
        password="mehak",
        host="localhost",
        port=5432
    )
    print("DB Connection: OK")
    conn.close()
except Exception as e:
    print(f"DB Connection FAILED: {e}")

# 2. Check Embedding Model
print("\nChecking Embedding Model...")
try:
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    print("Model Loading: OK")
    print("Encoding test:", model.encode("hello world")[:3])
except Exception as e:
    print(f"Model Loading FAILED: {e}")
