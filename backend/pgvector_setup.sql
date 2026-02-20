-- ============================================================
-- pgvector Setup for Smart Research Agent
-- Run these commands inside psql or your PostgreSQL client
-- ============================================================

-- 1. Create the database
CREATE DATABASE research_agent_db;
\c research_agent_db;

-- 2. Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- 3. After running Django migrations, create the vector index:
--    (Run AFTER: python manage.py migrate)

-- Cosine index on chunk embeddings (IVFFlat — fast approximate search)
CREATE INDEX IF NOT EXISTS chunk_embedding_cosine_idx
ON research_agent_chunk
USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 50);   -- tune: ~sqrt(num_rows)

-- B-tree index for efficient page_number filtering
CREATE INDEX IF NOT EXISTS chunk_page_idx
ON research_agent_chunk (document_id, page_number);

-- ============================================================
-- Verify setup
-- ============================================================
SELECT extname, extversion FROM pg_extension WHERE extname = 'vector';

-- ============================================================
-- Tuning notes
-- ============================================================
-- • lists=50 works well for < 500K chunks; increase to 100-200 for larger corpora
-- • For exact search (small datasets): use HNSW index instead:
--   CREATE INDEX chunk_embedding_hnsw_idx ON research_agent_chunk
--   USING hnsw (embedding vector_cosine_ops);
-- • Set ivfflat.probes = 10 at query time for better recall:
--   SET ivfflat.probes = 10;
