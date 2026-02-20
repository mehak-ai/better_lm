# Smart Research Agent — MVP

> A NotebookLM-style research assistant powered by Django, pgvector, and Next.js.

---

## Architecture

```
frontend/ (Next.js 14 App Router)
  src/app/          — routes & layout
  src/components/   — Chat, Sidebar, MermaidRenderer, CitationList
  src/lib/          — types.ts, api.ts (SSE streaming client)

backend/ (Django)
  research_agent/
    models.py       — Document, Chunk, ChatSession, SessionDocument, Message
    services.py     — EmbeddingService, ParserService, ChunkingService,
                      IngestionService, RetrievalService, LLMService,
                      IntentRouter, TitleService, PromptBuilder
    views.py        — UploadView, SessionListView, ChatView (SSE), …
    urls.py
```

---

## Quick Start

### 1. Database

```bash
# In psql:
CREATE DATABASE research_agent_db;
\c research_agent_db
CREATE EXTENSION IF NOT EXISTS vector;
```

### 2. Backend

```bash
cd backend
python -m venv venv
venv\Scripts\activate      # Windows
# source venv/bin/activate  # macOS/Linux

pip install -r requirements.txt

# Copy and fill in .env
copy .env.example .env

python manage.py migrate

# Create pgvector cosine index (after migrate)
# psql -d research_agent_db -f pgvector_setup.sql

python manage.py runserver
```

### 3. Frontend

```bash
cd frontend
npm install
npm run dev
```

Open http://localhost:3000

---

## API Endpoints

| Method | URL | Description |
|--------|-----|-------------|
| `POST` | `/api/upload/` | Upload files, create session |
| `GET`  | `/api/sessions/` | List all sessions |
| `GET`  | `/api/sessions/<id>/` | Session detail |
| `DELETE` | `/api/sessions/<id>/` | Delete session |
| `POST` | `/api/sessions/<id>/chat/` | Streaming RAG chat (SSE) |
| `GET`  | `/api/sessions/<id>/messages/` | Message history |

### Chat Request Body
```json
{
  "query": "What are the key findings?",
  "document_ids": [1, 2],
  "page_start": 1,
  "page_end": 10
}
```

### SSE Events
```
data: {"type": "token", "content": "..."}
data: {"type": "done", "sources": [...], "intent": "rag"}
data: {"type": "error", "message": "..."}
```

---

## Intent Routing (Rule-Based)

| Intent | Trigger Keywords | Behaviour |
|--------|-----------------|-----------|
| `rag` | (default) | Standard Q&A with citations |
| `compare` | compare, contrast, differences, vs, both… | Structured comparison prompt |
| `mermaid` | draw, diagram, flow, flowchart, architecture… | Mermaid-only LLM output |

---

## Features

- ✅ PDF/DOCX/TXT ingestion with page-aware chunking (500-800 tokens, 100 overlap)
- ✅ Embedding with `all-MiniLM-L6-v2` (kept in memory)
- ✅ pgvector cosine similarity top-5 retrieval
- ✅ SQL-level page-range pre-filtering
- ✅ ChatSession with auto-generated LLM title
- ✅ Multi-document comparison mode
- ✅ Mermaid diagram generation & rendering
- ✅ Groq (primary) + Ollama (fallback) with streaming
- ✅ Structured citations with page numbers & relevance scores
- ✅ Dark-themed Next.js App Router UI

---

## Environment Variables

### Backend (`backend/.env`)
```
SECRET_KEY=...
DEBUG=True
DB_NAME=research_agent_db
DB_USER=postgres
DB_PASSWORD=postgres
DB_HOST=localhost
DB_PORT=5432
GROQ_API_KEY=gsk_...
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3
CORS_ALLOWED_ORIGINS=http://localhost:3000
```

### Frontend (`frontend/.env.local`)
```
NEXT_PUBLIC_API_URL=http://localhost:8000/api
```
