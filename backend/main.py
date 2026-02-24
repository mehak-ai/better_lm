"""
main.py — FastAPI application.

Endpoints (identical contract to Django version):
  POST   /api/upload/                   upload files + create/join session
  POST   /api/sessions/create/          create empty session
  GET    /api/sessions/                 list all sessions
  GET    /api/sessions/{id}/            session detail
  DELETE /api/sessions/{id}/            delete session
  POST   /api/sessions/{id}/chat/       streaming SSE chat
  GET    /api/sessions/{id}/messages/   message history
  GET    /api/voice/token/              Deepgram API key
  POST   /api/voice/tts/               Deepgram TTS proxy
"""

import hashlib
import json
import logging
import shutil
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import httpx
from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sqlalchemy.orm import Session

import config
from database import get_db
from db_models import ChatSession, Chunk, Document, Message, SessionDocument
from fa_services import (
    CustomRetriever,
    IngestionService,
    IntentRouter,
    RAGService,
    TitleService,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# App bootstrap
# ---------------------------------------------------------------------------

app = FastAPI(title="Smart Research Agent API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=config.CORS_ORIGINS,
    allow_origin_regex=r"https://.*\.trycloudflare\.com",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve uploaded media files
app.mount("/media", StaticFiles(directory=str(config.MEDIA_ROOT)), name="media")

ALLOWED_TYPES = {"pdf", "docx", "txt"}


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------

class CreateSessionRequest(BaseModel):
    title: str = "New Research Session"


class ChatRequest(BaseModel):
    query: str
    document_ids: Optional[List[int]] = None
    page_start: Optional[int] = None
    page_end: Optional[int] = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def session_to_dict(s: ChatSession) -> dict:
    docs = []
    for sd in s.session_documents:
        d = sd.document
        docs.append({
            "id":          d.id,
            "title":       d.title,
            "file_type":   d.file_type,
            "uploaded_at": d.uploaded_at.isoformat() if d.uploaded_at else None,
            "total_pages": d.total_pages,
        })
    return {
        "id":         s.id,
        "title":      s.title,
        "created_at": s.created_at.isoformat() if s.created_at else None,
        "documents":  docs,
    }


def _sse(data: dict) -> str:
    return f"data: {json.dumps(data)}\n\n"


def save_upload(file: UploadFile) -> tuple[str, str]:
    """Save upload to media/documents/ and return (relative_path, file_hash)."""
    dest_dir = config.MEDIA_ROOT / "documents"
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / file.filename

    hasher = hashlib.sha256()
    content = file.file.read()
    hasher.update(content)
    file_hash = hasher.hexdigest()

    dest.write_bytes(content)
    # Store Django-style relative path (relative to MEDIA_ROOT)
    return f"documents/{file.filename}", file_hash


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/api/")
def health():
    return {"status": "ok", "framework": "FastAPI"}


# --- Create empty session ---------------------------------------------------

@app.post("/api/sessions/create", status_code=201)
@app.post("/api/sessions/create/", status_code=201)
def create_session(body: CreateSessionRequest, db: Session = Depends(get_db)):
    session = ChatSession(title=body.title, created_at=datetime.utcnow())
    db.add(session)
    db.commit()
    db.refresh(session)
    return session_to_dict(session)


# --- Upload documents -------------------------------------------------------

@app.post("/api/upload", status_code=201)
@app.post("/api/upload/", status_code=201)
async def upload(
    files: List[UploadFile] = File(...),
    session_id: Optional[str] = Form(None),
    db: Session = Depends(get_db),
):
    if not files:
        raise HTTPException(400, "No files provided.")

    documents: List[Document] = []

    for f in files:
        ext = f.filename.rsplit(".", 1)[-1].lower() if "." in f.filename else ""
        if ext not in ALLOWED_TYPES:
            raise HTTPException(400, f"Unsupported file type: {ext}")

        rel_path, file_hash = save_upload(f)

        # De-duplicate by hash
        doc = db.query(Document).filter(Document.file_hash == file_hash).first()
        if not doc:
            doc = Document(
                title=f.filename,
                file=rel_path,
                file_type=ext,
                file_hash=file_hash,
                uploaded_at=datetime.utcnow(),
            )
            db.add(doc)
            db.commit()
            db.refresh(doc)
            try:
                chunk_count = IngestionService.ingest(db, doc)
                logger.info("Ingested %d chunks for doc %d", chunk_count, doc.id)
            except Exception as e:
                db.delete(doc)
                db.commit()
                logger.error("Ingestion failed for %s: %s", f.filename, e)
                raise HTTPException(500, f"Ingestion failed: {e}")
        else:
            logger.info("Document already exists: %s", doc.title)

        documents.append(doc)

    # Get or create session
    if session_id:
        session = db.query(ChatSession).filter(ChatSession.id == int(session_id)).first()
        if not session:
            raise HTTPException(404, "Session not found.")
    else:
        session = ChatSession(title="Generating title…", created_at=datetime.utcnow())
        db.add(session)
        db.commit()
        db.refresh(session)

    # Link documents to session
    for doc in documents:
        existing = (
            db.query(SessionDocument)
            .filter_by(session_id=session.id, document_id=doc.id)
            .first()
        )
        if not existing:
            db.add(SessionDocument(session_id=session.id, document_id=doc.id, added_at=datetime.utcnow()))
    db.commit()

    # Auto-generate title for new session
    if not session_id:
        try:
            first_chunks = (
                db.query(Chunk)
                .filter(Chunk.document_id == documents[0].id)
                .order_by(Chunk.chunk_index)
                .limit(6)
                .all()
            )
            preview = " ".join(c.content for c in first_chunks)
            session.title = TitleService.generate(preview)
        except Exception as e:
            logger.warning("Title generation failed: %s", e)
            session.title = documents[0].title.rsplit(".", 1)[0]
        db.commit()

    db.refresh(session)
    return session_to_dict(session)


# --- Session list -----------------------------------------------------------

@app.get("/api/sessions/")
@app.get("/api/sessions")
def list_sessions(db: Session = Depends(get_db)):
    sessions = (
        db.query(ChatSession)
        .order_by(ChatSession.created_at.desc())
        .all()
    )
    return [session_to_dict(s) for s in sessions]


# --- Session detail / delete ------------------------------------------------

@app.get("/api/sessions/{session_id}/")
@app.get("/api/sessions/{session_id}")
def get_session(session_id: int, db: Session = Depends(get_db)):
    s = db.query(ChatSession).filter(ChatSession.id == session_id).first()
    if not s:
        raise HTTPException(404, "Session not found.")
    return session_to_dict(s)


@app.delete("/api/sessions/{session_id}/", status_code=204)
@app.delete("/api/sessions/{session_id}", status_code=204)
def delete_session(session_id: int, db: Session = Depends(get_db)):
    s = db.query(ChatSession).filter(ChatSession.id == session_id).first()
    if s:
        db.delete(s)
        db.commit()
    return


# --- Message history --------------------------------------------------------

@app.get("/api/sessions/{session_id}/messages")
@app.get("/api/sessions/{session_id}/messages/")
def get_messages(session_id: int, db: Session = Depends(get_db)):
    msgs = (
        db.query(Message)
        .filter(Message.session_id == session_id)
        .order_by(Message.created_at)
        .all()
    )
    return [
        {
            "id":         m.id,
            "role":       m.role,
            "content":    m.content,
            "sources":    m.sources,
            "intent":     m.intent,
            "created_at": m.created_at.isoformat() if m.created_at else None,
        }
        for m in msgs
    ]


# --- Streaming chat (SSE) ---------------------------------------------------

@app.post("/api/sessions/{session_id}/chat")
@app.post("/api/sessions/{session_id}/chat/")
def chat(session_id: int, body: ChatRequest, db: Session = Depends(get_db)):
    query = body.query.strip()
    if not query:
        raise HTTPException(400, "query is required.")

    session = db.query(ChatSession).filter(ChatSession.id == session_id).first()
    if not session:
        raise HTTPException(404, "Session not found.")

    all_doc_ids = [sd.document_id for sd in session.session_documents]
    document_ids = body.document_ids or all_doc_ids
    document_ids = [d for d in document_ids if d in all_doc_ids]

    # Save user message
    user_msg = Message(
        session_id=session_id, role="user", content=query, created_at=datetime.utcnow()
    )
    db.add(user_msg)
    db.commit()

    # Snapshot history before streaming (db session reuse in generator is tricky)
    history_rows = (
        db.query(Message)
        .filter(Message.session_id == session_id)
        .order_by(Message.created_at.desc())
        .limit(6)
        .all()
    )
    history = []
    for m in reversed(history_rows):
        if m.content:
            history.append({"role": m.role if m.role == "user" else "assistant", "content": m.content})
    if history and history[-1]["role"] == "user" and history[-1]["content"] == query:
        history.pop()

    chunks_data = CustomRetriever.retrieve_documents(db, query, document_ids)
    lc_docs = CustomRetriever.to_langchain_docs(chunks_data)

    # We need a fresh session for the generator (avoid threading issues)
    from database import SessionLocal

    def event_stream():
        gen_db = SessionLocal()
        try:
            if not chunks_data:
                yield _sse({"type": "token", "content": "No relevant content found for your query."})
                yield _sse({"type": "done", "sources": [], "intent": "rag"})
                return

            intent = IntentRouter.classify(query)
            full_response = []

            for token in RAGService.stream_response(query, lc_docs, intent, history=history):
                full_response.append(token)
                yield _sse({"type": "token", "content": token})

            sources = [
                {
                    "chunk_id":       c["chunk_id"],
                    "document_id":    c["document_id"],
                    "document_title": c["document_title"],
                    "page_number":    c["page_number"],
                    "score":          round(float(c["score"]), 4),
                    "excerpt":        c["content"][:200],
                }
                for c in chunks_data
            ]

            asst_msg = Message(
                session_id=session_id,
                role="assistant",
                content="".join(full_response),
                sources=sources,
                intent=intent,
                created_at=datetime.utcnow(),
            )
            gen_db.add(asst_msg)
            gen_db.commit()

            yield _sse({"type": "done", "sources": sources, "intent": intent})

        except Exception as e:
            logger.error("Chat error: %s", e, exc_info=True)
            yield _sse({"type": "error", "message": str(e)})
        finally:
            gen_db.close()

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ---------------------------------------------------------------------------
# Voice Assistant (Deepgram)
# ---------------------------------------------------------------------------

@app.get("/api/voice/token")
@app.get("/api/voice/token/")
def voice_token():
    key = config.DEEPGRAM_API_KEY
    if not key:
        raise HTTPException(503, "DEEPGRAM_API_KEY not configured.")
    return {"key": key}


@app.post("/api/voice/tts")
@app.post("/api/voice/tts/")
async def voice_tts(request: Request):
    body = await request.json()
    text = (body.get("text") or "").strip()
    if not text:
        raise HTTPException(400, "text is required.")

    key = config.DEEPGRAM_API_KEY
    if not key:
        raise HTTPException(503, "DEEPGRAM_API_KEY not configured.")

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
                "https://api.deepgram.com/v1/speak?model=aura-asteria-en",
                headers={"Authorization": f"Token {key}", "Content-Type": "application/json"},
                json={"text": text},
            )
            resp.raise_for_status()
            content_type = resp.headers.get("content-type", "audio/mpeg")
            return StreamingResponse(
                iter([resp.content]),
                media_type=content_type,
                headers={"Content-Disposition": "inline"},
            )
    except httpx.HTTPStatusError as e:
        raise HTTPException(502, f"Deepgram TTS failed: {e.response.status_code}")
    except Exception as e:
        raise HTTPException(500, str(e))
