"""
views.py — API views for the Smart Research Agent.

Endpoints:
  POST /api/upload/                → ingest documents, create session
  GET  /api/sessions/              → list all chat sessions
  GET  /api/sessions/<id>/         → session detail + documents
  POST /api/sessions/<id>/chat/    → streaming RAG chat (SSE)
  GET  /api/sessions/<id>/messages/→ message history
  DELETE /api/sessions/<id>/       → delete session
"""

import json
import logging

from django.http import StreamingHttpResponse, JsonResponse
from django.views import View
from django.views.decorators.csrf import csrf_exempt
from django.utils.decorators import method_decorator
from rest_framework.parsers import MultiPartParser
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status

from .models import Document, ChatSession, SessionDocument, Message
from .services import (
    IngestionService,
    RetrievalService,
    IntentRouter,
    LLMService,
    PromptBuilder,
    TitleService,
)

logger = logging.getLogger(__name__)

ALLOWED_TYPES = {"pdf", "docx", "txt"}


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def session_to_dict(s: ChatSession) -> dict:
    docs = list(
        s.documents.values("id", "title", "file_type", "uploaded_at", "total_pages")
    )
    return {
        "id": s.id,
        "title": s.title,
        "created_at": s.created_at.isoformat(),
        "documents": docs,
    }


# ---------------------------------------------------------------------------
# Upload & Session Creation
# ---------------------------------------------------------------------------

@method_decorator(csrf_exempt, name="dispatch")
class UploadView(APIView):
    parser_classes = [MultiPartParser]

    def post(self, request):
        files = request.FILES.getlist("files")
        if not files:
            return Response({"error": "No files provided."}, status=400)

        documents = []
        for f in files:
            ext = f.name.rsplit(".", 1)[-1].lower()
            if ext not in ALLOWED_TYPES:
                return Response({"error": f"Unsupported file type: {ext}"}, status=400)

            doc = Document.objects.create(
                title=f.name,
                file=f,
                file_type=ext,
            )
            try:
                chunk_count = IngestionService.ingest(doc)
                logger.info("Ingested %d chunks for doc %d", chunk_count, doc.id)
            except Exception as e:
                doc.delete()
                logger.error("Ingestion failed for %s: %s", f.name, e)
                return Response({"error": f"Ingestion failed: {e}"}, status=500)

            documents.append(doc)

        # Create ChatSession
        session = ChatSession.objects.create(title="Generating title…")
        for doc in documents:
            SessionDocument.objects.create(session=session, document=doc)

        # Auto-generate title from first document
        try:
            from .models import Chunk
            first_chunks = (
                Chunk.objects.filter(document=documents[0])
                .order_by("chunk_index")[:6]
            )
            preview_text = " ".join(c.content for c in first_chunks)
            session.title = TitleService.generate(preview_text)
            session.save(update_fields=["title"])
        except Exception as e:
            logger.warning("Title generation failed: %s", e)
            session.title = documents[0].title.rsplit(".", 1)[0]
            session.save(update_fields=["title"])

        return Response(session_to_dict(session), status=201)


# ---------------------------------------------------------------------------
# Session List / Detail / Delete
# ---------------------------------------------------------------------------

class SessionListView(APIView):
    def get(self, request):
        sessions = ChatSession.objects.prefetch_related("documents").all()
        return Response([session_to_dict(s) for s in sessions])


class SessionDetailView(APIView):
    def get(self, request, session_id):
        try:
            s = ChatSession.objects.prefetch_related("documents").get(pk=session_id)
        except ChatSession.DoesNotExist:
            return Response({"error": "Session not found."}, status=404)
        return Response(session_to_dict(s))

    def delete(self, request, session_id):
        try:
            s = ChatSession.objects.get(pk=session_id)
            s.delete()
        except ChatSession.DoesNotExist:
            pass
        return Response(status=204)


# ---------------------------------------------------------------------------
# Message History
# ---------------------------------------------------------------------------

class MessageListView(APIView):
    def get(self, request, session_id):
        msgs = Message.objects.filter(session_id=session_id)
        data = [
            {
                "id": m.id,
                "role": m.role,
                "content": m.content,
                "sources": m.sources,
                "intent": m.intent,
                "created_at": m.created_at.isoformat(),
            }
            for m in msgs
        ]
        return Response(data)


# ---------------------------------------------------------------------------
# Streaming Chat (SSE)
# ---------------------------------------------------------------------------

@method_decorator(csrf_exempt, name="dispatch")
class ChatView(View):
    """
    POST /api/sessions/<id>/chat/
    Body (JSON):
      {
        "query": "...",
        "document_ids": [1, 2],   // optional subset; defaults to all session docs
        "page_start": 1,          // optional
        "page_end": 10,           // optional
      }

    Returns: text/event-stream (SSE)
      data: {"type":"token","content":"..."}
      data: {"type":"done","sources":[...], "intent":"rag"}
      data: {"type":"error","message":"..."}
    """

    def post(self, request, session_id):
        try:
            body = json.loads(request.body)
        except json.JSONDecodeError:
            return JsonResponse({"error": "Invalid JSON."}, status=400)

        query = body.get("query", "").strip()
        if not query:
            return JsonResponse({"error": "query is required."}, status=400)

        try:
            session = ChatSession.objects.prefetch_related("documents").get(pk=session_id)
        except ChatSession.DoesNotExist:
            return JsonResponse({"error": "Session not found."}, status=404)

        # Determine doc scope
        all_doc_ids = list(session.documents.values_list("id", flat=True))
        document_ids = body.get("document_ids") or all_doc_ids
        document_ids = [d for d in document_ids if d in all_doc_ids]

        page_start = body.get("page_start")
        page_end = body.get("page_end")

        # Save user message
        Message.objects.create(session=session, role="user", content=query)

        def event_stream():
            try:
                # 1. Classify intent
                intent = IntentRouter.classify(query)

                # 2. Retrieve relevant chunks
                chunks = RetrievalService.retrieve(
                    query=query,
                    document_ids=document_ids,
                )

                if not chunks:
                    yield _sse({"type": "token", "content": "No relevant content found for your query."})
                    yield _sse({"type": "done", "sources": [], "intent": intent})
                    return

                # 3. Build prompt per intent
                if intent == "mermaid":
                    system, user_prompt = PromptBuilder.mermaid_prompt(query, chunks)
                elif intent == "compare":
                    system, user_prompt = PromptBuilder.compare_prompt(query, chunks)
                else:
                    system, user_prompt = PromptBuilder.rag_prompt(query, chunks)

                # 4. Stream LLM response
                full_response = []
                for token in LLMService.stream(system, user_prompt):
                    full_response.append(token)
                    yield _sse({"type": "token", "content": token})

                # 5. Build citations
                sources = [
                    {
                        "chunk_id": c["chunk_id"],
                        "document_id": c["document_id"],
                        "document_title": c["document_title"],
                        "page_number": c["page_number"],
                        "score": round(float(c["score"]), 4),
                        "excerpt": c["content"][:200],
                    }
                    for c in chunks
                ]

                # 6. Persist assistant message
                Message.objects.create(
                    session=session,
                    role="assistant",
                    content="".join(full_response),
                    sources=sources,
                    intent=intent,
                )

                yield _sse({"type": "done", "sources": sources, "intent": intent})

            except Exception as e:
                logger.error("Chat error: %s", e, exc_info=True)
                yield _sse({"type": "error", "message": str(e)})

        response = StreamingHttpResponse(event_stream(), content_type="text/event-stream")
        response["Cache-Control"] = "no-cache"
        response["X-Accel-Buffering"] = "no"
        return response


def _sse(data: dict) -> str:
    return f"data: {json.dumps(data)}\n\n"
