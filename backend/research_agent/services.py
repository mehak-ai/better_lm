"""
services.py — Core service layer for the Smart Research Agent.

Responsibilities:
  - EmbeddingService  : loads model once, encodes text
  - ParserService     : PDF / DOCX / TXT → plain text with page metadata
  - ChunkingService   : splits text into overlapping token chunks
  - IngestionService  : orchestrates parse → chunk → embed → store
  - RetrievalService  : vector search with optional page-range filter
  - TitleService      : auto-generates session title via LLM
  - IntentRouter      : rule-based query intent classification
  - LLMService        : Groq primary / Ollama fallback, with streaming
"""

import io
import re
import json
import logging
from typing import Generator, List, Dict, Any, Optional, Tuple

import httpx
from django.conf import settings
from django.db import connection

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# EmbeddingService — singleton pattern keeps model in memory
# ---------------------------------------------------------------------------

class EmbeddingService:
    _model = None

    @classmethod
    def get_model(cls):
        if cls._model is None:
            from sentence_transformers import SentenceTransformer
            logger.info("Loading embedding model: %s", settings.EMBEDDING_MODEL_NAME)
            cls._model = SentenceTransformer(settings.EMBEDDING_MODEL_NAME)
        return cls._model

    @classmethod
    def encode(cls, text: str) -> List[float]:
        model = cls.get_model()
        return model.encode(text, normalize_embeddings=True).tolist()

    @classmethod
    def encode_batch(cls, texts: List[str]) -> List[List[float]]:
        model = cls.get_model()
        return model.encode(texts, normalize_embeddings=True, batch_size=32).tolist()


# ---------------------------------------------------------------------------
# ParserService — extract text per page
# ---------------------------------------------------------------------------

class ParserService:
    @staticmethod
    def parse(file_obj, file_type: str) -> List[Dict[str, Any]]:
        """
        Returns list of {"page": int, "text": str} dicts.
        """
        ft = file_type.lower().strip(".")
        if ft == "pdf":
            return ParserService._parse_pdf(file_obj)
        elif ft == "docx":
            return ParserService._parse_docx(file_obj)
        elif ft == "txt":
            return ParserService._parse_txt(file_obj)
        else:
            raise ValueError(f"Unsupported file type: {ft}")

    @staticmethod
    def _clean(text: str) -> str:
        """Removes null bytes and other non-printable junk."""
        # Remove null bytes which crash Postgres
        return text.replace("\x00", "")

    @staticmethod
    def _parse_pdf(file_obj) -> List[Dict]:
        import fitz  # PyMuPDF
        data = file_obj.read() if hasattr(file_obj, "read") else open(file_obj, "rb").read()
        doc = fitz.open(stream=data, filetype="pdf")
        pages = []
        for i, page in enumerate(doc, start=1):
            text = page.get_text("text").strip()
            if text:
                pages.append({"page": i, "text": ParserService._clean(text)})
        doc.close()
        return pages

    @staticmethod
    def _parse_docx(file_obj) -> List[Dict]:
        from docx import Document as DocxDocument
        data = file_obj.read() if hasattr(file_obj, "read") else open(file_obj, "rb").read()
        doc = DocxDocument(io.BytesIO(data))
        # DOCX has no concept of pages; treat every 30 paragraphs as a "page"
        paras = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
        pages = []
        page_size = 30
        for i in range(0, len(paras), page_size):
            chunk_paras = paras[i: i + page_size]
            text = "\n".join(chunk_paras)
            pages.append({"page": i // page_size + 1, "text": ParserService._clean(text)})
        return pages

    @staticmethod
    def _parse_txt(file_obj) -> List[Dict]:
        data = file_obj.read() if hasattr(file_obj, "read") else open(file_obj, "r").read()
        if isinstance(data, bytes):
            data = data.decode("utf-8", errors="replace")
        lines = data.splitlines()
        pages = []
        page_size = 100  # lines per virtual page
        for i in range(0, len(lines), page_size):
            text = "\n".join(lines[i: i + page_size]).strip()
            if text:
                pages.append({"page": i // page_size + 1, "text": ParserService._clean(text)})
        return pages


# ---------------------------------------------------------------------------
# ChunkingService — word-based chunking with overlap
# ---------------------------------------------------------------------------

class ChunkingService:
    TARGET_TOKENS = 650   # midpoint of 500-800
    OVERLAP_TOKENS = 100

    @staticmethod
    def chunk_pages(pages: List[Dict]) -> List[Dict]:
        """
        Takes parsed pages, returns list of:
          {"content": str, "page_number": int, "chunk_index": int}
        """
        # Approximate tokens ≈ words (good enough for chunking)
        chunks = []
        idx = 0
        for page_info in pages:
            words = page_info["text"].split()
            start = 0
            while start < len(words):
                end = start + ChunkingService.TARGET_TOKENS
                chunk_words = words[start:end]
                content = " ".join(chunk_words)
                if content.strip():
                    chunks.append({
                        "content": content,
                        "page_number": page_info["page"],
                        "chunk_index": idx,
                    })
                    idx += 1
                start = end - ChunkingService.OVERLAP_TOKENS
                if start < 0:
                    break
        return chunks


# ---------------------------------------------------------------------------
# IngestionService — parse + chunk + embed + persist
# ---------------------------------------------------------------------------

class IngestionService:
    @staticmethod
    def ingest(document) -> int:
        """
        Full ingestion pipeline for a Document instance.
        Returns number of chunks created.
        """
        from .models import Chunk

        file_path = document.file.path
        file_type = document.file_type

        # 1. Parse
        with open(file_path, "rb") as f:
            pages = ParserService.parse(f, file_type)

        if not pages:
            return 0

        # 2. Chunk
        raw_chunks = ChunkingService.chunk_pages(pages)

        # 3. Embed (batch)
        texts = [c["content"] for c in raw_chunks]
        embeddings = EmbeddingService.encode_batch(texts)

        # 4. Persist in bulk
        from django.db.models import Max
        max_global_idx = Chunk.objects.aggregate(Max("global_index"))["global_index__max"]
        current_global_idx = (max_global_idx or -1) + 1

        chunk_objs = []
        for c_data, emb in zip(raw_chunks, embeddings):
            chunk_objs.append(Chunk(
                document=document,
                content=c_data["content"],
                embedding=emb,
                page_number=c_data["page_number"],
                chunk_index=c_data["chunk_index"],
                global_index=current_global_idx
            ))
            current_global_idx += 1

        Chunk.objects.bulk_create(chunk_objs)

        # Update total pages on document
        max_page = max((p["page"] for p in pages), default=0)
        document.total_pages = max_page
        document.save(update_fields=["total_pages"])

        return len(chunk_objs)


# ---------------------------------------------------------------------------
# RetrievalService — vector search with SQL-level pre-filtering
# ---------------------------------------------------------------------------

class RetrievalService:
    TOP_K = 2

    @staticmethod
    def retrieve(
        query: str,
        document_ids: List[int],
        mode: str = "auto",  # auto | single | compare
        top_k: int = 3,  # Top-K PER DOCUMENT
    ) -> List[Dict]:
        """
        Retrieves chunks strictly by document to support comparison.
        """
        import numpy as np
        from .models import Chunk

        if not document_ids:
            return []

        # 1. Embed query
        query_vec = np.array(EmbeddingService.encode(query))
        query_norm = np.linalg.norm(query_vec)
        if query_norm > 0:
            query_vec = query_vec / query_norm

        results = []

        # 2. Iterate per document to ensure fairness/comparison
        # If mode is single or only 1 doc, loop runs once.
        for doc_id in document_ids:
            # A. Strict Metadata Filter
            qs = Chunk.objects.filter(document_id=doc_id)
            
            # Optimization: fetch only necessary fields
            chunks = list(qs.select_related("document").values(
                "id", "document_id", "document__title", "document__authors", "document__publication_date",
                "content", "page_number", "embedding", "global_index"
            ))

            if not chunks:
                continue

            # B. Vector Search (In-Memory)
            doc_embeddings = np.array([c["embedding"] for c in chunks])
            doc_norms = np.linalg.norm(doc_embeddings, axis=1, keepdims=True)
            doc_norms[doc_norms == 0] = 1.0
            doc_embeddings_norm = doc_embeddings / doc_norms
            
            # Dot product
            scores = np.dot(doc_embeddings_norm, query_vec)

            scored_chunks = []
            for i, chunk in enumerate(chunks):
                chunk["score"] = float(scores[i])
                chunk["chunk_id"] = chunk.pop("id")
                chunk["document_title"] = chunk.pop("document__title")
                # Add extra metadata
                auth = chunk.pop("document__authors", "")
                date = chunk.pop("document__publication_date", "")
                chunk["metadata_str"] = f"{chunk['document_title']} ({date}) {auth}"
                
                del chunk["embedding"]
                scored_chunks.append(chunk)

            # Sort descending and take top_k
            scored_chunks.sort(key=lambda x: x["score"], reverse=True)
            results.extend(scored_chunks[:top_k])

        return results


# ---------------------------------------------------------------------------
# IntentRouter — rule-based MVP router
# ---------------------------------------------------------------------------

MERMAID_KEYWORDS = {"draw", "diagram", "flow", "flowchart", "architecture",
                    "chart", "visualize", "graph", "sequence", "er diagram"}
COMPARE_KEYWORDS = {"compare", "contrast", "difference", "vs", "versus",
                    "both", "agreement", "disagree", "similar", "differ"}


class IntentRouter:
    @staticmethod
    def classify(query: str) -> str:
        """Returns: 'mermaid' | 'compare' | 'rag'"""
        q = query.lower()
        words = set(re.findall(r"\w+", q))
        if words & MERMAID_KEYWORDS:
            return "mermaid"
        if words & COMPARE_KEYWORDS:
            return "compare"
        return "rag"


# ---------------------------------------------------------------------------
# LLMService — Groq primary / Ollama fallback, streaming generator
# ---------------------------------------------------------------------------

class LLMService:
    GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"

    @staticmethod
    def _build_messages(system: str, user: str) -> List[Dict]:
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

    @staticmethod
    def stream(system: str, user: str) -> Generator[str, None, None]:
        """Yields text tokens as they arrive. Falls back to Ollama on Groq error."""
        if not settings.GROQ_API_KEY:
            logger.warning("GROQ_API_KEY is missing/empty! Falling back to Ollama.")
            
        if settings.GROQ_API_KEY:
            yield from LLMService._stream_groq(system, user)
            return
        
        # Only use Ollama if Groq key is NOT set
        yield from LLMService._stream_ollama(system, user)

    @staticmethod
    def _stream_groq(system: str, user: str) -> Generator[str, None, None]:
        headers = {
            "Authorization": f"Bearer {settings.GROQ_API_KEY}",
            "Content-Type": "application/json",
        }
        body = {
            "model": settings.GROQ_MODEL,
            "messages": LLMService._build_messages(system, user),
            "stream": True,
            "temperature": 0.3,
            "max_tokens": 1024,
        }
        with httpx.stream("POST", LLMService.GROQ_URL, json=body,
                          headers=headers, timeout=60) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if line.startswith("data: "):
                    payload = line[6:]
                    if payload.strip() == "[DONE]":
                        break
                    try:
                        data = json.loads(payload)
                        delta = data["choices"][0]["delta"].get("content", "")
                        if delta:
                            yield delta
                    except (json.JSONDecodeError, KeyError, IndexError):
                        continue

    @staticmethod
    def _stream_ollama(system: str, user: str) -> Generator[str, None, None]:
        url = f"{settings.OLLAMA_BASE_URL}/api/chat"
        body = {
            "model": settings.OLLAMA_MODEL,
            "messages": LLMService._build_messages(system, user),
            "stream": True,
        }
        with httpx.stream("POST", url, json=body, timeout=120) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if line.strip():
                    try:
                        data = json.loads(line)
                        delta = data.get("message", {}).get("content", "")
                        if delta:
                            yield delta
                    except json.JSONDecodeError:
                        continue

    @staticmethod
    def complete(system: str, user: str) -> str:
        """Non-streaming completion (used for title generation)."""
        return "".join(LLMService.stream(system, user))


# ---------------------------------------------------------------------------
# TitleService — auto-generate session title from first 1000 tokens
# ---------------------------------------------------------------------------

class TitleService:
    SYSTEM = (
        "You are a concise research-title generator. "
        "Generate a short, descriptive 3-6 word title for the following text. "
        "Output ONLY the title, no quotes, no punctuation at the end."
    )

    @staticmethod
    def generate(text: str) -> str:
        # Take first ~1000 words as context
        preview = " ".join(text.split()[:1000])
        try:
            title = LLMService.complete(TitleService.SYSTEM, preview).strip()
            return title[:200] if title else "Research Session"
        except Exception as e:
            logger.warning("Title generation failed: %s", e)
            return "Research Session"


# ---------------------------------------------------------------------------
# PromptBuilder — builds LLM prompts per intent
# ---------------------------------------------------------------------------

class PromptBuilder:
    @staticmethod
    def build_context(chunks: List[Dict]) -> str:
        # Group by document
        docs = {}
        for c in chunks:
            title = c.get("document_title", "Unknown Doc")
            if title not in docs:
                docs[title] = []
            docs[title].append(c)
        
        parts = []
        for title, doc_chunks in docs.items():
            parts.append(f"Document: {title}")
            for c in doc_chunks:
                parts.append(f"[p.{c['page_number']}] {c['content']}")
            parts.append("") # spacer
            
        return "\n".join(parts)

    @staticmethod
    def rag_prompt(query: str, chunks: List[Dict]) -> Tuple[str, str]:
        context = PromptBuilder.build_context(chunks)
        system = (
            "You are a research assistant. Answer ONLY using the provided context. "
            "If the Context section is empty or does not contain the answer, say 'Information not available in retrieved context.' "
            "Do not use outside knowledge. Do not make up citations. "
            "Cite page numbers as [p. X]."
        )
        user = f"CONTEXT:\n{context}\n\nQUESTION:\n{query}"
        return system, user

    @staticmethod
    def compare_prompt(query: str, chunks: List[Dict]) -> Tuple[str, str]:
        context = PromptBuilder.build_context(chunks)
        system = (
            "You are a research analyst. Compare the provided documents. "
            "STRICT OUTPUT FORMAT:\n"
            "1. **Agreements**: Key shared points.\n"
            "2. **Differences**: Clear distinctions per document.\n"
            "3. **Supporting Evidence**: Cite as [p. X] attached to specific claims.\n"
            "Do NOT use outside knowledge."
        )
        user = f"DOCUMENTS:\n{context}\n\nCOMPARE BASED ON:\n{query}"
        return system, user

    @staticmethod
    def mermaid_prompt(query: str, chunks: List[Dict]) -> Tuple[str, str]:
        context = PromptBuilder.build_context(chunks)
        system = (
            "You are a diagram generator. Output ONLY valid Mermaid.js syntax. "
            "Do NOT include any explanation, markdown fences, or text outside the diagram. "
            "Start directly with the diagram type keyword (e.g. flowchart, sequenceDiagram, erDiagram). "
            "STRICT SYNTAX RULE: For labeled arrows, use `A -->|Label| B`. "
            "Do NOT use `-->|Label|> B` or other variations."
        )
        user = f"Based on this content:\n{context}\n\nGenerate a Mermaid diagram for: {query}"
        return system, user
