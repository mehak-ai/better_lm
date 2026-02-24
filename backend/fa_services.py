"""
fa_services.py — Core service layer for the FastAPI backend.

Direct port of the original Django services.py with Django ORM calls
replaced by SQLAlchemy session calls.  All LangChain logic is unchanged.
"""

import logging
import re
from pathlib import Path
from typing import List, Dict, Generator

import numpy as np

# LangChain Imports (identical to Django version)
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader, Docx2txtLoader, TextLoader
from langchain_core.documents import Document as LCDocument
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from langchain_community.chat_models import ChatOllama

import config

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# EmbeddingService
# ---------------------------------------------------------------------------

class EmbeddingService:
    _embeddings = None

    @classmethod
    def get_embeddings(cls):
        if cls._embeddings is None:
            logger.info("Loading embedding model: %s", config.EMBEDDING_MODEL_NAME)
            cls._embeddings = HuggingFaceEmbeddings(
                model_name=config.EMBEDDING_MODEL_NAME,
                encode_kwargs={"normalize_embeddings": True},
            )
        return cls._embeddings

    @classmethod
    def encode(cls, text: str) -> List[float]:
        return cls.get_embeddings().embed_query(text)

    @classmethod
    def encode_batch(cls, texts: List[str]) -> List[List[float]]:
        return cls.get_embeddings().embed_documents(texts)


# ---------------------------------------------------------------------------
# IngestionService
# ---------------------------------------------------------------------------

class IngestionService:
    @staticmethod
    def ingest(db, document) -> int:
        """
        Full ingestion pipeline.  `document` is a db_models.Document instance.
        `db` is a SQLAlchemy Session.
        Returns the number of chunks created.
        """
        from db_models import Chunk

        # Build absolute file path from the relative path stored in the DB
        file_path = str(config.MEDIA_ROOT / document.file)
        file_type = document.file_type.lower().strip(".")

        # 1. Load
        try:
            if file_type == "pdf":
                loader = PyMuPDFLoader(file_path)
            elif file_type == "docx":
                loader = Docx2txtLoader(file_path)
            elif file_type == "txt":
                loader = TextLoader(file_path, encoding="utf-8")
            else:
                raise ValueError(f"Unsupported file type: {file_type}")
            raw_docs = loader.load()
        except Exception as e:
            logger.error("Failed to load document %s: %s", document.title, e)
            return 0

        if not raw_docs:
            return 0

        # Strip NUL bytes
        for doc in raw_docs:
            doc.page_content = doc.page_content.replace("\x00", "")

        # 2. Split
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            add_start_index=True,
        )
        splits = splitter.split_documents(raw_docs)
        if not splits:
            return 0

        # 3. Embed (batch)
        texts = [s.page_content for s in splits]
        embeddings = EmbeddingService.encode_batch(texts)

        # 4. Persist
        max_gi = db.query(Chunk.global_index).order_by(Chunk.global_index.desc()).first()
        current_global_idx = ((max_gi[0] or -1) + 1) if max_gi else 0

        chunk_objs = []
        for i, (split, emb) in enumerate(zip(splits, embeddings)):
            page_num = split.metadata.get("page", 0) + 1
            clean = split.page_content.replace("\x00", "")
            chunk_objs.append(
                Chunk(
                    document_id=document.id,
                    content=clean,
                    embedding=emb,
                    page_number=page_num,
                    chunk_index=i,
                    global_index=current_global_idx,
                )
            )
            current_global_idx += 1

        db.bulk_save_objects(chunk_objs)

        # Update total_pages
        max_page = max((c.page_number for c in chunk_objs), default=0)
        document.total_pages = max_page
        db.commit()

        return len(chunk_objs)


# ---------------------------------------------------------------------------
# CustomRetriever
# ---------------------------------------------------------------------------

class CustomRetriever:
    @staticmethod
    def retrieve_documents(
        db,
        query: str,
        document_ids: List[int],
        top_k: int = 4,
    ) -> List[Dict]:
        from db_models import Chunk, Document

        if not document_ids:
            return []

        query_vec = np.array(EmbeddingService.encode(query))
        norm = np.linalg.norm(query_vec)
        if norm > 0:
            query_vec = query_vec / norm

        results = []
        for doc_id in document_ids:
            rows = (
                db.query(
                    Chunk.id,
                    Chunk.document_id,
                    Document.title.label("document_title"),
                    Chunk.content,
                    Chunk.page_number,
                    Chunk.embedding,
                    Chunk.global_index,
                )
                .join(Document, Chunk.document_id == Document.id)
                .filter(Chunk.document_id == doc_id)
                .all()
            )
            if not rows:
                continue

            embs = np.array([r.embedding for r in rows])
            norms = np.linalg.norm(embs, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            embs_norm = embs / norms
            scores = np.dot(embs_norm, query_vec)

            scored = []
            for i, r in enumerate(rows):
                scored.append(
                    {
                        "chunk_id":       r.id,
                        "document_id":    r.document_id,
                        "document_title": r.document_title,
                        "content":        r.content,
                        "page_number":    r.page_number,
                        "global_index":   r.global_index,
                        "score":          float(scores[i]),
                    }
                )
            scored.sort(key=lambda x: x["score"], reverse=True)
            results.extend(scored[:top_k])

        results.sort(key=lambda x: x["score"], reverse=True)
        return results

    @staticmethod
    def to_langchain_docs(scored_chunks: List[Dict]) -> List[LCDocument]:
        docs = []
        for c in scored_chunks:
            docs.append(
                LCDocument(
                    page_content=c["content"],
                    metadata={
                        "score":          c.get("score"),
                        "document_title": c.get("document_title"),
                        "page_number":    c.get("page_number"),
                        "document_id":    c.get("document_id"),
                    },
                )
            )
        return docs


# ---------------------------------------------------------------------------
# RAGService
# ---------------------------------------------------------------------------

class RAGService:
    @staticmethod
    def get_llm():
        if config.GROQ_API_KEY:
            return ChatGroq(
                temperature=0.3,
                model_name=config.GROQ_MODEL,
                api_key=config.GROQ_API_KEY,
                streaming=True,
            )
        return ChatOllama(
            model=config.OLLAMA_MODEL,
            temperature=0.3,
            streaming=True,
        )

    @staticmethod
    def stream_response(
        query: str,
        docs: List[LCDocument],
        intent: str = "rag",
        history: List[dict] = None,
    ) -> Generator[str, None, None]:
        llm = RAGService.get_llm()

        # Build context grouped by document
        doc_map: Dict[str, List[LCDocument]] = {}
        for d in docs:
            title = d.metadata.get("document_title", "Unknown")
            doc_map.setdefault(title, []).append(d)

        context_parts = []
        for title, d_list in doc_map.items():
            context_parts.append(f"Document: {title}")
            for d in d_list:
                page = d.metadata.get("page_number", "?")
                context_parts.append(f"[p.{page}] {d.page_content}")
            context_parts.append("")
        context_str = "\n".join(context_parts)

        messages = []
        if intent == "mermaid":
            messages.append((
                "system",
                "You are a diagram generator. Output ONLY valid Mermaid.js syntax. "
                "Do NOT include explanation, markdown fences, or text. "
                "Start with the diagram type keyword (e.g. flowchart, sequenceDiagram).",
            ))
            if history:
                for h in history:
                    messages.append((h["role"], h["content"]))
            messages.append(("user", "Based on:\n{context}\n\nGenerate a Mermaid diagram for: {query}"))

        elif intent == "compare":
            messages.append((
                "system",
                "You are a helpful research analyst. Compare the provided documents clearly. "
                "Output Format:\n1. **Agreements**\n2. **Differences**\n3. **Supporting Evidence** (Cite [p. X])\n"
                "Do NOT use outside knowledge.",
            ))
            if history:
                for h in history:
                    messages.append((h["role"], h["content"]))
            messages.append(("user", "DOCUMENTS:\n{context}\n\nCOMPARE:\n{query}"))

        else:
            messages.append((
                "system",
                "You are a friendly and helpful research assistant. "
                "You are having a conversation with the user about their documents. "
                "Answer their questions using ONLY the provided context. "
                "If the context doesn't contain the answer, say 'I couldn't find that in the documents'. "
                "Cite your sources as [p. X] where appropriate. "
                "Be conversational but precise.",
            ))
            if history:
                for h in history:
                    messages.append((h["role"], h["content"]))
            messages.append(("user", "CONTEXT:\n{context}\n\nQUESTION:\n{query}"))

        prompt = ChatPromptTemplate.from_messages(messages)
        chain = prompt | llm | StrOutputParser()
        return chain.stream({"context": context_str, "query": query})


# ---------------------------------------------------------------------------
# IntentRouter
# ---------------------------------------------------------------------------

class IntentRouter:
    MERMAID_KEYWORDS = {"draw", "diagram", "flow", "flowchart", "architecture", "chart", "visualize", "graph"}
    COMPARE_KEYWORDS = {"compare", "contrast", "difference", "vs", "versus", "both", "agreement", "disagree"}

    @staticmethod
    def classify(query: str) -> str:
        words = set(re.findall(r"\w+", query.lower()))
        if words & IntentRouter.MERMAID_KEYWORDS:
            return "mermaid"
        if words & IntentRouter.COMPARE_KEYWORDS:
            return "compare"
        return "rag"


# ---------------------------------------------------------------------------
# TitleService
# ---------------------------------------------------------------------------

class TitleService:
    @staticmethod
    def generate(text: str) -> str:
        preview = " ".join(text.split()[:1000])
        llm = RAGService.get_llm()
        prompt = ChatPromptTemplate.from_messages([
            ("system", "Generate a short 3-6 word title. Output ONLY the title."),
            ("user", "{text}"),
        ])
        chain = prompt | llm | StrOutputParser()
        try:
            return chain.invoke({"text": preview}).strip()[:200]
        except Exception:
            return "Research Session"
