"""
services.py — Core service layer for the Smart Research Agent using LangChain.

Responsibilities:
  - EmbeddingService  : LangChain HuggingFaceEmbeddings wrapper
  - IngestionService  : LangChain Loaders & Splitters -> Chunk model
  - CustomRetriever   : Manual vector search returning LangChain Documents
  - RAGService        : LangChain Chat Models & Prompts
  - TitleService      : Session title generation
  - IntentRouter      : Query classification
"""

import logging
import json
import re
from typing import List, Dict, Any, Generator, Tuple
from django.conf import settings
from django.db import connection
from django.db.models import Max

# LangChain Imports
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader, Docx2txtLoader, TextLoader
from langchain_core.documents import Document as LCDocument
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from langchain_community.chat_models import ChatOllama

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# EmbeddingService — Uses LangChain HuggingFaceEmbeddings
# ---------------------------------------------------------------------------

class EmbeddingService:
    _embeddings = None

    @classmethod
    def get_embeddings(cls):
        if cls._embeddings is None:
            logger.info("Loading embedding model: %s", settings.EMBEDDING_MODEL_NAME)
            cls._embeddings = HuggingFaceEmbeddings(
                model_name=settings.EMBEDDING_MODEL_NAME,
                encode_kwargs={'normalize_embeddings': True}
            )
        return cls._embeddings

    @classmethod
    def encode(cls, text: str) -> List[float]:
        return cls.get_embeddings().embed_query(text)

    @classmethod
    def encode_batch(cls, texts: List[str]) -> List[List[float]]:
        return cls.get_embeddings().embed_documents(texts)


# ---------------------------------------------------------------------------
# IngestionService — LangChain Loaders -> Splitters -> DB
# ---------------------------------------------------------------------------

class IngestionService:
    @staticmethod
    def ingest(document) -> int:
        """
        Full ingestion pipeline using LangChain loaders/splitters.
        Returns number of chunks created.
        """
        from .models import Chunk

        file_path = document.file.path
        file_type = document.file_type.lower().strip(".")

        # 1. Load Documents
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
            logger.error(f"Failed to load document {document.title}: {e}")
            return 0

        if not raw_docs:
            return 0

        # Strip NUL bytes — PostgreSQL text columns cannot store \x00
        for doc in raw_docs:
            doc.page_content = doc.page_content.replace("\x00", "")

        # 2. Split Text
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            add_start_index=True,
        )
        splits = text_splitter.split_documents(raw_docs)

        if not splits:
            return 0

        # 3. Embed (Batch)
        texts = [doc.page_content for doc in splits]
        embeddings = EmbeddingService.encode_batch(texts)

        # 4. Persist
        max_global_idx = Chunk.objects.aggregate(Max("global_index"))["global_index__max"]
        current_global_idx = (max_global_idx or -1) + 1

        chunk_objs = []
        for i, (split, emb) in enumerate(zip(splits, embeddings)):
            # Extract page number if available (PyMuPDF / Docx vs others)
            # PyMuPDF puts 'page' in metadata (0-indexed usually, but depends on loader)
            # We'll default to 1 if not present.
            page_num = split.metadata.get("page", 0) + 1
            
            clean_content = split.page_content.replace("\x00", "")
            chunk_objs.append(Chunk(
                document=document,
                content=clean_content,
                embedding=emb,
                page_number=page_num,
                chunk_index=i,
                global_index=current_global_idx
            ))
            current_global_idx += 1

        Chunk.objects.bulk_create(chunk_objs)

        # Update total pages
        max_page = max((c.page_number for c in chunk_objs), default=0)
        document.total_pages = max_page
        document.save(update_fields=["total_pages"])

        return len(chunk_objs)


# ---------------------------------------------------------------------------
# CustomRetriever — Manual Vector Search -> LangChain Documents
# ---------------------------------------------------------------------------

class CustomRetriever:
    @staticmethod
    def retrieve_documents(query: str, document_ids: List[int], top_k: int = 4) -> List[Dict]:
        """
        Performs vector search per document (to support fair comparison) 
        and returns list of dicts with score/metadata.
        """
        import numpy as np
        from .models import Chunk

        if not document_ids:
            return []

        # Embed query
        query_vec = np.array(EmbeddingService.encode(query))
        query_norm = np.linalg.norm(query_vec)
        if query_norm > 0:
            query_vec = query_vec / query_norm

        results = []

        # Iterate per document
        for doc_id in document_ids:
            qs = Chunk.objects.filter(document_id=doc_id)
            chunks = list(qs.select_related("document").values(
                "id", "document_id", "document__title", "document__authors", 
                "document__publication_date", "content", "page_number", 
                "embedding", "global_index"
            ))

            if not chunks:
                continue

            doc_embeddings = np.array([c["embedding"] for c in chunks])
            doc_norms = np.linalg.norm(doc_embeddings, axis=1, keepdims=True)
            doc_norms[doc_norms == 0] = 1.0
            doc_embeddings_norm = doc_embeddings / doc_norms
            
            scores = np.dot(doc_embeddings_norm, query_vec)

            scored_chunks = []
            for i, chunk in enumerate(chunks):
                chunk["score"] = float(scores[i])
                chunk["chunk_id"] = chunk.pop("id")
                chunk["document_title"] = chunk.pop("document__title")
                del chunk["embedding"]
                scored_chunks.append(chunk)

            scored_chunks.sort(key=lambda x: x["score"], reverse=True)
            results.extend(scored_chunks[:top_k])
            
        # Sort global results by score is usually good, but for comparison 
        # we might want to keep them grouped. 
        # Existing logic returned flat list, we'll sort gently overall.
        results.sort(key=lambda x: x["score"], reverse=True)
        return results

    @staticmethod
    def to_langchain_docs(scored_chunks: List[Dict]) -> List[LCDocument]:
        docs = []
        for c in scored_chunks:
            meta = {
               "score": c.get("score"),
               "document_title": c.get("document_title"),
               "page_number": c.get("page_number"),
               "document_id": c.get("document_id")
            }
            docs.append(LCDocument(page_content=c["content"], metadata=meta))
        return docs


# ---------------------------------------------------------------------------
# RAGService — LangChain Generation
# ---------------------------------------------------------------------------

class RAGService:
    @staticmethod
    def get_llm():
        if settings.GROQ_API_KEY:
            return ChatGroq(
                temperature=0.3, 
                model_name=settings.GROQ_MODEL, 
                api_key=settings.GROQ_API_KEY,
                streaming=True
            )
        else:
            return ChatOllama(
                model=settings.OLLAMA_MODEL,
                temperature=0.3,
                streaming=True
            )

    @staticmethod
    def stream_response(query: str, docs: List[LCDocument], intent: str = "rag", history: List[dict] = None) -> Generator[str, None, None]:
        llm = RAGService.get_llm()
        
        # Build Context String
        context_parts = []
        # Group by document for cleaner context
        doc_map = {}
        for d in docs:
            title = d.metadata.get("document_title", "Unknown")
            if title not in doc_map:
                doc_map[title] = []
            doc_map[title].append(d)
        
        for title, d_list in doc_map.items():
            context_parts.append(f"Document: {title}")
            for d in d_list:
                page = d.metadata.get("page_number", "?")
                context_parts.append(f"[p.{page}] {d.page_content}")
            context_parts.append("")
        
        context_str = "\n".join(context_parts)

        # Select Prompt
        messages = []
        
        if intent == "mermaid":
            system = (
                "You are a diagram generator. Output ONLY valid Mermaid.js syntax. "
                "Do NOT include explanation, markdown fences, or text. "
                "Start with the diagram type keyword (e.g. flowchart, sequenceDiagram)."
            )
            messages.append(("system", system))
            # History is less useful for diagrams but could provide context
            if history:
                for h in history:
                    messages.append((h["role"], h["content"]))
            messages.append(("user", "Based on:\n{context}\n\nGenerate a Mermaid diagram for: {query}"))

        elif intent == "compare":
            system = (
                "You are a helpful research analyst. Compare the provided documents clearly. "
                "Output Format:\n"
                "1. **Agreements**\n2. **Differences**\n3. **Supporting Evidence** (Cite [p. X])\n"
                "Do NOT use outside knowledge."
            )
            messages.append(("system", system))
            if history:
                for h in history:
                    messages.append((h["role"], h["content"]))
            messages.append(("user", "DOCUMENTS:\n{context}\n\nCOMPARE:\n{query}"))

        else:
            # Friendly / RAG
            system = (
                "You are a friendly and helpful research assistant. "
                "You are having a conversation with the user about their documents. "
                "Answer their questions using ONLY the provided context. "
                "If the context doesn't contain the answer, say 'I couldn't find that in the documents'. "
                "Cite your sources as [p. X] where appropriate. "
                "Be conversational but precise."
            )
            messages.append(("system", system))
            
            # Inject history
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
        q = query.lower()
        words = set(re.findall(r"\w+", q))
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
            ("user", "{text}")
        ])
        chain = prompt | llm | StrOutputParser()
        try:
            return chain.invoke({"text": preview}).strip()[:200]
        except:
             return "Research Session"
