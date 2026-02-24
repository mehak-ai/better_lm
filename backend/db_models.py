"""
db_models.py — SQLAlchemy ORM models.

These map to the SAME tables that the Django migrations already created,
so the existing data is fully preserved. Table/column names match exactly.
"""
from datetime import datetime, date
from typing import List, Optional

from sqlalchemy import (
    BigInteger, Boolean, Column, DateTime, ForeignKey,
    Integer, String, Text, UniqueConstraint, JSON,
)
from sqlalchemy.orm import relationship

from database import Base


class Document(Base):
    __tablename__ = "research_agent_document"

    id           = Column(BigInteger, primary_key=True, autoincrement=True)
    title        = Column(String(512), nullable=False)
    authors      = Column(String(1024), nullable=False, default="")
    file         = Column(String(255), nullable=False)   # relative path stored by Django FileField
    file_type    = Column(String(10), nullable=False)
    file_hash    = Column(String(64), unique=True, nullable=True)
    uploaded_at  = Column(DateTime, default=datetime.utcnow, nullable=False)
    total_pages  = Column(Integer, default=0)

    chunks    = relationship("Chunk",           back_populates="document", cascade="all, delete-orphan")
    sessions  = relationship("SessionDocument", back_populates="document", cascade="all, delete-orphan")


class Chunk(Base):
    __tablename__ = "research_agent_chunk"

    id           = Column(BigInteger, primary_key=True, autoincrement=True)
    document_id  = Column(BigInteger, ForeignKey("research_agent_document.id", ondelete="CASCADE"), nullable=False)
    content      = Column(Text, nullable=False)
    embedding    = Column(JSON, nullable=False)   # list of floats
    page_number  = Column(Integer, default=1)
    chunk_index  = Column(Integer, default=0)
    global_index = Column(Integer, unique=True, nullable=True)
    chunk_type   = Column(String(20), default="paragraph")  # heading|paragraph|table|image|caption

    document = relationship("Document", back_populates="chunks")


class ChatSession(Base):
    __tablename__ = "research_agent_chatsession"

    id         = Column(BigInteger, primary_key=True, autoincrement=True)
    title      = Column(String(256), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    session_documents = relationship("SessionDocument", back_populates="session", cascade="all, delete-orphan")
    messages          = relationship("Message",         back_populates="session", cascade="all, delete-orphan",
                                     order_by="Message.created_at")


class SessionDocument(Base):
    __tablename__ = "research_agent_sessiondocument"
    __table_args__ = (UniqueConstraint("session_id", "document_id"),)

    id          = Column(BigInteger, primary_key=True, autoincrement=True)
    session_id  = Column(BigInteger, ForeignKey("research_agent_chatsession.id", ondelete="CASCADE"), nullable=False)
    document_id = Column(BigInteger, ForeignKey("research_agent_document.id",    ondelete="CASCADE"), nullable=False)
    added_at    = Column(DateTime, default=datetime.utcnow, nullable=False)

    session  = relationship("ChatSession", back_populates="session_documents")
    document = relationship("Document",    back_populates="sessions")


class Message(Base):
    __tablename__ = "research_agent_message"

    id         = Column(BigInteger, primary_key=True, autoincrement=True)
    session_id = Column(BigInteger, ForeignKey("research_agent_chatsession.id", ondelete="CASCADE"), nullable=False)
    role       = Column(String(10), nullable=False)   # user | assistant
    content    = Column(Text, nullable=False)
    sources    = Column(JSON, default=list)
    intent     = Column(String(20), default="rag")
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    session = relationship("ChatSession", back_populates="messages")
