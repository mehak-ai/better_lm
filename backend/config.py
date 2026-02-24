"""
config.py — Application settings for the FastAPI backend.
Reads from the same .env file that the Django backend used.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(dotenv_path=BASE_DIR / ".env", override=True)

# -------------------------------------------------------------------
# Database
# -------------------------------------------------------------------
DB_NAME     = os.getenv("DB_NAME", "research_agent_db")
DB_USER     = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "")
DB_HOST     = os.getenv("DB_HOST", "localhost")
DB_PORT     = os.getenv("DB_PORT", "5432")

# Railway provides DATABASE_URL directly — use it if available
_raw_url = os.getenv("DATABASE_URL", "")
if _raw_url:
    # Railway uses postgres:// prefix; SQLAlchemy needs postgresql+psycopg2://
    DATABASE_URL = _raw_url.replace("postgres://", "postgresql+psycopg2://", 1)
else:
    DATABASE_URL = (
        f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    )

# -------------------------------------------------------------------
# Media / File storage
# -------------------------------------------------------------------
MEDIA_ROOT = BASE_DIR / "media"
MEDIA_ROOT.mkdir(parents=True, exist_ok=True)

# -------------------------------------------------------------------
# LLM / Embedding
# -------------------------------------------------------------------
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
GROQ_API_KEY         = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL           = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
OLLAMA_BASE_URL      = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL         = os.getenv("OLLAMA_MODEL", "llama3")
DEEPGRAM_API_KEY     = os.getenv("DEEPGRAM_API_KEY", "")

# -------------------------------------------------------------------
# CORS
# -------------------------------------------------------------------
CORS_ORIGINS = os.getenv("CORS_ALLOWED_ORIGINS", "http://localhost:3000").split(",")
