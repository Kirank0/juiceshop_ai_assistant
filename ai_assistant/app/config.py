import os
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(REPO_ROOT / ".env.openai")


class Settings:
    def __init__(self) -> None:
        self.openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
        self.openai_model: str = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
        self.juice_shop_url: str = os.getenv("JUICE_SHOP_URL", "http://localhost:3000").rstrip("/")
        self.product_cache_ttl: int = int(os.getenv("PRODUCT_CACHE_TTL", "300"))
        self.allowed_origins: list[str] = [
            origin.strip()
            for origin in os.getenv("ALLOWED_ORIGINS", "http://localhost:3000,http://localhost:4200").split(",")
            if origin.strip()
        ]
        self.max_history_turns: int = int(os.getenv("MAX_HISTORY_TURNS", "10"))
        # RAG / Chroma
        self.chroma_host: str = os.getenv("CHROMA_HOST", "localhost")
        self.chroma_port: int = int(os.getenv("CHROMA_PORT", "8001"))
        self.chroma_collection: str = os.getenv("CHROMA_COLLECTION", "juice_shop_products")
        self.embedding_model: str = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
        self.rag_top_k: int = int(os.getenv("RAG_TOP_K", "6"))
        self.index_on_startup: bool = os.getenv("INDEX_ON_STARTUP", "true").lower() == "true"


@lru_cache
def get_settings() -> Settings:
    return Settings()
