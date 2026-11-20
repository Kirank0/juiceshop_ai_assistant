"""Retrieval-Augmented Generation over Juice Shop product data stored in Chroma.

Each product and each customer review becomes one document. Embeddings are created with
OpenAI and stored in a Chroma server running in Docker (see docker-compose.yml).
"""
import asyncio
import hashlib
import logging
from dataclasses import dataclass, field

import chromadb
from chromadb.api.models.Collection import Collection

from . import assistant
from .config import get_settings
from .products import Product, Review, get_products, get_reviews

logger = logging.getLogger("ai_assistant.rag")

EMBEDDING_BATCH_SIZE = 100

_collection: Collection | None = None


class VectorStoreUnavailableError(Exception):
    """Raised when Chroma cannot be reached."""


@dataclass(frozen=True)
class Document:
    id: str
    text: str
    metadata: dict = field(default_factory=dict)


@dataclass(frozen=True)
class RetrievedChunk:
    text: str
    metadata: dict
    distance: float


@dataclass
class IndexResult:
    total: int
    embedded: int
    unchanged: int
    deleted: int


# ---------- Chroma ----------

def get_collection() -> Collection:
    """Connect to Chroma (lazily) and return the product collection."""
    global _collection
    if _collection is None:
        settings = get_settings()
        try:
            client = chromadb.HttpClient(host=settings.chroma_host, port=settings.chroma_port)
            _collection = client.get_or_create_collection(
                name=settings.chroma_collection,
                configuration={"hnsw": {"space": "cosine"}},
                embedding_function=None,  # embeddings are computed with OpenAI, not by Chroma
            )
        except Exception as err:
            raise VectorStoreUnavailableError(
                f"Could not connect to Chroma at {settings.chroma_host}:{settings.chroma_port}. "
                "Is the Docker container running? (docker compose up -d)"
            ) from err
    return _collection


def reset_connection() -> None:
    global _collection
    _collection = None


async def chroma_status() -> dict:
    try:
        collection = await asyncio.to_thread(get_collection)
        return {"status": "ok", "documents": await asyncio.to_thread(collection.count)}
    except Exception as err:
        reset_connection()
        return {"status": "unavailable", "documents": 0, "detail": str(err)}


# ---------- Documents ----------

def _hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def build_documents(products: list[Product], reviews: list[Review]) -> list[Document]:
    names = {p.id: p.name for p in products}
    documents = []
    for p in products:
        deluxe = f"${p.deluxe_price:.2f}" if p.deluxe_price is not None else "not available"
        text = (
            f"Product: {p.name}\n"
            f"Description: {p.description}\n"
            f"Price: ${p.price:.2f}\n"
            f"Deluxe member price: {deluxe}"
        )
        documents.append(Document(
            id=f"product-{p.id}",
            text=text,
            metadata={"type": "product", "product_id": p.id, "product_name": p.name, "hash": _hash(text)},
        ))
    seen_review_ids: set[str] = set()
    for r in reviews:
        review_id = f"review-{r.product_id}-{_hash(r.message)[:16]}"  # stable across Juice Shop restarts
        if r.product_id not in names or review_id in seen_review_ids:
            continue
        seen_review_ids.add(review_id)
        text = f"Customer review of {names[r.product_id]}: {r.message}\nLikes: {r.likes}"
        documents.append(Document(
            id=review_id,
            text=text,
            metadata={
                "type": "review",
                "product_id": r.product_id,
                "product_name": names[r.product_id],
                "hash": _hash(text),
            },
        ))
    return documents


# ---------- Embeddings ----------

async def embed(texts: list[str]) -> list[list[float]]:
    settings = get_settings()
    vectors: list[list[float]] = []
    for start in range(0, len(texts), EMBEDDING_BATCH_SIZE):
        batch = texts[start:start + EMBEDDING_BATCH_SIZE]
        response = await assistant.get_client().embeddings.create(model=settings.embedding_model, input=batch)
        vectors.extend(item.embedding for item in sorted(response.data, key=lambda d: d.index))
    return vectors


# ---------- Indexing ----------

async def index_documents(documents: list[Document]) -> IndexResult:
    """Upsert documents into Chroma, only re-embedding new or changed ones, and remove stale ones."""
    collection = await asyncio.to_thread(get_collection)
    existing = await asyncio.to_thread(collection.get, include=["metadatas"])
    existing_hashes = {
        doc_id: (meta or {}).get("hash") for doc_id, meta in zip(existing["ids"], existing["metadatas"] or [])
    }

    changed = [d for d in documents if existing_hashes.get(d.id) != d.metadata["hash"]]
    if changed:
        vectors = await embed([d.text for d in changed])
        await asyncio.to_thread(
            collection.upsert,
            ids=[d.id for d in changed],
            documents=[d.text for d in changed],
            metadatas=[d.metadata for d in changed],
            embeddings=vectors,
        )

    current_ids = {d.id for d in documents}
    stale = [doc_id for doc_id in existing_hashes if doc_id not in current_ids]
    if stale:
        await asyncio.to_thread(collection.delete, ids=stale)

    result = IndexResult(
        total=len(documents),
        embedded=len(changed),
        unchanged=len(documents) - len(changed),
        deleted=len(stale),
    )
    logger.info("Indexing done: %s", result)
    return result


async def reindex() -> IndexResult:
    """Load products + reviews from Juice Shop and sync them into Chroma."""
    products = await get_products()
    reviews = await get_reviews([p.id for p in products])
    return await index_documents(build_documents(products, reviews))


# ---------- Retrieval ----------

async def retrieve(question: str) -> list[RetrievedChunk]:
    settings = get_settings()
    collection = await asyncio.to_thread(get_collection)
    [query_vector] = await embed([question])
    result = await asyncio.to_thread(
        collection.query,
        query_embeddings=[query_vector],
        n_results=settings.rag_top_k,
        include=["documents", "metadatas", "distances"],
    )
    return [
        RetrievedChunk(text=text, metadata=meta or {}, distance=distance)
        for text, meta, distance in zip(result["documents"][0], result["metadatas"][0], result["distances"][0])
    ]


if __name__ == "__main__":
    # Manual (re)index: python -m app.rag
    logging.basicConfig(level=logging.INFO)
    print(asyncio.run(reindex()))
