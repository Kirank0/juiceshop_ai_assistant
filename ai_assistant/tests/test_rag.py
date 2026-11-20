import asyncio
import math

import pytest

from app import rag
from app.products import Product, Review

PRODUCTS = [
    Product(id=1, name="Apple Juice (1000ml)", description="The all-time classic.", price=1.99, deluxe_price=0.99),
    Product(id=2, name="Lemon Juice (500ml)", description="Sour but full of vitamins.", price=2.99, deluxe_price=None),
]
REVIEWS = [
    Review(id="r1", product_id=1, message="One of my favorites!", likes=3),
    Review(id="r2", product_id=99, message="Review of unknown product", likes=0),
]


class FakeCollection:
    """Minimal in-memory stand-in for a Chroma collection (cosine distance)."""

    def __init__(self):
        self.items: dict[str, dict] = {}

    def get(self, include=None):
        ids = list(self.items)
        return {"ids": ids, "metadatas": [self.items[i]["metadata"] for i in ids]}

    def upsert(self, ids, documents, metadatas, embeddings):
        for i, d, m, e in zip(ids, documents, metadatas, embeddings):
            self.items[i] = {"document": d, "metadata": m, "embedding": e}

    def delete(self, ids):
        for i in ids:
            self.items.pop(i, None)

    def count(self):
        return len(self.items)

    def query(self, query_embeddings, n_results, include):
        q = query_embeddings[0]

        def distance(v):
            dot = sum(a * b for a, b in zip(q, v))
            return 1 - dot / (math.hypot(*q) * math.hypot(*v))

        ranked = sorted(self.items.values(), key=lambda item: distance(item["embedding"]))[:n_results]
        return {
            "documents": [[r["document"] for r in ranked]],
            "metadatas": [[r["metadata"] for r in ranked]],
            "distances": [[distance(r["embedding"]) for r in ranked]],
        }


def fake_vector(text: str) -> list[float]:
    """Deterministic toy embedding: keyword presence."""
    lowered = text.lower()
    return [float("apple" in lowered), float("lemon" in lowered), float("favorite" in lowered), 0.1]


@pytest.fixture
def store(monkeypatch):
    collection = FakeCollection()
    calls: list[list[str]] = []

    async def fake_embed(texts):
        calls.append(list(texts))
        return [fake_vector(t) for t in texts]

    monkeypatch.setattr(rag, "get_collection", lambda: collection)
    monkeypatch.setattr(rag, "embed", fake_embed)
    return collection, calls


def test_build_documents_creates_product_and_review_docs():
    docs = rag.build_documents(PRODUCTS, REVIEWS)

    assert [d.id for d in docs][:2] == ["product-1", "product-2"]
    assert docs[2].id.startswith("review-1-") and len(docs) == 3
    assert "Price: $1.99" in docs[0].text and "Deluxe member price: $0.99" in docs[0].text
    assert "Deluxe member price: not available" in docs[1].text
    assert docs[2].text.startswith("Customer review of Apple Juice (1000ml): One of my favorites!")
    assert docs[2].metadata["product_id"] == 1
    assert all("@" not in d.text for d in docs)  # author emails are never embedded


def test_index_only_embeds_new_or_changed_documents_and_removes_stale(store):
    collection, calls = store
    docs = rag.build_documents(PRODUCTS, REVIEWS)

    first = asyncio.run(rag.index_documents(docs))
    assert (first.total, first.embedded, first.unchanged, first.deleted) == (3, 3, 0, 0)

    second = asyncio.run(rag.index_documents(docs))
    assert (second.embedded, second.unchanged) == (0, 3)
    assert len(calls) == 1  # nothing re-embedded

    changed_price = [Product(id=1, name="Apple Juice (1000ml)", description="The all-time classic.",
                             price=2.49, deluxe_price=0.99)]
    third = asyncio.run(rag.index_documents(rag.build_documents(changed_price, [])))
    assert (third.embedded, third.deleted) == (1, 2)
    assert set(collection.items) == {"product-1"}
    assert "Price: $2.49" in collection.items["product-1"]["document"]


def test_retrieve_returns_most_relevant_chunks(store, monkeypatch):
    monkeypatch.setattr(rag.get_settings(), "rag_top_k", 2)
    asyncio.run(rag.index_documents(rag.build_documents(PRODUCTS, REVIEWS)))

    chunks = asyncio.run(rag.retrieve("Is the apple juice a favorite?"))

    assert len(chunks) == 2
    assert chunks[0].metadata["type"] == "review"
    assert chunks[0].metadata["product_name"] == "Apple Juice (1000ml)"
    assert all("Lemon" not in c.text for c in chunks)


def test_duplicate_reviews_are_indexed_once():
    duplicates = [Review(id="a", product_id=1, message="Tasty!", likes=0),
                  Review(id="b", product_id=1, message="Tasty!", likes=0)]
    docs = rag.build_documents(PRODUCTS, duplicates)
    assert [d.metadata["type"] for d in docs].count("review") == 1
