import time
from dataclasses import dataclass

import httpx

from .config import get_settings


class ProductsUnavailableError(Exception):
    """Raised when the product catalog cannot be loaded from Juice Shop."""


@dataclass(frozen=True)
class Product:
    id: int
    name: str
    description: str
    price: float
    deluxe_price: float | None


_cache: list[Product] = []
_cache_time: float = 0.0


def _parse(raw: dict) -> Product:
    return Product(
        id=raw["id"],
        name=raw["name"],
        description=raw.get("description") or "",
        price=float(raw["price"]),
        deluxe_price=float(raw["deluxePrice"]) if raw.get("deluxePrice") is not None else None,
    )


async def get_products() -> list[Product]:
    """Return the product catalog, cached for PRODUCT_CACHE_TTL seconds.

    Falls back to the last cached catalog if Juice Shop is unreachable.
    """
    global _cache, _cache_time
    settings = get_settings()
    if _cache and time.monotonic() - _cache_time < settings.product_cache_ttl:
        return _cache
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            response = await client.get(f"{settings.juice_shop_url}/api/Products")
            response.raise_for_status()
            _cache = [_parse(p) for p in response.json()["data"]]
            _cache_time = time.monotonic()
    except (httpx.HTTPError, KeyError, ValueError) as err:
        if _cache:
            return _cache
        raise ProductsUnavailableError(
            f"Could not load products from {settings.juice_shop_url}. Is Juice Shop running?"
        ) from err
    return _cache


@dataclass(frozen=True)
class Review:
    id: str
    product_id: int
    message: str
    likes: int


async def get_reviews(product_ids: list[int]) -> list[Review]:
    """Fetch customer reviews for the given products (author emails are deliberately dropped)."""
    settings = get_settings()
    reviews: list[Review] = []
    async with httpx.AsyncClient(timeout=10) as client:
        for product_id in product_ids:
            try:
                response = await client.get(f"{settings.juice_shop_url}/rest/products/{product_id}/reviews")
                response.raise_for_status()
                for raw in response.json().get("data", []):
                    if raw.get("message"):
                        reviews.append(Review(
                            id=str(raw["_id"]),
                            product_id=product_id,
                            message=str(raw["message"]),
                            likes=int(raw.get("likesCount") or 0),
                        ))
            except (httpx.HTTPError, KeyError, ValueError):
                continue
    return reviews


def clear_cache() -> None:
    global _cache, _cache_time
    _cache = []
    _cache_time = 0.0
