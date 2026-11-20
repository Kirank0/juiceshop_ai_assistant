import asyncio
import logging
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from openai import AuthenticationError, OpenAIError, RateLimitError
from pydantic import BaseModel, Field

from . import assistant, rag
from .config import get_settings
from .guardrails import REFUSAL_MESSAGE, is_recipe_request
from .products import ProductsUnavailableError, get_products

logging.basicConfig(level=logging.INFO, format="%(levelname)s:     %(name)s - %(message)s")
logging.getLogger("httpx").setLevel(logging.WARNING)  # health checks would flood the log
logger = logging.getLogger("ai_assistant")
settings = get_settings()

STARTUP_INDEX_ATTEMPTS = 30
STARTUP_INDEX_RETRY_SECONDS = 10


async def _index_in_background() -> None:
    """Sync Chroma on startup, retrying while Juice Shop or Chroma are still starting (e.g. in Docker)."""
    for attempt in range(1, STARTUP_INDEX_ATTEMPTS + 1):
        try:
            result = await rag.reindex()
            logger.info("Startup indexing done: %s", result)
            return
        except (AuthenticationError, RateLimitError) as err:  # retrying would not help
            logger.error("Startup indexing failed, OpenAI rejected the request: %s", err)
            return
        except Exception as err:  # startup must not fail because Chroma/Juice Shop is down
            rag.reset_connection()
            logger.warning("Startup indexing attempt %d/%d failed: %s", attempt, STARTUP_INDEX_ATTEMPTS, err)
            await asyncio.sleep(STARTUP_INDEX_RETRY_SECONDS)
    logger.error("Startup indexing gave up; call POST /admin/reindex once Juice Shop and Chroma are up")


@asynccontextmanager
async def lifespan(_: FastAPI):
    task = asyncio.create_task(_index_in_background()) if settings.index_on_startup else None
    yield
    if task and not task.done():
        task.cancel()


app = FastAPI(
    title="Juice Shop AI Assistant",
    description="Answers product and pricing questions for OWASP Juice Shop using RAG over Chroma. "
                "Does not provide recipes.",
    version="2.0.0",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


class HistoryMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(max_length=4000)


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=1000)
    history: list[HistoryMessage] = Field(default_factory=list, max_length=40)


class ChatResponse(BaseModel):
    reply: str
    blocked: bool = False
    sources: list[str] = Field(default_factory=list, description="Products whose knowledge was used")


class ReindexResponse(BaseModel):
    total: int
    embedded: int
    unchanged: int
    deleted: int


def _openai_http_error(err: OpenAIError) -> HTTPException:
    if isinstance(err, (AuthenticationError, RateLimitError)):
        logger.error("OpenAI request failed: %s", err)
        return HTTPException(status_code=503, detail=f"OpenAI rejected the request: {err.message}")
    logger.exception("OpenAI request failed")
    return HTTPException(status_code=502, detail="The AI service is currently unavailable")


@app.get("/health")
async def health() -> dict:
    try:
        products = await get_products()
        shop = {"status": "ok", "products_loaded": len(products)}
    except ProductsUnavailableError as err:
        shop = {"status": "unavailable", "products_loaded": 0, "detail": str(err)}
    vector_store = await rag.chroma_status()
    status = "ok" if shop["status"] == "ok" and vector_store["status"] == "ok" else "degraded"
    return {"status": status, "juice_shop": shop, "vector_store": vector_store}


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    if is_recipe_request(request.message):
        return ChatResponse(reply=REFUSAL_MESSAGE, blocked=True)
    try:
        products = await get_products()
    except ProductsUnavailableError as err:
        raise HTTPException(status_code=503, detail=str(err)) from err
    if not settings.openai_api_key:
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY is not configured")

    try:
        chunks = await rag.retrieve(request.message)
    except OpenAIError as err:
        raise _openai_http_error(err) from err
    except Exception as err:  # Chroma down: still answer from the catalog alone
        logger.warning("Retrieval failed, answering without RAG context: %s", err)
        rag.reset_connection()
        chunks = []

    try:
        reply = await assistant.ask(
            request.message,
            [m.model_dump() for m in request.history],
            products,
            chunks,
        )
    except OpenAIError as err:
        raise _openai_http_error(err) from err

    sources = list(dict.fromkeys(c.metadata["product_name"] for c in chunks if c.metadata.get("product_name")))
    return ChatResponse(reply=reply, blocked=REFUSAL_MESSAGE in reply, sources=sources)


@app.post("/admin/reindex", response_model=ReindexResponse)
async def reindex() -> ReindexResponse:
    """Re-sync products and reviews from Juice Shop into Chroma (only changed documents are re-embedded)."""
    try:
        result = await rag.reindex()
    except ProductsUnavailableError as err:
        raise HTTPException(status_code=503, detail=str(err)) from err
    except rag.VectorStoreUnavailableError as err:
        rag.reset_connection()
        raise HTTPException(status_code=503, detail=str(err)) from err
    except OpenAIError as err:
        raise _openai_http_error(err) from err
    return ReindexResponse(**vars(result))
