# Juice Shop AI Assistant

A FastAPI service that answers **product** and **pricing** questions for OWASP Juice Shop using OpenAI.
It does **not** give recipes or explain how products are prepared or made.

The Juice Shop frontend shows it as a chat button in the bottom-right corner (`frontend/src/app/ai-assistant-widget/`).

> This is separate from Juice Shop's built-in `/chatbot`, which is intentionally vulnerable for the LLM hacking challenges.

## How it works

```
question ──► recipe guardrail ──► embed question (OpenAI) ──► Chroma (Docker) top-k ──► prompt ──► OpenAI chat ──► answer
                                                                     ▲
Juice Shop /api/Products + /rest/products/{id}/reviews ──► documents ┘ (embedded with text-embedding-3-small)
```

- **RAG knowledge base:** every product (name, description, price, deluxe price) and every customer review is
  one document. Documents are embedded with OpenAI `text-embedding-3-small` and stored in **Chroma**, which runs in Docker.
  Review author emails are never embedded.
- **Indexing:** runs automatically when the service starts, and on `POST /admin/reindex` or `python -m app.rag`.
  Each document stores a content hash, so only new or changed documents are re-embedded and deleted ones are removed.
- **Answering:** the top `RAG_TOP_K` chunks for the question are added to the prompt. The compact product catalog
  is also always included, so price questions ("cheapest juice", totals) stay exact.
  Reviews are treated as untrusted data in the prompt (Juice Shop reviews deliberately contain injection payloads).
- **Fallback:** if Chroma is down, `/chat` still answers from the catalog alone, and `/health` reports `degraded`.
- Products are loaded live from Juice Shop's `GET /api/Products` and cached for 5 minutes.
  If Juice Shop is down, the last loaded catalog is used. If nothing was ever loaded, `/chat` returns `503`.
- Recipe protection has two layers:
  1. `app/guardrails.py`: a keyword/regex check refuses recipe or "how is it made" questions **before** OpenAI is called.
  2. A system prompt rule refuses indirect attempts the regex misses (role-play, hypotheticals, ...).

## Setup

The OpenAI key is read from `.env.openai` in the repo root (gitignored, and excluded from Docker images):

```
OPENAI_API_KEY=sk-...
```

## Run everything with Docker (recommended)

Requires [Docker Desktop](https://www.docker.com/products/docker-desktop/) running. From the **repo root**:

```powershell
docker compose up -d --build     # first build takes 10-20 min (Juice Shop + Angular build)
docker compose ps                # juice-shop, juice-shop-chroma, juice-shop-ai-assistant
docker compose logs -f ai-assistant
```

| Container | URL | Notes |
|---|---|---|
| `juice-shop` | http://localhost:3000 | Shop with the chat widget (bottom-right) |
| `juice-shop-ai-assistant` | http://localhost:8000/docs | Reaches the shop at `http://juice-shop:3000` and Chroma at `chroma:8000` on the compose network |
| `juice-shop-chroma` | http://localhost:8001 | Embeddings kept in the `ai_assistant_chroma-data` volume |

The browser calls the assistant at `http://localhost:8000` (`aiAssistantUrl` in the frontend environments), and CORS allows `http://localhost:3000`.
The assistant waits for Juice Shop to be healthy, then indexes into Chroma. It retries while Chroma or the shop is still starting.

- Stop: `docker compose down`. Also delete the embeddings: `docker compose down -v`
- After changing assistant code: `docker compose up -d --build ai-assistant`
- After changing Juice Shop/frontend code: `docker compose up -d --build juice-shop`

## Run locally without Docker for the app (development)

Requires Python 3.11+.

```powershell
cd ai_assistant
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt
docker compose up -d chroma                                  # from the repo root: only the vector DB
npm start                                                    # from the repo root: Juice Shop on :3000
.venv\Scripts\uvicorn app.main:app --port 8000 --reload     # in ai_assistant/
```

## API

- Swagger UI: http://localhost:8000/docs
- `GET /health` → `{"status": "ok", "juice_shop": {...}, "vector_store": {"status": "ok", "documents": ...}}`
- `POST /admin/reindex` → `{"total": .., "embedded": .., "unchanged": .., "deleted": ..}` (local admin use only; not authenticated). Run it after products or reviews change.
- `POST /chat`

```json
{ "message": "How much is Apple Juice?", "history": [{ "role": "user", "content": "..." }, { "role": "assistant", "content": "..." }] }
```
→ `{ "reply": "...", "blocked": false, "sources": ["Apple Juice (1000ml)"] }`
(`blocked` is `true` when the recipe rule refused the request; `sources` lists products whose knowledge was retrieved)

## Configuration (environment variables, optional)

| Variable | Default | Purpose |
|---|---|---|
| `OPENAI_API_KEY` | (required) | OpenAI API key |
| `OPENAI_MODEL` | `gpt-4.1-mini` | Model used for answers |
| `JUICE_SHOP_URL` | `http://localhost:3000` | Where products are fetched from |
| `PRODUCT_CACHE_TTL` | `300` | Product cache lifetime in seconds |
| `ALLOWED_ORIGINS` | `http://localhost:3000,http://localhost:4200` | CORS origins allowed to call the API |
| `MAX_HISTORY_TURNS` | `10` | Conversation turns sent to the model |
| `CHROMA_HOST` / `CHROMA_PORT` | `localhost` / `8001` | Chroma server (Docker) |
| `CHROMA_COLLECTION` | `juice_shop_products` | Collection name |
| `EMBEDDING_MODEL` | `text-embedding-3-small` | OpenAI embedding model |
| `RAG_TOP_K` | `6` | Chunks retrieved per question |
| `INDEX_ON_STARTUP` | `true` | Sync Chroma with Juice Shop when the service starts |

Changing `EMBEDDING_MODEL` changes the vector size, so delete the old data first (`docker compose down -v`).

The frontend reads the service URL from `aiAssistantUrl` in `frontend/src/environments/environment*.ts`.

## Tests

```powershell
cd ai_assistant
.venv\Scripts\pytest
```
