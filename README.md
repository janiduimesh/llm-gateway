# 🚀 Enterprise LLM Gateway

A **production-grade API gateway** built with Python, FastAPI, and Redis that sits between internal teams and external AI providers (Groq primary, OpenAI fallback). Handles security, semantic caching, fault tolerance, and cost control.

---

## ✨ Features

| Layer | Capability |
|---|---|
| 🔒 **Authentication** | Per-team Bearer API keys with a JSON team registry |
| ⚡ **Rate Limiting** | Atomic sliding-window via Redis Lua script (per-team RPM caps) |
| 🛡️ **Guardrails** | Regex PII redaction (email, SSN, CC, phone, IP, secrets) + jailbreak detection |
| 🧠 **Semantic Cache** | Two-phase: exact SHA-256 hit → cosine-similarity fuzzy scan via `sentence-transformers` |
| 🔄 **Multi-Provider Failover** | Groq primary → OpenAI fallback with automatic retry & linear back-off |
| 🚦 **Circuit Breaker** | Three-state (CLOSED/OPEN/HALF-OPEN) per-provider fault isolation |
| 📡 **Streaming** | Native SSE streaming via `StreamingResponse` |
| 📊 **Observability** | Structured JSON logging on every request/response lifecycle event |

---

## 📂 Project Structure

```
llm-gateway/
├── app/
│   ├── main.py                 # FastAPI app factory, lifespan, middleware, admin routes
│   ├── api/
│   │   └── chat.py             # POST /v1/chat/completions orchestration
│   ├── core/
│   │   └── config.py           # Pydantic-settings, team registry, all config
│   ├── services/
│   │   ├── circuit_breaker.py  # 3-state CB with thread-safe per-provider state
│   │   ├── guardrails.py       # PII redaction + injection detection
│   │   ├── llm_router.py       # Multi-provider failover engine
│   │   ├── rate_limiter.py     # Redis Lua atomic sliding-window limiter
│   │   └── semantic_cache.py   # Vector similarity cache backed by Redis
│   └── utils/
│       └── logger.py           # JSON structured logging formatter
├── .env                        # Local secrets (git-ignored)
├── .gitignore
├── Dockerfile                  # Multi-stage build (builder + slim runtime)
├── docker-compose.yml          # FastAPI + Redis orchestration
└── requirements.txt
```

---

## ⚡ Quick Start

### Option 1 – Docker Compose (Recommended)

```bash
# 1. Clone the repo and enter the directory
git clone <your-repo-url> llm-gateway && cd llm-gateway

# 2. Copy the env template and fill in your API keys
cp .env .env.local
# Edit .env  →  set GROQ_API_KEY and OPENAI_API_KEY

# 3. Build and start all services
docker compose up --build

# 4. Test it
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Authorization: Bearer your-gateway-key" \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "Hello!"}]}'
```

### Option 2 – Local Development

```bash
# 1. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Start Redis locally (or use Docker)
docker run -d -p 6379:6379 redis:7-alpine

# 4. Set environment variables
cp .env .env.local
# Edit .env  →  set REDIS_URL=redis://localhost:6379/0 + API keys

# 5. Run the gateway
uvicorn app.main:app --reload --port 8000
```

---

## 🔑 Team Configuration

Teams are configured via the `TEAM_API_KEYS` environment variable as a JSON object:

```json
{
  "sk-team-eng-001": {
    "name": "engineering",
    "rate_limit_rpm": 120,
    "max_tokens_per_request": 8192,
    "allowed_models": ["llama3-8b-8192", "gpt-4o-mini"],
    "cost_budget_usd": 50.0
  },
  "sk-team-ds-001": {
    "name": "data-science",
    "rate_limit_rpm": 60,
    "max_tokens_per_request": 4096,
    "allowed_models": [],
    "cost_budget_usd": 20.0
  }
}
```

Set it as a single-line JSON string in `.env`:

```env
TEAM_API_KEYS='{"sk-team-eng-001": {"name": "engineering", "rate_limit_rpm": 120, ...}}'
```

---

## 📡 API Reference

### `POST /v1/chat/completions`

OpenAI-compatible endpoint.

**Headers:**
```
Authorization: Bearer <gateway-api-key>
Content-Type: application/json
```

**Request body:**
```json
{
  "model": "llama3-8b-8192",
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Explain circuit breakers in software."}
  ],
  "temperature": 0.7,
  "max_tokens": 1024,
  "stream": false
}
```

**Response (non-streaming):**
```json
{
  "id": "chatcmpl-xxx",
  "object": "chat.completion",
  "model": "llama3-8b-8192",
  "choices": [...],
  "usage": {"prompt_tokens": 30, "completion_tokens": 200, "total_tokens": 230},
  "_gateway": {
    "request_id": "uuid",
    "cache_hit": false,
    "latency_ms": 340.5,
    "team_id": "engineering",
    "guardrail_report": {...}
  }
}
```

### Utility Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Liveness probe |
| `GET` | `/status` | Readiness + circuit states + cache stats |
| `GET` | `/docs` | Swagger UI |
| `GET` | `/admin/cache/flush` | Flush all semantic cache entries |
| `GET` | `/admin/circuit/{provider}/reset` | Reset a provider's circuit breaker |
| `GET` | `/admin/rate-limit/{team_id}/reset` | Reset a team's rate limit counter |

---

## 🚦 Circuit Breaker States

```
CLOSED ──(≥N failures)──► OPEN ──(timeout elapsed)──► HALF-OPEN
  ▲                                                        │
  └──────────────(≥M successes)───────────────────────────┘
```

| State | Behaviour |
|---|---|
| `CLOSED` | Normal. Failures counted. |
| `OPEN` | All requests rejected immediately. No upstream calls. |
| `HALF-OPEN` | One probe allowed. Success → CLOSED. Failure → OPEN. |

Configure thresholds in `.env`:
```env
CB_FAILURE_THRESHOLD=5     # failures to trip
CB_RECOVERY_TIMEOUT=30     # seconds in OPEN
CB_SUCCESS_THRESHOLD=2     # successes to close
```

---

## 🛡️ Guardrails

**PII Detection & Redaction** (regex-based, zero-latency):

| Type | Example | Replaced with |
|---|---|---|
| Email | `user@company.com` | `[REDACTED_EMAIL]` |
| SSN | `123-45-6789` | `[REDACTED_SSN]` |
| Credit Card | `4111111111111111` | `[REDACTED_CC]` |
| Phone | `+1-800-555-0100` | `[REDACTED_PHONE]` |
| IPv4 | `192.168.1.100` | `[REDACTED_IP]` |
| Secrets | `password: abc123` | `[REDACTED_SECRET]` |
| Date of Birth | `01/15/1990` | `[REDACTED_DOB]` |

**Injection Detection** blocks requests matching patterns like:
- `"ignore all previous instructions"`
- `"you are now DAN"`
- `"override your safety filters"`

---

## 🧪 Running Tests

```bash
pip install pytest pytest-asyncio fakeredis
pytest tests/ -v
```

---

## ⚙️ Key Configuration Reference

| Variable | Default | Description |
|---|---|---|
| `GROQ_API_KEY` | *(required)* | Groq API key |
| `OPENAI_API_KEY` | *(required)* | OpenAI API key (fallback) |
| `REDIS_URL` | `redis://redis:6379/0` | Redis connection string |
| `CACHE_SIMILARITY_THRESHOLD` | `0.92` | Fuzzy cache hit threshold |
| `CACHE_TTL_SECONDS` | `3600` | Cache entry lifetime |
| `DEFAULT_RATE_LIMIT_RPM` | `60` | RPM for unregistered keys |
| `CB_FAILURE_THRESHOLD` | `5` | Failures before circuit trips |
| `GUARDRAILS_ENABLED` | `true` | Enable/disable guardrails |
| `PII_REDACTION_ENABLED` | `true` | Redact detected PII |
| `MAX_PROMPT_LENGTH` | `32768` | Max total prompt chars |

---

## 📜 License

MIT