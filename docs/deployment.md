# Deployment Guide

## Local Development

### Prerequisites
- Python 3.11+
- Rust toolchain (for `rust_gateway`)
- Node.js (optional, for frontend tooling)

### Setup

```bash
# 1. Clone and enter project root
cd "Hotel booking"

# 2. Create virtual environment
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Unix

# 3. Install Python dependencies
pip install -r requirements.txt

# 4. Copy and fill environment variables
copy backend\.env .env        # Windows
# cp backend/.env .env        # Unix
# Edit .env with your API keys

# 5. Start Rust gateway (separate terminal)
cd rust_gateway
cargo run --release

# 6. Start FastAPI backend (project root)
uvicorn app.main:app --reload --port 8000

# 7. Start Chainlit frontend (separate terminal)
chainlit run frontend/chainlit_app.py --port 8501
```

### Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `OPENAI_API_KEY` | Yes | OpenAI API key |
| `GROQ_API_KEY` | Yes | Groq API key (Llama model) |
| `DATABASE_URL` | No | PostgreSQL URL (Supabase) |
| `ADK_ENABLED` | No | `1` = V2 ADK pipeline, `0` = V1 fallback |
| `ADK_DISPATCHER_MODEL` | No | Default: `openai/gpt-5-nano` |
| `ADK_VOICE_MODEL` | No | Default: `groq/llama-3.3-70b-versatile` |
| `DPO_TELEMETRY_ENABLED` | No | Default: `1` |
| `RUST_GATEWAY_URL` | No | Default: `http://localhost:3001` |

## Docker Compose

```bash
# Build and start all services
docker-compose up --build

# Stop services
docker-compose down
```

Services started:
- `fastapi` → http://localhost:8000
- `chainlit` → http://localhost:8501
- `rust_gateway` → http://localhost:3001

## Production Checklist

- [ ] Set `CORS` origins to specific frontend domain in `app/main.py`
- [ ] Use a strong PostgreSQL `DATABASE_URL` (not SQLite)
- [ ] Set `STRIPE_WEBHOOK_SECRET` for payment webhooks
- [ ] Enable HTTPS termination at reverse proxy (nginx/caddy)
- [ ] Set `RATE_LIMIT_ENABLED=1` on Rust gateway
- [ ] Configure `DPO_SQLITE_PATH` to a persistent volume mount
