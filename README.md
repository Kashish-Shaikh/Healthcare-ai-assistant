# 🏥 Healthcare AI Assistant

> A production-grade Retrieval-Augmented Generation (RAG) system for healthcare information — built with FastAPI, LangChain, FAISS, Ollama/Mistral, and Streamlit.

[![Python 3.11+](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111-green.svg)](https://fastapi.tiangolo.com)
[![LangChain](https://img.shields.io/badge/LangChain-0.2-orange.svg)](https://langchain.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 📋 Table of Contents

1. [Features](#features)
2. [Architecture](#architecture)
3. [Quick Start](#quick-start)
4. [API Reference](#api-reference)
5. [Configuration](#configuration)
6. [Development](#development)
7. [Testing](#testing)
8. [Deployment](#deployment)
9. [Security & HIPAA](#security--hipaa)

---

## ✨ Features

| Feature | Description |
|---|---|
| **RAG Pipeline** | Answers questions exclusively from uploaded healthcare documents |
| **FAISS Vector Store** | Sub-millisecond semantic retrieval across thousands of document chunks |
| **Mistral 7B LLM** | Local inference via Ollama — no data leaves your network |
| **Source Citations** | Every answer cites the exact document and excerpt |
| **Hallucination Prevention** | Confidence scoring + FAISS similarity threshold guards |
| **Safety Guardrails** | Emergency detection, diagnosis refusal, prompt injection blocking |
| **Appointment Agent** | Keyword-routed scheduling tool with mock slot availability |
| **Streamlit UI** | Professional chat interface with confidence indicators |
| **Structured Logging** | JSON logs with request IDs for full observability |
| **Docker Ready** | Single `docker-compose up` to run the full stack |

---

## 🏗️ Architecture

```
User Query
    │
    ▼
Streamlit UI (port 8501)
    │  HTTP POST /api/v1/ask
    ▼
FastAPI Backend (port 8000)
    │
    ├─ Input Validation (Pydantic)
    ├─ Safety Checks (Emergency / Diagnosis)
    │
    ▼
LangChain Agent Router
    │
    ├─ Keywords: appointment/book/schedule/availability
    │       └──▶ check_available_slots() [mock tool]
    │                   └──▶ Mistral (formatting)
    │                           └──▶ AskResponse
    │
    └─ All other queries
            └──▶ FAISS Retriever (Top-K=5, MMR)
                        └──▶ Context Assembly
                                └──▶ RAG Prompt + Mistral
                                        └──▶ Response Parser
                                                └──▶ Confidence Scorer
                                                        └──▶ AskResponse

FAISS Index ◀── MiniLM-L6-v2 Embeddings ◀── Document Chunks ◀── Healthcare Docs
```

---

## 🚀 Quick Start

### Option A — Docker (Recommended)

```bash
# 1. Clone the repository
git clone https://github.com/your-org/healthcare-ai-assistant.git
cd healthcare-ai-assistant

# 2. Configure environment
cp .env.example .env

# 3. Start all services (Ollama + Backend + Frontend)
docker-compose up -d

# 4. Wait for Ollama to pull Mistral (~4GB, first time only)
docker-compose logs -f ollama

# 5. Ingest documents
curl -X POST http://localhost:8000/api/v1/ingest \
  -H "Content-Type: application/json" \
  -d '{"documents_dir": "data/documents"}'

# 6. Open the UI
open http://localhost:8501
```

### Option B — Local Development

```bash
# Prerequisites: Python 3.11+, Ollama installed locally

# 1. Install Ollama and pull Mistral
curl -fsSL https://ollama.ai/install.sh | sh
ollama pull mistral

# 2. Set up Python environment
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt

# 3. Configure environment
cp .env.example .env
# Edit .env: set OLLAMA_BASE_URL=http://localhost:11434

# 4. Start the backend
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload

# 5. In a second terminal, start the frontend
streamlit run frontend/app.py --server.port 8501

# 6. Ingest documents via UI sidebar or API
```

---

## 📡 API Reference

### `GET /api/v1/health`
Check system health (FAISS index + Ollama LLM).

```bash
curl http://localhost:8000/api/v1/health
```

**Response:**
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "components": {
    "faiss":  { "status": "healthy", "detail": "Index loaded with 127 chunks." },
    "ollama": { "status": "healthy", "detail": "Model 'mistral' is available.", "latency_ms": 42.5 }
  },
  "index_document_count": 127,
  "uptime_seconds": 3600.0
}
```

---

### `POST /api/v1/ingest`
Load and index healthcare documents.

```bash
curl -X POST http://localhost:8000/api/v1/ingest \
  -H "Content-Type: application/json" \
  -d '{
    "documents_dir": "data/documents",
    "force_reload": false
  }'
```

**Response:**
```json
{
  "success": true,
  "message": "Successfully ingested 6 document(s) into 127 chunks.",
  "documents_processed": 6,
  "chunks_created": 127,
  "index_path": "data/faiss_index",
  "processing_time_seconds": 18.4
}
```

---

### `POST /api/v1/ask`
Ask a healthcare question.

```bash
curl -X POST http://localhost:8000/api/v1/ask \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What documents do I need for a telehealth appointment?",
    "conversation_id": "session-abc-123",
    "include_sources": true
  }'
```

**Response:**
```json
{
  "query": "What documents do I need for a telehealth appointment?",
  "answer": "For a telehealth appointment you will need: (Source: telehealth_policy.txt)\n• Valid photo ID\n• Current insurance card\n• List of current medications...",
  "confidence": "HIGH",
  "confidence_score": 0.87,
  "query_type": "KNOWLEDGE",
  "sources": [
    {
      "document_name": "telehealth_policy.txt",
      "page_or_section": "Section 5",
      "relevant_excerpt": "Please arrive 15 minutes early...",
      "similarity_score": 0.91
    }
  ],
  "disclaimer": "⚠️ This AI assistant provides general healthcare information only...",
  "processing_time_seconds": 2.14,
  "conversation_id": "session-abc-123"
}
```

**Appointment Query Example:**
```bash
curl -X POST http://localhost:8000/api/v1/ask \
  -d '{"query": "Show me available appointment slots this week"}'
```

---

## ⚙️ Configuration

All settings are controlled via environment variables (see `.env.example`):

| Variable | Default | Description |
|---|---|---|
| `OLLAMA_BASE_URL` | `http://ollama:11434` | Ollama server URL |
| `OLLAMA_MODEL` | `mistral` | LLM model name |
| `LLM_TEMPERATURE` | `0.1` | Sampling temperature (low = deterministic) |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | HuggingFace embedding model |
| `FAISS_TOP_K` | `5` | Number of chunks to retrieve per query |
| `CHUNK_SIZE` | `1000` | Characters per document chunk |
| `CHUNK_OVERLAP` | `200` | Overlap between adjacent chunks |
| `MIN_CONFIDENCE_THRESHOLD` | `0.25` | Minimum FAISS similarity to answer |
| `REFUSE_DIAGNOSIS` | `true` | Block medical diagnosis requests |

---

## 🛠️ Development

### Project Structure
```
healthcare-ai-assistant/
├── backend/
│   ├── main.py          # FastAPI app
│   ├── config.py        # Settings (env-driven)
│   ├── logger.py        # Structured JSON logging
│   ├── prompts.py       # LLM prompt templates
│   ├── embeddings.py    # MiniLM-L6-v2 wrapper
│   ├── llm.py           # Ollama/Mistral wrapper
│   ├── vector_store.py  # FAISS manager
│   ├── rag.py           # Full RAG pipeline
│   ├── agent.py         # Router + appointment tool
│   ├── routers/         # FastAPI endpoint modules
│   ├── models/          # Pydantic schemas
│   └── utils/           # Document loader
├── frontend/app.py      # Streamlit UI
├── data/documents/      # Healthcare source documents
├── tests/               # Pytest test suite
└── docs/                # This documentation
```

### Adding New Documents
Drop any `.txt`, `.pdf`, `.md`, or `.docx` file into `data/documents/` and call `POST /api/v1/ingest`. The system will chunk, embed, and index the new content automatically.

---

## 🧪 Testing

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ -v --cov=backend --cov-report=html

# Run specific test file
pytest tests/test_api.py -v
pytest tests/test_rag.py -v
pytest tests/test_agent.py -v
```

---

## 🐳 Deployment

### Production Docker Compose

```bash
# Build images
docker-compose build

# Start in background
docker-compose up -d

# View logs
docker-compose logs -f

# Scale backend (requires a load balancer)
docker-compose up -d --scale backend=3

# Stop everything
docker-compose down

# Stop and remove all data (including FAISS index)
docker-compose down -v
```

### First-Time Setup Sequence
```bash
docker-compose up -d
# Wait ~2 minutes for Ollama to download Mistral

# Trigger document ingestion
curl -X POST http://localhost:8000/api/v1/ingest \
  -H "Content-Type: application/json" \
  -d '{"documents_dir": "data/documents"}'

# Verify health
curl http://localhost:8000/api/v1/health
```

---

## 🔒 Security & HIPAA

This system is designed with healthcare data privacy in mind:

- **Local LLM**: Mistral runs entirely on-premises via Ollama — no patient data is sent to external APIs.
- **No PHI Storage**: The system ingests de-identified policy documents only. Patient queries are processed in memory and not stored.
- **Prompt Injection Blocking**: Input validation rejects known injection patterns before they reach the LLM.
- **TLS in Production**: Configure a reverse proxy (nginx/Traefik) with TLS certificates for all endpoints.
- **Access Control**: Add JWT authentication middleware before deploying to production.
- **Audit Logging**: JSON-structured logs capture every request with a unique request ID for HIPAA audit trail requirements.

> ⚠️ **Important**: This is a demonstration system. Before deploying in a real clinical environment, conduct a full HIPAA risk assessment, implement authentication/authorisation, enable TLS, and engage a compliance officer.

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.
