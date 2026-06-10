# Architecture Documentation
## Healthcare AI Assistant — Technical Deep Dive

---

## 1. System Overview

The Healthcare AI Assistant is a **Retrieval-Augmented Generation (RAG)** system that answers questions by retrieving relevant passages from a curated set of healthcare documents and using a local LLM to synthesise a grounded, cited response.

The system deliberately avoids:
- External API calls for LLM inference (HIPAA compliance)
- Answering from the LLM's parametric knowledge (hallucination prevention)
- Providing medical diagnosis or personal treatment advice (safety)

---

## 2. High-Level Data Flow

```
┌──────────────────────────────────────────────────────────────────────────┐
│                          INGESTION PIPELINE                              │
│                                                                          │
│  data/documents/*.txt                                                    │
│         │                                                                │
│         ▼                                                                │
│  DocumentLoader ──────────── Reads .txt/.pdf/.md/.docx                  │
│         │                    Attaches source metadata                    │
│         ▼                                                                │
│  RecursiveCharacterTextSplitter                                          │
│         │   chunk_size=1000, chunk_overlap=200                           │
│         ▼                                                                │
│  MiniLM-L6-v2 (HuggingFace)  ── 384-dim dense vectors                   │
│         │                                                                │
│         ▼                                                                │
│  FAISS IndexFlatL2 ──────── Persisted to data/faiss_index/              │
└──────────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────────┐
│                           QUERY PIPELINE                                 │
│                                                                          │
│  User Query                                                              │
│      │                                                                   │
│      ▼                                                                   │
│  Pydantic Validation ───── Length, injection pattern checks              │
│      │                                                                   │
│      ▼                                                                   │
│  Safety Pre-Checks ─────── Emergency keywords → 911 response            │
│      │                     Diagnosis keywords → refusal                  │
│      ▼                                                                   │
│  Agent Router                                                            │
│      │                                                                   │
│      ├── APPOINTMENT keywords ──▶ check_available_slots() [mock]         │
│      │                               │                                   │
│      │                               ▼                                   │
│      │                          APPOINTMENT_PROMPT + Mistral             │
│      │                               │                                   │
│      │                               ▼                                   │
│      │                          AskResponse (QueryType=APPOINTMENT)      │
│      │                                                                   │
│      └── KNOWLEDGE ──────────▶ FAISS Similarity Search (MMR, K=5)       │
│                                    │                                     │
│                                    ▼                                     │
│                               Confidence Guard                           │
│                               (max_similarity < 0.25 → REFUSAL)         │
│                                    │                                     │
│                                    ▼                                     │
│                               Context Assembly                           │
│                               [Doc 1|Source:X|Rel:0.92]\n...            │
│                                    │                                     │
│                                    ▼                                     │
│                               RAG_PROMPT (strict grounding rules)        │
│                                    │                                     │
│                                    ▼                                     │
│                               Mistral 7B (Ollama, temp=0.1)             │
│                                    │                                     │
│                                    ▼                                     │
│                               Response Parser                            │
│                               ANSWER / CONFIDENCE / SOURCES / DISCLAIMER│
│                                    │                                     │
│                                    ▼                                     │
│                               Confidence Blending                        │
│                               (0.6 × FAISS_score + 0.4 × LLM_score)    │
│                                    │                                     │
│                                    ▼                                     │
│                               Citation Builder                           │
│                               SourceCitation[] with excerpts            │
│                                    │                                     │
│                                    ▼                                     │
│                               AskResponse (QueryType=KNOWLEDGE)          │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Component Breakdown

### 3.1 FastAPI Backend (`backend/main.py`)

- **ASGI framework** — asyncio event loop, handles concurrent requests.
- **Lifespan hook** — loads FAISS index from disk at startup.
- **Middleware** — request-ID injection (distributed tracing), CORS, global exception handler.
- **Routers** — `health`, `ingest`, `ask` mounted under `/api/v1`.

### 3.2 Document Loader (`backend/utils/document_loader.py`)

- Scans a directory for `.txt`, `.pdf`, `.md`, `.docx` files.
- Uses LangChain community loaders: `TextLoader`, `PyPDFLoader`, `UnstructuredMarkdownLoader`, `Docx2txtLoader`.
- Attaches `source`, `file_path`, `file_size_kb`, `load_time_s` metadata to every `Document`.
- Returns `List[Document]` to the ingestion pipeline.

### 3.3 Embedding Model (`backend/embeddings.py`)

| Property | Value |
|---|---|
| Model | `sentence-transformers/all-MiniLM-L6-v2` |
| Dimensions | 384 |
| Max tokens | 256 |
| Normalisation | L2 → cosine similarity via dot product |
| Device | CPU (configurable to CUDA) |
| Caching | `@lru_cache` singleton — loads once per process |

**Why MiniLM-L6-v2:**
- 80MB model — fast CPU inference, no GPU required.
- State-of-the-art recall on English semantic similarity tasks.
- Open-source (Apache 2.0) — no API cost or vendor lock-in.
- Supports batch encoding for efficient ingestion.

### 3.4 FAISS Vector Store (`backend/vector_store.py`)

| Property | Value |
|---|---|
| Index type | `IndexFlatL2` (exact nearest-neighbour) |
| Distance metric | L2 (equivalent to cosine on normalised vectors) |
| Search strategy | MMR (Maximum Marginal Relevance) |
| Top-K | 5 chunks per query |
| MMR fetch_k | 15 (3× Top-K candidate pool) |
| MMR lambda | 0.7 (70% relevance, 30% diversity) |
| Persistence | Saved to `data/faiss_index/` after ingestion |

**Chunking Strategy:**
```
RecursiveCharacterTextSplitter
  chunk_size    = 1000 characters
  chunk_overlap = 200 characters
  separators    = ["\n\n", "\n", ". ", " ", ""]
```
The recursive splitter tries to break on paragraph boundaries first (preferred for healthcare documents with distinct sections), then sentences, then words.

### 3.5 LLM (`backend/llm.py`)

| Property | Value |
|---|---|
| Model | Mistral 7B Instruct |
| Serving | Ollama (local HTTP API) |
| Temperature | 0.1 (near-deterministic) |
| Max tokens | 1024 |
| Repeat penalty | 1.1 (reduces verbatim context echoing) |
| Timeout | 120 seconds |

**Why local inference:**
- Healthcare data sensitivity — no PHI sent to external APIs.
- HIPAA compliance — data stays within the network boundary.
- No per-token cost — suitable for high-volume clinical environments.

### 3.6 RAG Pipeline (`backend/rag.py`)

**Hallucination Prevention (4 layers):**

1. **Prompt-level:** Explicit rule "Answer ONLY from context; if not present, refuse."
2. **FAISS threshold:** If max similarity score < 0.25, return canned refusal — never call LLM.
3. **LLM self-reporting:** Model outputs `CONFIDENCE: NONE` when context is insufficient; parser catches and overrides to refusal.
4. **Blended scoring:** `0.6 × FAISS_score + 0.4 × LLM_score` — LLM cannot self-report HIGH confidence when FAISS evidence is weak.

**Confidence Scoring Algorithm:**
```python
def _compute_confidence_score(llm_confidence, similarity_scores):
    llm_score  = {HIGH: 0.9, MEDIUM: 0.65, LOW: 0.35, NONE: 0.0}[llm_confidence]
    faiss_score = mean(top_3_similarity_scores)
    return 0.6 * faiss_score + 0.4 * llm_score    # evidence-weighted
```

**Confidence Level Mapping:**
```
score ≥ 0.75  → HIGH
score ≥ 0.50  → MEDIUM
score ≥ 0.25  → LOW
score < 0.25  → NONE (triggers refusal)
```

### 3.7 Agent Router (`backend/agent.py`)

**Routing Priority:**
1. **Keyword scan** (O(n) set intersection) — fastest, handles 95%+ of appointment queries.
2. **RAG pipeline** — default for all knowledge queries.

**Appointment Keyword Set:**
```python
{"appointment", "appointments", "book", "booking", "schedule", "scheduling",
 "availability", "available", "slot", "slots", "cancel", "cancellation",
 "reschedule", "visit", "consultation", "see a doctor", "make an appointment"}
```

**Mock Scheduling Tool (`check_available_slots`):**
- Generates pseudo-random availability for next 7 business days.
- Returns 8 slots across 5 provider types (Primary Care, Internal Medicine, Cardiology, etc.).
- Skips weekends (Mon–Fri only).
- In production: replace with EHR/Epic/Cerner calendar API call.

### 3.8 Streamlit Frontend (`frontend/app.py`)

**UI Components:**
- **Sidebar:** Real-time health status (polling /health every 30s), document ingestion trigger, session info, capability guide.
- **Chat area:** Message bubbles with role-specific styling, suggestion chips on empty state.
- **Per-response metadata:** Confidence badge (HIGH/MEDIUM/LOW/NONE), query type badge, numeric confidence progress bar, processing time.
- **Citation panel:** Collapsible expander per response showing source document, section, excerpt, and similarity score.
- **Disclaimer box:** Warm-yellow safety notice appended to every clinical answer.

---

## 4. Data Models

### AskResponse (the core contract)
```python
class AskResponse:
    query:                   str
    answer:                  str
    confidence:              ConfidenceLevel  # HIGH|MEDIUM|LOW|NONE
    confidence_score:        float            # 0.0 – 1.0
    query_type:              QueryType        # KNOWLEDGE|APPOINTMENT
    sources:                 List[SourceCitation]
    disclaimer:              str
    processing_time_seconds: float
    conversation_id:         Optional[str]
```

### SourceCitation
```python
class SourceCitation:
    document_name:    str
    page_or_section:  Optional[str]
    relevant_excerpt: str             # ≤500 chars
    similarity_score: Optional[float] # 0.0 – 1.0
```

---

## 5. Security Architecture

```
Internet
    │
    ▼
[Nginx / Traefik] ── TLS termination, rate limiting
    │
    ▼
[FastAPI Backend] ── JWT auth middleware (add for production)
    │                Pydantic validation
    │                Injection pattern blocking
    │                Request-ID tracing
    ▼
[Ollama LLM] ─────── Internal network only (not exposed externally)
    │
[FAISS Index] ─────── Local filesystem (encrypted at-rest in production)
```

**HIPAA Technical Safeguards Implemented:**
- ✅ Audit logging (JSON with timestamps and request IDs)
- ✅ Data minimisation (no PHI stored — policy documents only)
- ✅ Local inference (no external data transmission)
- ✅ Input validation and injection prevention
- ⚙️ TLS encryption (configure nginx/Traefik in production)
- ⚙️ Authentication (add JWT/OAuth2 middleware for production)
- ⚙️ At-rest encryption (configure Docker volume encryption)

---

## 6. Scalability Pathways

| Scale Challenge | Solution |
|---|---|
| More documents (>100k chunks) | Migrate from `IndexFlatL2` to `IndexIVFFlat` (approximate NN) |
| Higher query throughput | Scale backend containers behind a load balancer |
| Faster LLM inference | Add GPU to Ollama container; or switch to vLLM |
| Multi-tenancy | Partition FAISS index per organisation; add JWT with org claims |
| Distributed index | Migrate to Weaviate, Pinecone, or Milvus for sharded vector store |
| Real appointment booking | Replace `check_available_slots()` with Epic/Cerner FHIR API call |

---

## 7. Prompt Engineering Decisions

**RAG Prompt Design:**
1. Role framing: "You are a Healthcare AI Assistant" — sets domain context.
2. Explicit rule list with numbered items — Mistral follows numbered lists reliably.
3. REFUSAL MESSAGE verbatim in prompt — model copy-pastes it on uncertainty (reduces creative refusals).
4. Structured output format (ANSWER/CONFIDENCE/SOURCES/DISCLAIMER) — enables deterministic parsing.
5. Inline source citation instruction — "Reference document names inline" — drives citation behaviour.

**Temperature = 0.1:**
- Near-zero temperature means the model almost always takes the highest-probability token.
- This eliminates creative variation that could introduce hallucinated facts.
- Appropriate for factual Q&A where consistency > creativity.

---

*Document Version: 1.0 | Last Updated: 2024*
