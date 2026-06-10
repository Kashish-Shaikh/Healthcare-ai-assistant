# Interview Preparation Guide
## Healthcare AI Assistant — Top 50 Q&A

---

## Part A: Architecture & Design Decisions

---

**Q1. Explain the overall architecture of your Healthcare AI Assistant.**

The system is a Retrieval-Augmented Generation (RAG) pipeline composed of five layers:

1. **Ingestion layer** — DocumentLoader reads healthcare policy documents, RecursiveCharacterTextSplitter chunks them into 1000-character pieces with 200-character overlap, MiniLM-L6-v2 generates 384-dimensional embeddings, and FAISS stores them persistently on disk.

2. **API layer** — FastAPI with three endpoints: `POST /ingest` (builds the index), `POST /ask` (runs the pipeline), and `GET /health` (system monitoring). Pydantic models enforce strict request/response contracts.

3. **Routing layer** — A LangChain agent performs keyword scanning to route appointment-related queries to a mock scheduling tool, and all other queries to the RAG pipeline.

4. **RAG layer** — Four-stage hallucination prevention: safety pre-checks → FAISS similarity threshold guard → structured LLM prompt with grounding rules → blended confidence scoring (60% FAISS evidence, 40% LLM self-report).

5. **Presentation layer** — Streamlit frontend with confidence badges, citation panels, source excerpts, and a health dashboard.

The key architectural decision is **keeping everything local**: Ollama runs Mistral 7B on-premises, so no patient-related data ever leaves the network — critical for HIPAA compliance.

---

**Q2. Why did you choose FAISS over alternatives like Pinecone, Weaviate, or Chroma?**

For this use case, FAISS is the optimal choice for five reasons:

1. **In-process, zero-latency retrieval** — FAISS runs inside the Python process with no network hop. Retrieval takes <5ms for 100k vectors vs 50-200ms for hosted vector databases.

2. **Privacy/compliance** — FAISS stores vectors locally as files. No vectors are transmitted to external services, which is essential when those vectors might encode sensitive healthcare information.

3. **No infrastructure dependency** — No separate database server to provision, secure, or pay for. A single Docker volume holds the entire index.

4. **Maturity** — FAISS (Facebook AI Similarity Search) has been battle-tested at Meta scale since 2017. It's used in production by teams at Google, Microsoft, and Spotify.

5. **Sufficient scale** — For a clinic's policy documents (< 10k chunks), `IndexFlatL2` (exact nearest-neighbour) is perfect. Scaling to millions of chunks would switch to `IndexIVFFlat` (approximate NN) without changing the API.

**When would I choose differently?** If the organisation needed multi-tenancy, real-time index updates, filtered metadata search (e.g., "only search cardiology documents"), or horizontal scaling, I would migrate to Weaviate or Milvus.

---

**Q3. Why Mistral over GPT-4, Claude, or Llama 2?**

Four factors drove this decision:

1. **Local deployment / HIPAA** — GPT-4 and Claude send data to external APIs. For a healthcare system handling anything adjacent to PHI, local inference is the only compliant option without extensive Business Associate Agreements (BAAs) and contractual guarantees. Mistral runs via Ollama — fully on-premises.

2. **Performance per parameter** — Mistral 7B outperforms Llama 2 13B on most benchmarks despite being half the size. Its sliding window attention and grouped-query attention make it efficient on CPU/single-GPU inference.

3. **Instruction following** — The Mistral Instruct variant reliably follows structured output formats (ANSWER/CONFIDENCE/SOURCES/DISCLAIMER). This is critical for our response parser — if the model deviates, parsing fails.

4. **Open licence** — Apache 2.0 licence allows commercial deployment without royalties.

**Trade-off:** GPT-4 or Claude would produce better answers for complex, ambiguous questions. For a RAG system where the answer is grounded in retrieved context, the quality gap narrows significantly — the retriever does the heavy lifting, the LLM just synthesises.

---

**Q4. Why MiniLM-L6-v2 for embeddings?**

The `sentence-transformers/all-MiniLM-L6-v2` model was chosen for:

1. **CPU viability** — 80MB model that runs in ~10ms per batch on CPU. No GPU required for a clinic deploying 1-2 servers.

2. **Semantic quality** — Despite being tiny, it achieves 78.9% on the SBERT benchmark (Semantic Textual Similarity tasks) — excellent for English healthcare text.

3. **L2 normalisation** — When vectors are L2-normalised (which we enable), cosine similarity reduces to a dot product — FAISS can use this for efficient retrieval.

4. **Open source** — HuggingFace Hub model, Apache 2.0, no API key or cost.

5. **Consistency** — The same model creates both document embeddings (during ingestion) and query embeddings (at query time). This is mandatory — you cannot mix embedding models between index creation and retrieval.

**When would I upgrade?** For multilingual support, `paraphrase-multilingual-MiniLM-L12-v2`. For higher accuracy on clinical text, `pritamdeka/S-PubMedBert-MS-MARCO` fine-tuned on medical literature.

---

**Q5. Why LangChain? Could you have built this without it?**

Yes — I could build this without LangChain. The framework provides convenience, not magic. I use specific LangChain components:

- **`RecursiveCharacterTextSplitter`** — Smart recursive splitting at paragraph/sentence/word boundaries. Writing this from scratch would take hours.
- **`FAISS.from_documents()`** — Handles batch embedding + index construction in one call.
- **`PromptTemplate`** — Type-safe, versioned prompt management with input variable validation.
- **LCEL (`|` operator)** — Composable chain syntax: `prompt | llm | parser` is readable and easy to trace.
- **Community loaders** — `PyPDFLoader`, `Docx2txtLoader` handle complex document formats.

**Trade-offs of LangChain:**
- `+` Accelerates development significantly for RAG patterns.
- `+` Ecosystem of integrations (swap FAISS for Pinecone in one line).
- `-` Abstraction overhead — debugging is harder when the chain fails silently.
- `-` Frequent API breaking changes between versions.
- `-` Some overhead from Python object instantiation in hot paths.

For a production system at scale, I'd evaluate whether to keep LangChain or replace the critical path with direct library calls.

---

## Part B: RAG & ML Fundamentals

---

**Q6. Walk me through the complete RAG workflow in your system.**

```
1. INGESTION (one-time):
   Files → DocumentLoader → chunks (1000 chars, 200 overlap) 
   → MiniLM embeddings (384-dim) → FAISS index → disk

2. QUERY (real-time):
   User query → Pydantic validation → safety checks
   → FAISS: embed query → find top-5 similar chunks (MMR)
   → check: max_similarity < 0.25? → refuse (no LLM call)
   → format context: "[Doc 1|Source:X|Rel:0.92]\n..."
   → fill RAG_PROMPT with {context, question}
   → Mistral generates: ANSWER/CONFIDENCE/SOURCES/DISCLAIMER
   → parse structured response
   → blend confidence: 0.6*FAISS + 0.4*LLM_self_report
   → build SourceCitation[] from retrieved chunks
   → return AskResponse
```

The critical insight: **the retriever determines answer quality, not the LLM**. If irrelevant chunks are retrieved, even GPT-4 will generate a poor answer. MMR (Maximum Marginal Relevance) helps by selecting diverse chunks rather than 5 near-identical passages.

---

**Q7. How do you prevent hallucinations?**

Four-layer defence:

**Layer 1 — Input filtering:** Emergency keywords and diagnosis patterns are caught before the LLM is called. These return canned responses with zero hallucination risk.

**Layer 2 — FAISS threshold guard:** If the maximum cosine similarity between the query and any retrieved chunk is below 0.25, the system returns a refusal without calling the LLM. This handles out-of-domain questions cleanly.

**Layer 3 — Prompt-level grounding:** The system prompt contains explicit rules: "Answer ONLY using information explicitly found in the context. Never speculate." The REFUSAL MESSAGE is included verbatim so the model copy-pastes it rather than creatively paraphrasing (which can introduce errors).

**Layer 4 — Confidence blending:** The final confidence score is 60% FAISS evidence + 40% LLM self-report. This prevents the model from claiming HIGH confidence when retrieved chunks are only weakly relevant (similarity 0.3 would pull the score down even if the LLM says HIGH).

**Temperature = 0.1** is also a hallucination control — near-zero temperature means the model takes the highest-probability path, eliminating creative deviation.

---

**Q8. What is Maximum Marginal Relevance (MMR) and why do you use it?**

MMR is a retrieval algorithm that balances **relevance** (how similar a chunk is to the query) and **diversity** (how different chunks are from each other).

**Standard top-K retrieval problem:** If you retrieve the 5 most similar chunks to "telehealth hours", you might get 5 nearly-identical paragraphs from the same section of the telehealth policy. The LLM's context window is filled with redundant information.

**MMR solution:** Iteratively select the next chunk that maximises:
```
λ × similarity(chunk, query) - (1-λ) × max(similarity(chunk, selected_chunks))
```
With `λ=0.7` (our setting), 70% weight on relevance + 30% penalty for redundancy.

**Result:** The 5 retrieved chunks cover different aspects of the question — the LLM gets richer context and produces a more complete answer.

---

**Q9. How does your confidence scoring work?**

```python
# FAISS score: mean of top-3 similarity scores (cosine, 0-1)
faiss_score = mean(sorted(similarity_scores, reverse=True)[:3])

# LLM self-report: mapped to numeric (model writes HIGH/MEDIUM/LOW/NONE)
llm_score = {HIGH: 0.9, MEDIUM: 0.65, LOW: 0.35, NONE: 0.0}[llm_confidence]

# Weighted blend: FAISS evidence dominates
confidence_score = 0.6 * faiss_score + 0.4 * llm_score
```

**Why 60/40?** FAISS similarity is a more objective signal — it measures actual vector distance. LLM self-reported confidence can be overconfident (sycophancy) or underconfident. We trust the retrieval evidence more than the model's self-assessment.

**Level mapping:**
- ≥ 0.75 → HIGH
- ≥ 0.50 → MEDIUM
- ≥ 0.25 → LOW
- < 0.25 → NONE (triggers refusal even if LLM generated an answer)

---

**Q10. What chunk size did you choose and why?**

`chunk_size=1000, chunk_overlap=200`.

**Why 1000 characters:**
- Typical paragraph in a healthcare policy document is 200-500 characters.
- 1000 characters captures 2-4 paragraphs — enough to provide meaningful context to the LLM but not so long that one chunk dominates the entire retrieval.
- MiniLM-L6-v2 has a 256-token limit — 1000 characters ≈ 200-250 tokens, safely within bounds.
- Longer chunks reduce the number of chunks (fewer vectors to search) but reduce specificity. Shorter chunks increase specificity but may lose inter-sentence context.

**Why 200 characters overlap:**
- 20% overlap ensures that information near chunk boundaries isn't lost.
- A sentence split across two chunks will appear fully in at least one of them.
- Without overlap, a question about information that straddles a chunk boundary may fail to retrieve that content.

**Experiment guidance:** For dense clinical notes, reduce to 512 characters. For legal/regulatory documents with long structured sections, consider 1500 characters.

---

## Part C: Agent & Tool Design

---

**Q11. Explain your agent routing architecture.**

The router uses a two-tier approach:

**Tier 1 — Keyword scan (primary):**
A `frozenset` of 20 appointment-related terms is checked against the tokenised query using set intersection — O(n) where n = query tokens. This handles >95% of appointment queries in microseconds.

```python
APPOINTMENT_KEYWORDS = frozenset({"appointment", "book", "schedule", "availability", ...})
tokens = set(query.lower().split())
is_appointment = bool(tokens & APPOINTMENT_KEYWORDS)
```

**Why keywords over LLM classification?**
- 100x faster (μs vs seconds)
- Deterministic — no ambiguity, no model variance
- No API cost / latency for a simple intent classification
- The appointment domain is well-defined with clear vocabulary

**Tier 2 — LLM classifier (fallback):**
An optional `CLASSIFICATION_PROMPT` is available for ambiguous queries that pass keyword scanning. It instructs Mistral to return a single token: `APPOINTMENT` or `KNOWLEDGE`. Not used in the primary path — available for future refinement.

---

**Q12. How would you replace the mock appointment tool with a real system?**

The `check_available_slots()` function is designed as a drop-in replacement point:

```python
# Current (mock)
def check_available_slots(query: str) -> str:
    # Returns pseudo-random slots as formatted string
    return generated_slots_string

# Production replacement
def check_available_slots(query: str) -> str:
    # 1. Extract date/time preferences from query (NER or regex)
    preferences = extract_scheduling_preferences(query)
    
    # 2. Call EHR FHIR API (Epic/Cerner/Athena)
    response = fhir_client.get_slots(
        service_type="primary-care",
        start=preferences.start_date,
        end=preferences.end_date,
        practitioner=preferences.preferred_provider,
    )
    
    # 3. Format FHIR Slot resources as human-readable string
    return format_fhir_slots(response.json())
```

Additional production considerations:
- Authentication: OAuth2 SMART on FHIR for EHR integration
- Rate limiting: FHIR APIs have request limits
- Caching: Cache slot availability for 60 seconds
- Error handling: If FHIR API fails, return graceful fallback message

---

## Part D: Healthcare & Privacy

---

**Q13. What HIPAA considerations apply to this system?**

**Technical Safeguards implemented:**
- Local LLM inference — no ePHI transmitted to external APIs
- JSON audit logs with timestamps and request IDs (addressable requirement)
- Input validation prevents injection attacks that could exfiltrate data
- No PHI stored — the system indexes policy documents, not patient records

**For full HIPAA compliance in production, add:**
- TLS 1.2+ on all endpoints (nginx/Traefik reverse proxy)
- JWT authentication with role-based access (who can call /ask vs /ingest)
- At-rest encryption for the FAISS index volume (AES-256)
- BAA signed with any cloud provider hosting the containers
- Minimum Necessary analysis: does the chatbot need to see PHI?
- Access controls: only authorized workforce members can access the system
- Incident response plan for data breaches

**Key point:** This system is intentionally designed to answer from policy documents, not patient records. If it were modified to access patient EHR data, the HIPAA obligations would increase substantially.

---

**Q14. How do you handle requests for medical diagnosis?**

Three-layer defence:

1. **Keyword pre-filter:** DIAGNOSIS_KEYWORDS list checked before any LLM call. Patterns like "diagnose me", "do I have", "prescribe", "what's wrong with me" trigger an immediate canned refusal.

2. **Prompt-level rule:** The RAG prompt contains: "NEVER provide a medical diagnosis, prescribe treatments, or recommend specific medications for individual patients."

3. **Canned response:** `DIAGNOSIS_REFUSAL_RESPONSE` returns a message directing the user to a licensed provider — not a vague deflection, but a clear explanation of why the system won't answer.

This matters clinically: if an AI system provides a wrong diagnosis or medication recommendation, the consequences are patient harm. The legal liability for the healthcare organisation is also significant.

---

**Q15. What happens when a user asks about a medical emergency?**

The system checks for emergency keywords (chest pain, heart attack, stroke, seizure, severe bleeding, suicidal, etc.) **before any other processing** — before the FAISS lookup, before the LLM call.

If triggered, it returns `EMERGENCY_RESPONSE` immediately:

```
🚨 EMERGENCY DETECTED — If you or someone else is experiencing a medical emergency,
call 911 (or your local emergency number) IMMEDIATELY. Do not rely on an AI assistant
for emergency medical guidance. Stay on the line with emergency services.
```

The confidence score is set to 1.0 (HIGH) — we are highly confident this is the correct response to an emergency query.

**Why fast-path this?** An LLM call takes 2-10 seconds. In a genuine emergency, those seconds matter. The keyword check takes microseconds.

---

## Part E: Infrastructure & Operations

---

**Q16. Walk through your Docker deployment architecture.**

Three services in `docker-compose.yml`:

**`ollama`** — Base image `ollama/ollama:latest`. Custom entrypoint runs `ollama serve` then pulls the Mistral model on first boot. Data persisted in Docker volume `ollama_data`. Health check pings `/api/tags`.

**`backend`** — Multi-stage Dockerfile. Builder stage installs Python dependencies into `/opt/venv`. Runtime stage copies only the venv and source code (no build tools) — smaller final image. Non-root user `appuser` for security. FAISS index mounted as volume `faiss_index` (survives restarts without re-ingestion). Health check calls `/api/v1/health`.

**`frontend`** — Same image as backend (different CMD). Streamlit runs on port 8501. Configured with `BACKEND_URL=http://backend:8000` (internal Docker network).

**Startup dependency chain:** `backend` depends on `ollama` (healthy) → `frontend` depends on `backend` (healthy).

---

**Q17. How does your structured logging support observability?**

The `JSONFormatter` emits each log record as a single JSON line:

```json
{
  "timestamp": "2024-01-15T10:23:45.123Z",
  "level": "INFO",
  "logger": "backend.rag",
  "request_id": "a3f9b2c1",
  "message": "RAG pipeline completed",
  "confidence_level": "HIGH",
  "confidence_score": 0.87,
  "citations": 3,
  "processing_time_s": 2.14
}
```

**Benefits:**
- `request_id` (set by middleware, propagated via `ContextVar`) links all log lines for one HTTP request across multiple modules.
- Machine-parseable: Datadog, ELK, CloudWatch can filter/aggregate on any field.
- HIPAA audit: every query logged with timestamp and request ID (no PHI in logs — only query previews of 80 chars).
- Performance tracking: `processing_time_s` logged at pipeline completion enables P95/P99 latency monitoring.

---

## Part F: Extended Q&A (Q18–Q50)

---

**Q18. How would you scale this system to handle 10,000 queries/day?**

At 10k queries/day ≈ 7 queries/minute peak, a single backend instance handles this easily (assuming 2-5s per LLM response). Scaling beyond:

- **LLM bottleneck:** Add GPU to Ollama container (A10G handles ~50 concurrent Mistral requests). Or deploy vLLM for higher throughput via continuous batching.
- **FAISS bottleneck:** Unlikely — FAISS retrieval is <5ms even for 1M vectors. For millions of chunks, switch to `IndexIVFFlat` with HNSW for approximate NN.
- **Backend scaling:** Stateless FastAPI behind an nginx load balancer. Share FAISS index via a shared NFS volume or migrate to a distributed vector store.
- **Embedding bottleneck:** Batch embed queries (if doing bulk processing). Cache embeddings for frequently-asked questions.

---

**Q19. What would you add if you had 2 more weeks?**

Priority additions:
1. **JWT authentication** — Protect all endpoints, add role-based access (admin can ingest, users can only ask).
2. **Conversation memory** — Store conversation history in Redis; pass last N turns in the RAG prompt for multi-turn coherence.
3. **Feedback loop** — Thumbs up/down per answer; store in database; use to identify poorly-answered questions for document improvements.
4. **Streaming responses** — FastAPI SSE + Streamlit `st.write_stream` to display Mistral's output token-by-token (dramatically improves perceived latency).
5. **Document management API** — `DELETE /documents/{name}`, `GET /documents` listing, per-document re-indexing.
6. **Evaluation harness** — Automated RAGAS metrics (faithfulness, answer relevancy, context recall) against a golden Q&A dataset.

---

**Q20. How do you evaluate RAG pipeline quality?**

Standard RAG evaluation uses the **RAGAS framework** with four metrics:

1. **Faithfulness:** Are all claims in the answer supported by the retrieved context? (Measured by NLI model checking each sentence against context.)

2. **Answer Relevancy:** Does the answer address the question? (Measured by generating reverse questions from the answer and checking similarity to original.)

3. **Context Recall:** Did retrieval find all the necessary information? (Requires ground-truth answers; checks if context contains the information needed.)

4. **Context Precision:** Are the retrieved chunks actually useful, or is there noise? (Proportion of retrieved chunks that contributed to the answer.)

**Practical approach:** Build a golden dataset of 50 question-answer pairs from the healthcare documents. Run RAGAS on every code change. Set quality gates (faithfulness > 0.85, relevancy > 0.80) in CI/CD.

---

**Q21. What are the failure modes of your system?**

| Failure | Symptom | Mitigation |
|---|---|---|
| Ollama service down | 503 from /ask | Health check alerts; Ollama auto-restart in Docker |
| FAISS index not built | 503 "call /ingest first" | Startup hook tries to load from disk; clear UI message |
| Low similarity, bad answer | NONE confidence | Threshold guard prevents LLM call; returns refusal |
| LLM timeout (>120s) | 504 Gateway Timeout | Retry logic in frontend; reduce max_tokens |
| Mistral ignores format | Parser fallback | Regex with fallbacks; treats full output as answer |
| Out-of-domain question | Low FAISS similarity | Threshold guard + "I don't know" refusal |
| Prompt injection | Validation catches patterns | Pydantic validator with regex; deny-list |
| Model hallucination | Answer contradicts context | 4-layer defence as described in Q7 |

---

**Q22. How do you handle multi-turn conversations?**

Currently: `conversation_id` is logged but not used for context injection. Each question is answered independently.

**To add true multi-turn:**
```python
# In agent.py
def route_query(request: AskRequest) -> AskResponse:
    # Retrieve conversation history
    history = conversation_store.get(request.conversation_id, [])
    
    # Inject last 3 turns into the prompt
    history_text = "\n".join([
        f"User: {turn['query']}\nAssistant: {turn['answer'][:200]}..."
        for turn in history[-3:]
    ])
    
    # Modified RAG prompt includes history
    context_with_history = f"Previous conversation:\n{history_text}\n\n{context}"
    
    # Store this turn
    conversation_store[request.conversation_id].append({
        "query": request.query, "answer": response.answer
    })
```

Use Redis with TTL for conversation storage — stateless backend, scalable.

---

**Q23. How do you handle documents in different formats?**

The `DocumentLoader` maps file extensions to LangChain loaders:
- `.txt` → `TextLoader` (UTF-8 text, simplest)
- `.md` → `UnstructuredMarkdownLoader` (strips markdown syntax, preserves structure)
- `.pdf` → `PyPDFLoader` (page-by-page extraction; metadata includes page number)
- `.docx` → `Docx2txtLoader` (extracts plain text from Word documents)

For production clinical documents, additional formats may need:
- `.html` → `BSHTMLLoader` (web-based policies)
- Scanned PDFs → `pytesseract` OCR pipeline before `TextLoader`
- EHR CDA/HL7 documents → Custom FHIR parser

---

**Q24. Why do you use `@lru_cache` on `get_embeddings()` and `get_llm()`?**

These functions load large models (80MB for MiniLM, GB-scale for Mistral client setup) that should only be initialised once per process.

`@lru_cache(maxsize=1)` caches the return value after the first call. Every subsequent call returns the same cached instance. This means:
- Model loads once at startup (or first request), not per query.
- Saves 500ms-2s per request (model loading time).
- Consistent: all requests use the same model instance (no state divergence).

This is equivalent to the Singleton pattern but more Pythonic.

---

**Q25. What is the purpose of the `MIN_CONFIDENCE_THRESHOLD`?**

`MIN_CONFIDENCE_THRESHOLD = 0.25` is the minimum FAISS cosine similarity score required to proceed with an LLM call.

If all 5 retrieved chunks have similarity < 0.25, the question is considered out-of-scope for the knowledge base, and a refusal is returned without calling Mistral.

**Why 0.25?** This is an empirical threshold:
- MiniLM cosine similarity for completely unrelated text is typically 0.0-0.15.
- Loosely related text (same domain but different topic) is 0.15-0.30.
- Relevant text is typically 0.50+.

At 0.25, we catch clearly out-of-scope questions (quantum computing, cooking recipes) while still attempting to answer loosely healthcare-adjacent questions where context might be partially relevant.

This threshold should be tuned on a validation set — track refusal rate and precision.

---

**Q26–Q50: Rapid-fire Q&A**

**Q26. What HTTP status codes do you return?**
- 200: Success (ask, ingest, health)
- 422: Validation error (bad request body, injection detected, short query)
- 404: Documents directory not found
- 503: Service unavailable (index not loaded, Ollama down)
- 500: Unexpected internal error (caught by global exception handler)

**Q27. How does your prompt injection prevention work?**
Pydantic `@field_validator` on `AskRequest.query` uses regex to match patterns like "ignore previous instructions", "you are now", "act as if". If matched, Pydantic raises a `ValueError` → FastAPI returns 422 before the query reaches any pipeline.

**Q28. What is RecursiveCharacterTextSplitter?**
A LangChain text splitter that tries separators in order (`\n\n`, `\n`, `. `, ` `, `""`), recursively splitting chunks that exceed `chunk_size` using the next separator. This preserves semantic boundaries (paragraphs > sentences > words) better than a fixed-character split.

**Q29. How do source citations work?**
After retrieval, each `(Document, score)` pair carries metadata including `source` (filename). After LLM response parsing, `_build_citations()` matches LLM-cited source names against retrieved documents, deduplicates by filename, and builds `SourceCitation` objects with excerpts (first 400 chars of the chunk).

**Q30. What is the LCEL chain syntax?**
LangChain Expression Language: `prompt | llm | parser`. The `|` operator composes `Runnable` objects. `chain.invoke({"context": ctx, "question": q})` passes data through each stage. Equivalent to `parser(llm(prompt.format(...)))` but composable, traceable, and async-compatible.

**Q31. Why async FastAPI but sync LangChain?**
FastAPI is `async` at the route level, but LangChain's `chain.invoke()` is synchronous (it calls Ollama over HTTP synchronously). In production, wrap sync calls in `asyncio.run_in_executor()` to avoid blocking the event loop under concurrent requests. For this demo, single-worker uvicorn is sufficient.

**Q32. How do you handle the Ollama startup time?**
`docker-compose` `depends_on` with `condition: service_healthy` ensures the backend waits for Ollama's health check to pass. The backend's startup hook (`load_index()`) is non-blocking — it logs a warning and continues if the index isn't found.

**Q33. What is the `request_id_var` ContextVar for?**
Python `contextvars.ContextVar` provides per-async-task storage (like thread-local but for async). The middleware sets a unique request ID for each incoming request. Every logger call in any downstream function reads this var and includes the ID in the JSON log. This enables tracing a complete request across log lines.

**Q34. How do you test without a real LLM or FAISS index?**
By patching at import time using `unittest.mock.patch` as a context manager: `with patch("backend.embeddings.get_embeddings"):`. This prevents torch from loading during tests. Individual test functions patch `route_query`, `similarity_search_with_scores`, and `get_llm` with mock objects.

**Q35. What is `force_reload` in the ingest endpoint?**
If `force_reload=True`, the existing FAISS index directory is deleted before rebuilding. Used when documents have been updated and a full re-index is needed. Without it, calling `/ingest` twice rebuilds on top of the existing index.

**Q36. Why store FAISS on a Docker volume?**
A named Docker volume (`faiss_index`) persists the FAISS index across container restarts and rebuilds. Without it, every container restart would require re-running `/ingest` (which takes 15-30s for the demo documents, but minutes for a larger corpus).

**Q37. What does the health endpoint check?**
Two components: (1) FAISS — `_faiss_store is not None` and `index.ntotal > 0`. (2) Ollama — HTTP GET to `/api/tags` and checks that `mistral` appears in the model list. Overall status: healthy if both are healthy; degraded if FAISS not loaded; unhealthy if Ollama unreachable.

**Q38. How do you avoid returning sources from unrelated documents?**
`_build_citations()` checks whether each retrieved document's filename appears in the LLM's SOURCES list. Only documents that the LLM cited are included. If the LLM didn't parse any sources, all retrieved documents are included (fallback). This prevents citing documents that happened to be retrieved but weren't actually used.

**Q39. What is the MMR lambda parameter?**
`lambda_mult=0.7` in `as_retriever(search_type="mmr")`. At λ=1.0, MMR is identical to standard similarity search. At λ=0.0, it maximises diversity (retrieves maximally different chunks). λ=0.7 means 70% relevance + 30% diversity — biased toward relevance while avoiding redundant chunks.

**Q40. How do you validate that the LLM's answer is grounded?**
Current approach: FAISS threshold + structured prompt + blended confidence. A more rigorous approach would use NLI (Natural Language Inference) to check each sentence in the answer against retrieved chunks (RAGAS faithfulness metric). For production, add a post-generation faithfulness checker.

**Q41. What Python version and why 3.11?**
Python 3.11 introduced significant performance improvements (~25% faster than 3.10 via specialised adaptive interpreter). Pydantic v2 requires 3.8+. FastAPI recommends 3.11+. `asyncio` is stable and performant on 3.11.

**Q42. What is the Docx2txtLoader limitation?**
`Docx2txtLoader` extracts plain text, losing formatting (tables, headers become flat text). For documents with critical tabular data (e.g., insurance co-pay tables), use `python-docx` with a custom loader that preserves table structure. For this demo, the policy documents are prose-heavy, making this limitation acceptable.

**Q43. How does the Streamlit frontend handle errors gracefully?**
Four error cases handled: (1) `requests.ConnectionError` → "Cannot connect to backend"; (2) `requests.Timeout` → "LLM may be overloaded"; (3) API 503 with `detail` field → shows message + "click Ingest Documents"; (4) API error with `error` field → shows error message. Never crashes the UI.

**Q44. Why not use OpenAI embeddings?**
OpenAI embeddings (text-embedding-3-small) are excellent but: (1) Require API key and internet access; (2) Healthcare queries/documents sent to OpenAI servers — potential HIPAA concern; (3) Cost: ~$0.02/1M tokens; (4) Vendor lock-in. MiniLM-L6-v2 is 100% local, free, and sufficient for English healthcare text.

**Q45. How would you add authentication?**
Add FastAPI middleware: `from fastapi.security import OAuth2PasswordBearer`. Add JWT verification in a dependency: `Depends(verify_token)`. Use `python-jose` for JWT decoding. For enterprise, integrate with existing IdP (Okta, Azure AD) via OIDC. Add role claims: `ADMIN` (can ingest), `USER` (can ask), `VIEWER` (read-only health).

**Q46. What is the `repeat_penalty=1.1` in the LLM?**
A penalty applied to the log-probability of tokens that have appeared in the context. Value > 1.0 reduces repetition. At 1.1, tokens that appeared in the retrieved context are slightly penalised, preventing the model from simply echoing back the context verbatim instead of synthesising an answer.

**Q47. How do you handle concurrent requests?**
FastAPI uses asyncio — multiple requests are handled concurrently in a single-threaded event loop. The LangChain `chain.invoke()` is synchronous (blocking). Under high concurrency, this blocks the event loop. Solution: `await asyncio.get_event_loop().run_in_executor(None, chain.invoke, inputs)` to run sync LLM calls in a thread pool. With `--workers 4` in uvicorn, 4 concurrent LLM calls are possible.

**Q48. What would you do differently in production?**
Top 5 changes: (1) Add JWT auth; (2) Move to async LangChain (`chain.ainvoke`); (3) Add Redis for conversation history caching; (4) Implement RAGAS evaluation in CI; (5) Add Prometheus metrics (`/metrics` endpoint) for query latency histograms and confidence score distributions.

**Q49. How do you ensure the FAISS index and embedding model stay in sync?**
The `get_embeddings()` singleton returns the same model instance for both index creation (in `build_index`) and query time (in `similarity_search_with_scores`). If the `EMBEDDING_MODEL` config changes, `force_reload=True` must be passed to `/ingest` to rebuild the index with the new model. Mixing models (index created with model A, queries with model B) produces garbage results.

**Q50. What's the hardest technical challenge in building this system?**
Calibrating the confidence threshold. Too low (0.10) → the system answers out-of-domain questions with low-quality responses. Too high (0.50) → the system refuses too many valid questions. The right threshold depends on:
- The quality of your documents (well-written = higher similarity scores)
- The specificity of user questions (vague questions → lower scores)
- Your risk tolerance (healthcare: prefer refusal over wrong answer)

The solution is to build a labeled evaluation set of (question, expected_answer, should_answer: bool) tuples and measure precision/recall of the refusal system across thresholds. Pick the threshold that maximises F1 or meets your precision requirement.
