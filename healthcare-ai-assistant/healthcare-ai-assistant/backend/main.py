"""
main.py - FastAPI application entry point for the Healthcare AI Assistant.

Responsibilities:
  1. Create and configure the FastAPI app instance.
  2. Register all routers (health, ingest, ask).
  3. Configure CORS middleware.
  4. Add request-ID injection middleware for distributed tracing.
  5. Run startup lifecycle: load FAISS index if it exists on disk.
  6. Expose a root endpoint for quick sanity checks.

Run locally:
  uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload

Run in Docker:
  CMD set in Dockerfile — docker-compose handles port binding.
"""

import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from backend.config import settings
from backend.logger import get_logger, request_id_var
from backend.routers import ask, health, ingest
from backend.vector_store import load_index

logger = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# LIFESPAN — startup & shutdown hooks
# ─────────────────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Runs once at startup (before accepting traffic) and once at shutdown.
    Startup: attempt to load an existing FAISS index from disk.
    """
    logger.info("Healthcare AI Assistant starting up …", extra={"version": settings.APP_VERSION})

    # Try to load a previously built index — speeds up restart without re-ingesting
    loaded = load_index()
    if loaded:
        logger.info("FAISS index loaded from disk at startup ✓")
    else:
        logger.warning(
            "No FAISS index found — call POST /ingest to initialise the knowledge base."
        )

    yield  # ← application runs here

    logger.info("Healthcare AI Assistant shutting down.")


# ─────────────────────────────────────────────────────────────────────────────
# APP FACTORY
# ─────────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description=(
        "A Retrieval-Augmented Generation (RAG) Healthcare AI Assistant. "
        "Answers questions from uploaded healthcare documents, "
        "provides source citations, and routes appointment queries "
        "to an integrated scheduling tool.\n\n"
        "**Important:** This system does not provide medical diagnosis or "
        "personalised treatment advice. Always consult a qualified healthcare provider."
    ),
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)


# ─────────────────────────────────────────────────────────────────────────────
# MIDDLEWARE
# ─────────────────────────────────────────────────────────────────────────────

# CORS — open for development; restrict origins in production
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def request_id_middleware(request: Request, call_next) -> Response:
    """
    Inject a unique request ID into every request context.
    The ID flows through structured logs for end-to-end tracing.
    """
    req_id = request.headers.get("X-Request-ID", str(uuid.uuid4())[:8])
    request_id_var.set(req_id)

    t0 = time.perf_counter()
    response = await call_next(request)
    elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)

    response.headers["X-Request-ID"] = req_id
    response.headers["X-Response-Time-Ms"] = str(elapsed_ms)

    logger.info(
        "HTTP request",
        extra={
            "method": request.method,
            "path": str(request.url.path),
            "status_code": response.status_code,
            "duration_ms": elapsed_ms,
        },
    )
    return response


# ─────────────────────────────────────────────────────────────────────────────
# EXCEPTION HANDLERS
# ─────────────────────────────────────────────────────────────────────────────

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Catch-all handler — prevents stack traces leaking to clients."""
    logger.error(
        "Unhandled exception",
        extra={"path": str(request.url.path), "error": str(exc)},
        exc_info=True,
    )
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": "An unexpected error occurred. Please try again.",
            "request_id": request_id_var.get("unknown"),
        },
    )


# ─────────────────────────────────────────────────────────────────────────────
# ROUTERS
# ─────────────────────────────────────────────────────────────────────────────

app.include_router(health.router, prefix="/api/v1")
app.include_router(ingest.router, prefix="/api/v1")
app.include_router(ask.router,    prefix="/api/v1")


# ─────────────────────────────────────────────────────────────────────────────
# ROOT ENDPOINT
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/", tags=["System"], summary="Root")
async def root():
    """Quick sanity-check endpoint."""
    return {
        "app": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "docs": "/docs",
        "health": "/api/v1/health",
    }


# ─────────────────────────────────────────────────────────────────────────────
# DIRECT RUN (development only)
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "backend.main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=settings.DEBUG,
        log_level=settings.LOG_LEVEL.lower(),
    )
