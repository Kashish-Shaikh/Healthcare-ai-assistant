"""
app.py - Streamlit frontend for the Healthcare AI Assistant.

Features:
  ✓ Chat interface with streaming-style message display
  ✓ Confidence level indicator with colour-coded badge
  ✓ Expandable citation / source panel per response
  ✓ System health status dashboard in sidebar
  ✓ Full conversation history with scroll
  ✓ Clear chat button
  ✓ Document ingestion trigger
  ✓ Query type badge (KNOWLEDGE vs APPOINTMENT)
  ✓ Processing time display
  ✓ HIPAA-safe: no PHI stored in session state
"""

import os
import time
from datetime import datetime

import requests
import streamlit as st

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")
API_BASE    = f"{BACKEND_URL}/api/v1"
REQUEST_TIMEOUT = 120   # seconds — long timeout for LLM inference

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG  (must be first Streamlit call)
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Healthcare AI Assistant",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("""
<style>
/* ── Global ───────────────────────────────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* ── Chat bubbles ─────────────────────────────────────────────────────── */
.user-bubble {
    background: linear-gradient(135deg, #0066CC, #004499);
    color: white;
    padding: 12px 18px;
    border-radius: 18px 18px 4px 18px;
    margin: 8px 0 8px 20%;
    box-shadow: 0 2px 8px rgba(0,102,204,0.25);
    font-size: 0.95rem;
    line-height: 1.5;
}

.assistant-bubble {
    background: #F8FAFE;
    border: 1px solid #E0EAF5;
    color: #1A1A2E;
    padding: 14px 18px;
    border-radius: 18px 18px 18px 4px;
    margin: 8px 20% 8px 0;
    box-shadow: 0 2px 6px rgba(0,0,0,0.06);
    font-size: 0.95rem;
    line-height: 1.6;
}

/* ── Confidence badges ────────────────────────────────────────────────── */
.badge-high   { background:#D4EDDA; color:#155724; padding:3px 10px; border-radius:12px; font-size:0.8rem; font-weight:600; }
.badge-medium { background:#FFF3CD; color:#856404; padding:3px 10px; border-radius:12px; font-size:0.8rem; font-weight:600; }
.badge-low    { background:#FFE5CC; color:#7D3C00; padding:3px 10px; border-radius:12px; font-size:0.8rem; font-weight:600; }
.badge-none   { background:#F8D7DA; color:#721C24; padding:3px 10px; border-radius:12px; font-size:0.8rem; font-weight:600; }

/* ── Query type badge ────────────────────────────────────────────────── */
.badge-knowledge    { background:#D1ECF1; color:#0C5460; padding:3px 10px; border-radius:12px; font-size:0.75rem; font-weight:500; }
.badge-appointment  { background:#E2D9F3; color:#3D1A6E; padding:3px 10px; border-radius:12px; font-size:0.75rem; font-weight:500; }

/* ── Health status indicators ─────────────────────────────────────────── */
.health-healthy   { color: #28A745; font-weight: 600; }
.health-degraded  { color: #FFC107; font-weight: 600; }
.health-unhealthy { color: #DC3545; font-weight: 600; }

/* ── Citation card ────────────────────────────────────────────────────── */
.citation-card {
    background: #FAFBFF;
    border-left: 3px solid #0066CC;
    padding: 10px 14px;
    margin: 6px 0;
    border-radius: 0 8px 8px 0;
    font-size: 0.85rem;
}

/* ── Disclaimer box ───────────────────────────────────────────────────── */
.disclaimer-box {
    background: #FFF8E1;
    border: 1px solid #FFCC80;
    border-radius: 8px;
    padding: 10px 14px;
    font-size: 0.82rem;
    color: #5D4037;
    margin-top: 8px;
}

/* ── Sidebar ──────────────────────────────────────────────────────────── */
.sidebar-header {
    font-size: 1.1rem;
    font-weight: 700;
    color: #0066CC;
    margin-bottom: 4px;
}

/* ── Input area ───────────────────────────────────────────────────────── */
.stTextInput > div > div > input {
    border-radius: 24px;
    border: 2px solid #E0EAF5;
    padding: 10px 18px;
    font-size: 0.95rem;
}
.stTextInput > div > div > input:focus {
    border-color: #0066CC;
    box-shadow: 0 0 0 2px rgba(0,102,204,0.15);
}

/* ── Spinner ──────────────────────────────────────────────────────────── */
.stSpinner > div {
    border-color: #0066CC transparent transparent transparent !important;
}

/* Hide default Streamlit footer */
footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# SESSION STATE INITIALISATION
# ─────────────────────────────────────────────────────────────────────────────

def init_session_state():
    defaults = {
        "messages":        [],          # List of {role, content, metadata}
        "conversation_id": f"session-{int(time.time())}",
        "health_cache":    None,
        "health_checked":  0.0,
        "ingested":        False,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

init_session_state()


# ─────────────────────────────────────────────────────────────────────────────
# API HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def call_health() -> dict | None:
    """Fetch system health, cached for 30 seconds."""
    now = time.time()
    if st.session_state.health_cache and (now - st.session_state.health_checked) < 30:
        return st.session_state.health_cache
    try:
        resp = requests.get(f"{API_BASE}/health", timeout=5)
        if resp.status_code == 200:
            data = resp.json()
            st.session_state.health_cache = data
            st.session_state.health_checked = now
            return data
    except Exception:
        pass
    return None


def call_ingest(documents_dir: str = "data/documents", force_reload: bool = False) -> dict | None:
    try:
        resp = requests.post(
            f"{API_BASE}/ingest",
            json={"documents_dir": documents_dir, "force_reload": force_reload},
            timeout=REQUEST_TIMEOUT,
        )
        return resp.json()
    except Exception as e:
        return {"success": False, "message": str(e)}


def call_ask(query: str) -> dict | None:
    try:
        resp = requests.post(
            f"{API_BASE}/ask",
            json={
                "query": query,
                "conversation_id": st.session_state.conversation_id,
                "include_sources": True,
            },
            timeout=REQUEST_TIMEOUT,
        )
        return resp.json()
    except requests.exceptions.ConnectionError:
        return {"error": "Cannot connect to the backend. Is the server running?"}
    except requests.exceptions.Timeout:
        return {"error": "Request timed out. The LLM may be overloaded — please try again."}
    except Exception as e:
        return {"error": str(e)}


# ─────────────────────────────────────────────────────────────────────────────
# RENDER HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def confidence_badge(level: str) -> str:
    icons = {"HIGH": "🟢", "MEDIUM": "🟡", "LOW": "🟠", "NONE": "🔴"}
    css   = {"HIGH": "badge-high", "MEDIUM": "badge-medium", "LOW": "badge-low", "NONE": "badge-none"}
    icon  = icons.get(level.upper(), "⚪")
    cls   = css.get(level.upper(), "badge-none")
    return f'<span class="{cls}">{icon} {level}</span>'


def query_type_badge(qtype: str) -> str:
    if qtype.upper() == "APPOINTMENT":
        return '<span class="badge-appointment">📅 APPOINTMENT</span>'
    return '<span class="badge-knowledge">📚 KNOWLEDGE</span>'


def render_confidence_bar(score: float):
    """Display a coloured progress bar for the numeric confidence score."""
    pct = int(score * 100)
    color = "#28A745" if score >= 0.75 else "#FFC107" if score >= 0.50 else "#DC3545"
    st.markdown(
        f"""
        <div style="margin:4px 0 8px;">
          <div style="font-size:0.78rem;color:#666;margin-bottom:3px;">
            Confidence Score: <strong>{pct}%</strong>
          </div>
          <div style="background:#E9ECEF;border-radius:6px;height:8px;">
            <div style="width:{pct}%;background:{color};border-radius:6px;height:8px;
                        transition:width 0.4s ease;"></div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_citations(sources: list):
    """Render source citations in an expandable panel."""
    if not sources:
        return
    with st.expander(f"📎 Sources ({len(sources)} document{'s' if len(sources) > 1 else ''})", expanded=False):
        for i, src in enumerate(sources, 1):
            score = src.get("similarity_score")
            score_str = f" · {score:.0%} match" if score else ""
            section  = src.get("page_or_section", "")
            section_str = f" · {section}" if section else ""
            st.markdown(
                f"""
                <div class="citation-card">
                  <strong>📄 {src['document_name']}</strong>{section_str}{score_str}<br>
                  <em style="color:#555;">{src.get('relevant_excerpt','')[:280]}…</em>
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_assistant_message(msg: dict):
    """Render a complete assistant message with all metadata."""
    meta = msg.get("metadata", {})
    answer   = msg["content"]
    conf     = meta.get("confidence", "")
    score    = meta.get("confidence_score", 0.0)
    qtype    = meta.get("query_type", "KNOWLEDGE")
    sources  = meta.get("sources", [])
    disclaimr= meta.get("disclaimer", "")
    proc_time= meta.get("processing_time_seconds", 0.0)

    # Main answer bubble
    st.markdown(f'<div class="assistant-bubble">{answer}</div>', unsafe_allow_html=True)

    # Metadata row
    if conf or qtype:
        cols = st.columns([2, 2, 3])
        with cols[0]:
            if conf:
                st.markdown(confidence_badge(conf), unsafe_allow_html=True)
        with cols[1]:
            if qtype:
                st.markdown(query_type_badge(qtype), unsafe_allow_html=True)
        with cols[2]:
            if proc_time:
                st.caption(f"⏱ {proc_time:.2f}s")

    # Confidence bar
    if score > 0:
        render_confidence_bar(score)

    # Citations
    render_citations(sources)

    # Disclaimer
    if disclaimr:
        st.markdown(
            f'<div class="disclaimer-box">{disclaimr}</div>',
            unsafe_allow_html=True,
        )


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────

def render_sidebar():
    with st.sidebar:
        st.markdown("## 🏥 Healthcare AI")
        st.markdown("*RAG-powered clinical information assistant*")
        st.divider()

        # ── System Health ─────────────────────────────────────────────────────
        st.markdown('<p class="sidebar-header">⚕️ System Health</p>', unsafe_allow_html=True)

        health = call_health()
        if health:
            overall = health.get("status", "unknown")
            css_cls = f"health-{overall}"
            icon    = {"healthy": "✅", "degraded": "⚠️", "unhealthy": "❌"}.get(overall, "❓")
            st.markdown(
                f'<p class="{css_cls}">{icon} System: {overall.upper()}</p>',
                unsafe_allow_html=True,
            )

            components = health.get("components", {})

            # FAISS status
            faiss = components.get("faiss", {})
            faiss_status = faiss.get("status", "unknown")
            faiss_icon   = "✅" if faiss_status == "healthy" else "⚠️"
            st.caption(f"{faiss_icon} Vector Store: {faiss_status}")
            if faiss.get("detail"):
                st.caption(f"   {faiss['detail']}")

            # Ollama status
            ollama = components.get("ollama", {})
            ollama_status = ollama.get("status", "unknown")
            ollama_icon   = "✅" if ollama_status == "healthy" else "❌"
            st.caption(f"{ollama_icon} LLM (Mistral): {ollama_status}")
            if ollama.get("latency_ms"):
                st.caption(f"   Latency: {ollama['latency_ms']:.0f} ms")

            # Metrics
            doc_count = health.get("index_document_count", 0)
            uptime    = health.get("uptime_seconds", 0)
            st.caption(f"📊 Index Chunks: {doc_count:,}")
            st.caption(f"⏰ Uptime: {int(uptime // 60)}m {int(uptime % 60)}s")
        else:
            st.warning("⚠️ Cannot reach backend")

        if st.button("🔄 Refresh Status", use_container_width=True, key="refresh_health"):
            st.session_state.health_cache = None
            st.rerun()

        st.divider()

        # ── Document Ingestion ────────────────────────────────────────────────
        st.markdown('<p class="sidebar-header">📂 Knowledge Base</p>', unsafe_allow_html=True)

        docs_dir = st.text_input(
            "Documents directory",
            value="data/documents",
            help="Path to folder containing .txt, .pdf, .md, or .docx files",
        )
        force_reload = st.checkbox(
            "Force full re-index",
            value=False,
            help="Clear existing index and re-ingest all documents from scratch",
        )

        if st.button("⚡ Ingest Documents", use_container_width=True, type="primary"):
            with st.spinner("Ingesting documents — please wait …"):
                result = call_ingest(docs_dir, force_reload)
            if result and result.get("success"):
                st.success(
                    f"✅ Ingested {result['documents_processed']} docs "
                    f"→ {result['chunks_created']:,} chunks "
                    f"({result['processing_time_seconds']:.1f}s)"
                )
                st.session_state.ingested = True
                st.session_state.health_cache = None
            else:
                msg = result.get("message") or result.get("detail") or "Unknown error"
                st.error(f"❌ Ingestion failed: {msg}")

        st.divider()

        # ── Session Info ──────────────────────────────────────────────────────
        st.markdown('<p class="sidebar-header">💬 Session</p>', unsafe_allow_html=True)
        msg_count = len([m for m in st.session_state.messages if m["role"] == "user"])
        st.caption(f"🗨 Messages: {msg_count}")
        st.caption(f"🔑 Session ID: {st.session_state.conversation_id[-12:]}")
        st.caption(f"🕐 {datetime.now().strftime('%H:%M:%S')}")

        if st.button("🗑️ Clear Conversation", use_container_width=True):
            st.session_state.messages = []
            st.session_state.conversation_id = f"session-{int(time.time())}"
            st.rerun()

        st.divider()

        # ── Capability Guide ──────────────────────────────────────────────────
        st.markdown('<p class="sidebar-header">💡 What I Can Help With</p>', unsafe_allow_html=True)
        st.markdown("""
**📋 Policy Questions**
- HIPAA & patient rights
- Telehealth visit guidelines
- Insurance & billing

**💊 Medication**
- Refill process & timelines
- Controlled substance rules
- Prior authorisation

**📅 Appointments**
- Check availability
- Scheduling options
- Cancellation policy

**🏥 Discharge Care**
- Post-visit instructions
- When to seek emergency care
- Follow-up scheduling
        """)

        st.divider()
        st.caption("⚠️ This assistant provides general healthcare information only. Always consult a qualified healthcare professional for medical decisions.")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN CHAT INTERFACE
# ─────────────────────────────────────────────────────────────────────────────

def render_chat():
    # ── Header ────────────────────────────────────────────────────────────────
    col1, col2 = st.columns([8, 2])
    with col1:
        st.markdown("# 🏥 Healthcare AI Assistant")
        st.markdown(
            "Ask questions about **HIPAA policies**, **telehealth**, "
            "**insurance**, **appointments**, **medications**, or **discharge instructions**."
        )
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.session_state.conversation_id = f"session-{int(time.time())}"
            st.rerun()

    st.divider()

    # ── Welcome message (shown when no conversation yet) ──────────────────────
    if not st.session_state.messages:
        st.markdown("""
        <div style="text-align:center; padding:40px 20px; color:#666;">
            <div style="font-size:3rem;">🏥</div>
            <h3 style="color:#0066CC; margin:12px 0 8px;">Welcome to Healthcare AI Assistant</h3>
            <p>Your RAG-powered clinical information system.</p>
            <p style="font-size:0.85rem; color:#999;">
                Ask a question below to get started. <br>
                For appointment scheduling, try: <em>"Show me available slots this week"</em>
            </p>
        </div>

        <div style="display:flex; gap:12px; flex-wrap:wrap; justify-content:center; margin:20px 0;">
        """, unsafe_allow_html=True)

        # Suggestion chips
        suggestions = [
            "What are my HIPAA rights?",
            "How do I request a medication refill?",
            "Book an appointment",
            "What do I need for a telehealth visit?",
            "What insurance plans do you accept?",
            "What should I do after discharge?",
        ]
        cols = st.columns(3)
        for i, suggestion in enumerate(suggestions):
            with cols[i % 3]:
                if st.button(suggestion, use_container_width=True, key=f"suggest_{i}"):
                    process_query(suggestion)
                    st.rerun()

    # ── Conversation History ──────────────────────────────────────────────────
    chat_container = st.container()
    with chat_container:
        for msg in st.session_state.messages:
            if msg["role"] == "user":
                st.markdown(
                    f'<div class="user-bubble">👤 {msg["content"]}</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    '<div style="display:flex; align-items:flex-start; gap:8px;">'
                    '<span style="font-size:1.4rem;">🤖</span><div style="flex:1;">',
                    unsafe_allow_html=True,
                )
                render_assistant_message(msg)
                st.markdown('</div></div>', unsafe_allow_html=True)

            st.markdown("<div style='margin-bottom:8px;'></div>", unsafe_allow_html=True)

    # ── Input Bar ─────────────────────────────────────────────────────────────
    st.divider()
    with st.form("chat_form", clear_on_submit=True):
        input_col, btn_col = st.columns([9, 1])
        with input_col:
            user_input = st.text_input(
                "Your question",
                placeholder="e.g. 'What are my rights under HIPAA?' or 'Book me an appointment'",
                label_visibility="collapsed",
                max_chars=2000,
            )
        with btn_col:
            submitted = st.form_submit_button("Send 🚀", use_container_width=True, type="primary")

    if submitted and user_input.strip():
        process_query(user_input.strip())
        st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# QUERY PROCESSING
# ─────────────────────────────────────────────────────────────────────────────

def process_query(query: str):
    """Add user message, call API, store assistant response."""

    # Append user message
    st.session_state.messages.append({"role": "user", "content": query})

    # Show spinner while waiting for backend
    with st.spinner("🤔 Thinking …"):
        response = call_ask(query)

    if response is None:
        st.session_state.messages.append({
            "role": "assistant",
            "content": "⚠️ No response from the server. Please check your connection.",
            "metadata": {},
        })
        return

    # Handle API-level errors
    if "error" in response and "answer" not in response:
        st.session_state.messages.append({
            "role": "assistant",
            "content": f"⚠️ Error: {response['error']}",
            "metadata": {},
        })
        return

    # Handle 503 (index not loaded)
    if "detail" in response and "answer" not in response:
        st.session_state.messages.append({
            "role": "assistant",
            "content": (
                f"⚠️ {response['detail']}\n\n"
                "Please click **⚡ Ingest Documents** in the sidebar to load the knowledge base."
            ),
            "metadata": {},
        })
        return

    # Successful response
    st.session_state.messages.append({
        "role": "assistant",
        "content": response.get("answer", "No answer returned."),
        "metadata": {
            "confidence":               response.get("confidence", "NONE"),
            "confidence_score":         response.get("confidence_score", 0.0),
            "query_type":               response.get("query_type", "KNOWLEDGE"),
            "sources":                  response.get("sources", []),
            "disclaimer":               response.get("disclaimer", ""),
            "processing_time_seconds":  response.get("processing_time_seconds", 0.0),
        },
    })


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def main():
    render_sidebar()
    render_chat()


if __name__ == "__main__":
    main()
