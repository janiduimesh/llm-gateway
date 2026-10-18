import streamlit as st
import requests
import json
import time
from datetime import datetime

st.set_page_config(
    page_title="LLM Gateway Chat",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* ── Page background ── */
.stApp {
    background: linear-gradient(135deg, #0f0f1a 0%, #1a1a2e 50%, #16213e 100%);
    min-height: 100vh;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: rgba(255,255,255,0.03);
    border-right: 1px solid rgba(255,255,255,0.08);
    backdrop-filter: blur(20px);
}

[data-testid="stSidebar"] .stTextInput input,
[data-testid="stSidebar"] .stSelectbox select,
[data-testid="stSidebar"] .stSlider {
    background: rgba(255,255,255,0.05) !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    color: #e2e8f0 !important;
    border-radius: 8px !important;
}

/* ── Chat container ── */
.chat-container {
    max-width: 860px;
    margin: 0 auto;
    padding: 0 1rem;
}

/* ── Header ── */
.gateway-header {
    text-align: center;
    padding: 2.5rem 0 1.5rem;
}

.gateway-header h1 {
    font-size: 2.2rem;
    font-weight: 700;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.3rem;
}

.gateway-header p {
    color: rgba(255,255,255,0.45);
    font-size: 0.9rem;
    font-weight: 400;
}

/* ── Message bubbles ── */
.msg-row {
    display: flex;
    margin-bottom: 1.2rem;
    animation: fadeSlideIn 0.25s ease;
}

@keyframes fadeSlideIn {
    from { opacity: 0; transform: translateY(8px); }
    to   { opacity: 1; transform: translateY(0); }
}

.msg-row.user { justify-content: flex-end; }
.msg-row.assistant { justify-content: flex-start; }

.msg-bubble {
    max-width: 72%;
    padding: 0.85rem 1.1rem;
    border-radius: 16px;
    font-size: 0.93rem;
    line-height: 1.65;
    word-wrap: break-word;
    white-space: pre-wrap;
}

.msg-bubble.user {
    background: linear-gradient(135deg, #667eea, #764ba2);
    color: #fff;
    border-bottom-right-radius: 4px;
    box-shadow: 0 4px 20px rgba(102,126,234,0.35);
}

.msg-bubble.assistant {
    background: rgba(255,255,255,0.06);
    color: #e2e8f0;
    border: 1px solid rgba(255,255,255,0.1);
    border-bottom-left-radius: 4px;
    backdrop-filter: blur(10px);
}

.msg-avatar {
    width: 32px;
    height: 32px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 0.85rem;
    flex-shrink: 0;
    margin-top: 2px;
}

.msg-avatar.user {
    background: linear-gradient(135deg, #667eea, #764ba2);
    margin-left: 10px;
    order: 2;
}

.msg-avatar.assistant {
    background: rgba(255,255,255,0.08);
    margin-right: 10px;
    border: 1px solid rgba(255,255,255,0.12);
}

/* ── Meta badge ── */
.meta-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    font-size: 0.7rem;
    color: rgba(255,255,255,0.35);
    margin-top: 5px;
    padding: 2px 8px;
    border-radius: 20px;
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.07);
}

.meta-badge.cache-hit { color: #68d391; border-color: rgba(104,211,145,0.2); background: rgba(104,211,145,0.06); }
.meta-badge.cache-miss { color: #63b3ed; border-color: rgba(99,179,237,0.2); background: rgba(99,179,237,0.06); }

/* ── Status pills ── */
.pill {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 20px;
    font-size: 0.72rem;
    font-weight: 500;
}

.pill-green  { background: rgba(104,211,145,0.15); color: #68d391; border: 1px solid rgba(104,211,145,0.25); }
.pill-red    { background: rgba(252,129,129,0.15); color: #fc8181; border: 1px solid rgba(252,129,129,0.25); }
.pill-yellow { background: rgba(246,224,94,0.15);  color: #f6e05e; border: 1px solid rgba(246,224,94,0.25); }
.pill-blue   { background: rgba(99,179,237,0.15);  color: #63b3ed; border: 1px solid rgba(99,179,237,0.25); }

/* ── Stats card ── */
.stat-card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 12px;
    padding: 0.9rem 1rem;
    text-align: center;
    backdrop-filter: blur(10px);
}

.stat-card .stat-val {
    font-size: 1.6rem;
    font-weight: 700;
    background: linear-gradient(135deg, #667eea, #f093fb);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.stat-card .stat-lbl {
    font-size: 0.7rem;
    color: rgba(255,255,255,0.4);
    margin-top: 2px;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

/* ── Divider ── */
.chat-divider {
    border: none;
    border-top: 1px solid rgba(255,255,255,0.06);
    margin: 1.5rem 0;
}

/* ── Input area overrides ── */
.stTextArea textarea {
    background: rgba(255,255,255,0.05) !important;
    border: 1px solid rgba(255,255,255,0.12) !important;
    border-radius: 12px !important;
    color: #e2e8f0 !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.93rem !important;
    resize: none !important;
}

.stTextArea textarea:focus {
    border-color: rgba(102,126,234,0.6) !important;
    box-shadow: 0 0 0 3px rgba(102,126,234,0.15) !important;
}

/* ── Buttons ── */
.stButton button {
    border-radius: 10px !important;
    font-weight: 500 !important;
    font-family: 'Inter', sans-serif !important;
    transition: all 0.2s ease !important;
}

.stButton button[kind="primary"] {
    background: linear-gradient(135deg, #667eea, #764ba2) !important;
    border: none !important;
    box-shadow: 0 4px 15px rgba(102,126,234,0.4) !important;
}

.stButton button[kind="primary"]:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 20px rgba(102,126,234,0.55) !important;
}

/* ── Scrollable chat area ── */
.chat-scroll {
    max-height: 62vh;
    overflow-y: auto;
    padding-right: 4px;
    scroll-behavior: smooth;
}

.chat-scroll::-webkit-scrollbar { width: 4px; }
.chat-scroll::-webkit-scrollbar-track { background: transparent; }
.chat-scroll::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.12); border-radius: 4px; }

/* ── Sidebar labels ── */
[data-testid="stSidebar"] label, [data-testid="stSidebar"] .stMarkdown p {
    color: rgba(255,255,255,0.7) !important;
    font-size: 0.82rem !important;
}

[data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
    color: #e2e8f0 !important;
}

.sidebar-section {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 10px;
    padding: 0.8rem;
    margin-bottom: 0.8rem;
}

/* ── Error / warning boxes ── */
.stAlert {
    border-radius: 10px !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
}

/* ── Typing indicator ── */
.typing-dot {
    display: inline-block;
    width: 7px; height: 7px;
    border-radius: 50%;
    background: rgba(255,255,255,0.5);
    margin: 0 2px;
    animation: bounce 1.2s infinite;
}
.typing-dot:nth-child(2) { animation-delay: 0.2s; }
.typing-dot:nth-child(3) { animation-delay: 0.4s; }

@keyframes bounce {
    0%, 80%, 100% { transform: scale(0.6); opacity: 0.5; }
    40%            { transform: scale(1.0); opacity: 1.0; }
}
</style>
""", unsafe_allow_html=True)


# ── Session state initialisation ──────────────────────────────────────────────

if "messages" not in st.session_state:
    st.session_state.messages = []

if "session_stats" not in st.session_state:
    st.session_state.session_stats = {
        "total_requests": 0,
        "cache_hits": 0,
        "total_latency_ms": 0,
        "providers_used": [],
    }


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## ⚙️ Gateway Config")

    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    gateway_url = st.text_input(
        "Gateway URL",
        value="http://localhost:8000",
        help="Base URL of your LLM Gateway instance",
    )
    api_key = st.text_input(
        "API Key",
        value="",
        type="password",
        help="Your team's gateway Bearer API key",
    )
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.markdown("**Model**")
    model_options = [
        "llama3-8b-8192",
        "llama3-70b-8192",
        "mixtral-8x7b-32768",
        "gemma-7b-it",
        "gpt-4o-mini",
        "gpt-4o",
        "custom…",
    ]
    model_choice = st.selectbox("Model", model_options, label_visibility="collapsed")
    if model_choice == "custom…":
        model_choice = st.text_input("Custom model name", placeholder="e.g. llama3-8b-8192")

    system_prompt = st.text_area(
        "System Prompt",
        value="You are a helpful, precise, and concise assistant.",
        height=80,
    )
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.markdown("**Parameters**")
    temperature = st.slider("Temperature", 0.0, 2.0, 0.7, 0.05)
    max_tokens = st.slider("Max Tokens", 64, 8192, 1024, 64)
    streaming = st.toggle("Streaming", value=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.divider()

    # ── Session stats ──
    stats = st.session_state.session_stats
    total = stats["total_requests"]
    hits = stats["cache_hits"]
    avg_lat = round(stats["total_latency_ms"] / total, 0) if total else 0
    hit_rate = round(hits / total * 100, 0) if total else 0

    st.markdown("**📊 Session Stats**")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-val">{total}</div>
            <div class="stat-lbl">Requests</div>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-val">{int(hit_rate)}%</div>
            <div class="stat-lbl">Cache Hit</div>
        </div>""", unsafe_allow_html=True)

    col3, col4 = st.columns(2)
    with col3:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-val">{int(avg_lat)}</div>
            <div class="stat-lbl">Avg ms</div>
        </div>""", unsafe_allow_html=True)
    with col4:
        providers = list(set(stats["providers_used"])) or ["—"]
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-val" style="font-size:0.85rem">{providers[-1]}</div>
            <div class="stat-lbl">Last Provider</div>
        </div>""", unsafe_allow_html=True)

    st.divider()

    col_clr, col_health = st.columns(2)
    with col_clr:
        if st.button("🗑️ Clear", use_container_width=True):
            st.session_state.messages = []
            st.session_state.session_stats = {
                "total_requests": 0,
                "cache_hits": 0,
                "total_latency_ms": 0,
                "providers_used": [],
            }
            st.rerun()

    with col_health:
        if st.button("❤️ Health", use_container_width=True):
            try:
                r = requests.get(f"{gateway_url}/health", timeout=3)
                if r.status_code == 200:
                    st.success("Online")
                else:
                    st.error(f"{r.status_code}")
            except Exception:
                st.error("Unreachable")


# ── Main content ──────────────────────────────────────────────────────────────

st.markdown("""
<div class="gateway-header">
    <h1>🚀 LLM Gateway Chat</h1>
    <p>Powered by Groq · OpenAI · Redis Semantic Cache · Circuit Breaker</p>
</div>
""", unsafe_allow_html=True)


def render_messages():
    for msg in st.session_state.messages:
        role = msg["role"]
        content = msg["content"]
        meta = msg.get("meta", {})

        if role == "user":
            st.markdown(f"""
            <div class="msg-row user">
                <div class="msg-bubble user">{content}</div>
                <div class="msg-avatar user">👤</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            cache_hit = meta.get("cache_hit", False)
            latency = meta.get("latency_ms", "")
            provider = meta.get("provider", "")

            badge_class = "cache-hit" if cache_hit else "cache-miss"
            badge_icon = "⚡" if cache_hit else "🔄"
            badge_text = "Cache Hit" if cache_hit else provider or "LLM"
            lat_text = f" · {latency}ms" if latency else ""

            st.markdown(f"""
            <div class="msg-row assistant">
                <div class="msg-avatar assistant">🤖</div>
                <div>
                    <div class="msg-bubble assistant">{content}</div>
                    <div style="margin-left:4px;margin-top:4px;">
                        <span class="meta-badge {badge_class}">{badge_icon} {badge_text}{lat_text}</span>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)


render_messages()

st.markdown('<hr class="chat-divider">', unsafe_allow_html=True)


# ── Input area ────────────────────────────────────────────────────────────────

with st.form("chat_form", clear_on_submit=True):
    col_input, col_send = st.columns([9, 1])
    with col_input:
        user_input = st.text_area(
            "Message",
            placeholder="Ask anything…  (Shift+Enter for newline)",
            height=68,
            label_visibility="collapsed",
        )
    with col_send:
        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
        submitted = st.form_submit_button("Send", use_container_width=True, type="primary")


# ── Send & receive ────────────────────────────────────────────────────────────

def build_payload(user_text: str) -> dict:
    messages = []
    if system_prompt.strip():
        messages.append({"role": "system", "content": system_prompt.strip()})
    for m in st.session_state.messages:
        if m["role"] in ("user", "assistant"):
            messages.append({"role": m["role"], "content": m["content"]})
    messages.append({"role": "user", "content": user_text})
    return {
        "model": model_choice or None,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": streaming,
    }


def call_gateway_stream(payload: dict, headers: dict) -> tuple[str, dict]:
    url = f"{gateway_url.rstrip('/')}/v1/chat/completions"
    full_text = ""
    meta = {}

    with requests.post(url, json=payload, headers=headers, stream=True, timeout=60) as resp:
        resp.raise_for_status()
        for raw_line in resp.iter_lines():
            if not raw_line:
                continue
            line = raw_line.decode("utf-8") if isinstance(raw_line, bytes) else raw_line
            if not line.startswith("data:"):
                continue
            data_str = line[5:].strip()
            if data_str == "[DONE]":
                break
            try:
                chunk = json.loads(data_str)
                if "error" in chunk:
                    raise RuntimeError(chunk["error"].get("message", "Unknown stream error"))
                delta = chunk.get("choices", [{}])[0].get("delta", {})
                if "content" in delta:
                    full_text += delta["content"]
            except json.JSONDecodeError:
                continue

    return full_text, meta


def call_gateway_sync(payload: dict, headers: dict) -> tuple[str, dict]:
    url = f"{gateway_url.rstrip('/')}/v1/chat/completions"
    t0 = time.perf_counter()
    resp = requests.post(url, json=payload, headers=headers, timeout=60)
    elapsed = round((time.perf_counter() - t0) * 1000, 1)
    resp.raise_for_status()
    data = resp.json()

    content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
    gw = data.get("_gateway", {})
    gw_meta = data.get("_gateway_meta", {})

    meta = {
        "cache_hit": gw.get("cache_hit", False),
        "latency_ms": gw.get("latency_ms") or elapsed,
        "provider": gw_meta.get("provider", ""),
        "request_id": gw.get("request_id", ""),
    }
    return content, meta


if submitted and user_input.strip():
    if not api_key.strip():
        st.warning("⚠️  Enter your Gateway API key in the sidebar to continue.", icon="🔑")
        st.stop()

    user_text = user_input.strip()
    st.session_state.messages.append({"role": "user", "content": user_text})

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = build_payload(user_text)

    t0 = time.perf_counter()

    if streaming:
        with st.spinner(""):
            placeholder = st.empty()
            placeholder.markdown("""
            <div class="msg-row assistant">
                <div class="msg-avatar assistant">🤖</div>
                <div class="msg-bubble assistant">
                    <span class="typing-dot"></span>
                    <span class="typing-dot"></span>
                    <span class="typing-dot"></span>
                </div>
            </div>
            """, unsafe_allow_html=True)

            try:
                reply, meta = call_gateway_stream(payload, headers)
                elapsed = round((time.perf_counter() - t0) * 1000, 1)
                if not meta:
                    meta = {"latency_ms": elapsed, "cache_hit": False, "provider": "groq"}
            except Exception as exc:
                reply = f"❌ **Error:** {exc}"
                meta = {}

        placeholder.empty()
    else:
        with st.spinner("Thinking…"):
            try:
                reply, meta = call_gateway_sync(payload, headers)
            except Exception as exc:
                reply = f"❌ **Error:** {exc}"
                meta = {}

    st.session_state.messages.append({"role": "assistant", "content": reply, "meta": meta})

    stats = st.session_state.session_stats
    stats["total_requests"] += 1
    if meta.get("cache_hit"):
        stats["cache_hits"] += 1
    if meta.get("latency_ms"):
        stats["total_latency_ms"] += meta["latency_ms"]
    if meta.get("provider"):
        stats["providers_used"].append(meta["provider"])

    st.rerun()
