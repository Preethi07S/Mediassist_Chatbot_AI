"""
app.py - MediAssist AI: Main Streamlit Application
A medical knowledge chatbot with RAG, live web search, and response modes.
"""

from dotenv import load_dotenv
load_dotenv()

import logging
import streamlit as st

# ─────────────────────────────────────────────
# LOGGING SETUP
# ─────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# PAGE CONFIG  (must be first Streamlit call)
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="MediAssist AI",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# IMPORTS
# ─────────────────────────────────────────────
from config.config import (
    APP_TITLE, APP_SUBTITLE, APP_ICON,
    DEFAULT_LLM_PROVIDER, TOP_K_RESULTS,
)
from models.llm import get_llm_response, get_available_providers
from models.embeddings import embed_texts  # invoked via rag pipeline
from utils.rag_utils import SimpleVectorStore, build_vector_store_from_file, retrieve_context
from utils.search_utils import web_search, should_search_web
from utils.prompt_utils import build_system_prompt, build_user_message, format_chat_history

# ─────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
    /* Global font & background */
    html, body, [class*="css"] {
        font-family: 'Inter', 'Segoe UI', sans-serif;
    }

    /* Main header */
    .main-header {
        background: linear-gradient(135deg, #1a6b4a 0%, #2d9e6b 50%, #3dbf82 100%);
        color: white;
        padding: 2rem 2.5rem;
        border-radius: 16px;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 20px rgba(26,107,74,0.3);
    }
    .main-header h1 { margin: 0; font-size: 2.2rem; font-weight: 700; }
    .main-header p  { margin: 0.3rem 0 0; font-size: 1rem; opacity: 0.9; }

    /* Chat messages */
    .stChatMessage { border-radius: 12px; margin-bottom: 0.5rem; }

    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #f0fdf4 0%, #dcfce7 100%);
    }
    [data-testid="stSidebar"] .block-container { padding-top: 1rem; }

    /* Status badges */
    .status-badge {
        display: inline-block;
        padding: 2px 10px;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
        margin: 2px;
    }
    .badge-rag   { background: #dcfce7; color: #166534; }
    .badge-web   { background: #dbeafe; color: #1e40af; }
    .badge-ai    { background: #f3e8ff; color: #6b21a8; }

    /* Info box */
    .info-box {
        background: #f0fdf4;
        border-left: 4px solid #16a34a;
        padding: 0.8rem 1rem;
        border-radius: 0 8px 8px 0;
        margin: 0.5rem 0;
        font-size: 0.9rem;
        color: #166534;
    }

    /* Upload area */
    .upload-section {
        background: white;
        border: 2px dashed #86efac;
        border-radius: 12px;
        padding: 1rem;
        margin: 0.5rem 0;
        text-align: center;
    }

    /* Disclaimer */
    .disclaimer {
        background: #fef9c3;
        border: 1px solid #fde047;
        border-radius: 8px;
        padding: 0.6rem 1rem;
        font-size: 0.8rem;
        color: #713f12;
        margin-top: 0.5rem;
    }

    /* Source expander */
    .source-box {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 0.6rem;
        font-size: 0.82rem;
        color: #475569;
        margin-top: 0.3rem;
    }

    /* Hide Streamlit branding */
    #MainMenu, footer, header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# SESSION STATE INIT
# ─────────────────────────────────────────────
def init_session_state():
    """Initialize all session state variables."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = SimpleVectorStore()
    if "uploaded_files" not in st.session_state:
        st.session_state.uploaded_files = []
    if "llm_provider" not in st.session_state:
        st.session_state.llm_provider = DEFAULT_LLM_PROVIDER
    if "response_mode" not in st.session_state:
        st.session_state.response_mode = "Detailed"
    if "use_web_search" not in st.session_state:
        st.session_state.use_web_search = False
    if "use_rag" not in st.session_state:
        st.session_state.use_rag = True
    if "auto_web_search" not in st.session_state:
        st.session_state.auto_web_search = False


init_session_state()


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Configuration")
    st.markdown("---")

    # LLM Provider
    st.markdown("### 🤖 AI Model")
    available_providers = get_available_providers()
    provider_labels = {
        "openai": "🟢 OpenAI GPT",
        "groq": "⚡ Groq (LLaMA)",
        "gemini": "💎 Google Gemini",
    }
    provider_options = available_providers
    provider_display = [provider_labels.get(p, p) for p in provider_options]

    selected_idx = 0
    if st.session_state.llm_provider in provider_options:
        selected_idx = provider_options.index(st.session_state.llm_provider)

    selected_display = st.selectbox(
        "Select Provider",
        provider_display,
        index=selected_idx,
        help="Choose which AI model powers your chat.",
    )
    st.session_state.llm_provider = provider_options[provider_display.index(selected_display)]

    st.markdown("---")

    # Response Mode
    st.markdown("### 📝 Response Mode")
    st.session_state.response_mode = st.radio(
        "Mode",
        ["Concise", "Detailed"],
        index=["Concise", "Detailed"].index(st.session_state.response_mode),
        help="**Concise**: Short, direct answers\n**Detailed**: Comprehensive explanations",
    )
    mode_desc = {
        "Concise": "⚡ Short, to-the-point answers",
        "Detailed": "📚 In-depth explanations with context",
    }
    st.caption(mode_desc[st.session_state.response_mode])

    st.markdown("---")

    # Features
    st.markdown("### 🔧 Features")
    st.session_state.use_rag = st.toggle(
        "📄 Document Knowledge (RAG)",
        value=st.session_state.use_rag,
        help="Use uploaded documents to answer questions.",
    )
    st.session_state.use_web_search = st.toggle(
        "🌐 Live Web Search",
        value=st.session_state.use_web_search,
        help="Search the web for real-time information.",
    )
    if st.session_state.use_web_search:
        st.session_state.auto_web_search = st.checkbox(
            "Auto-detect when to search",
            value=st.session_state.auto_web_search,
            help="Let AI decide when web search is needed.",
        )

    st.markdown("---")

    # Document Upload
    st.markdown("### 📁 Upload Documents")
    st.caption("Upload medical PDFs, guidelines, or research papers.")

    uploaded_files = st.file_uploader(
        "Choose files",
        type=["pdf", "txt", "md"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    if uploaded_files:
        new_files = [f for f in uploaded_files if f.name not in st.session_state.uploaded_files]
        if new_files:
            with st.spinner("🔄 Processing documents..."):
                for file in new_files:
                    try:
                        n_chunks = build_vector_store_from_file(file, st.session_state.vector_store)
                        st.session_state.uploaded_files.append(file.name)
                        st.success(f"✅ {file.name} ({n_chunks} chunks indexed)")
                        logger.info(f"Indexed: {file.name} → {n_chunks} chunks")
                    except Exception as e:
                        st.error(f"❌ Failed to process {file.name}: {str(e)}")
                        logger.error(f"File processing error: {e}")

    if st.session_state.uploaded_files:
        st.markdown("**Indexed documents:**")
        for fname in st.session_state.uploaded_files:
            st.markdown(f"• 📄 {fname}")
        st.caption(f"Total chunks: {st.session_state.vector_store.count()}")

        if st.button("🗑️ Clear Documents", use_container_width=True):
            st.session_state.vector_store = SimpleVectorStore()
            st.session_state.uploaded_files = []
            st.rerun()

    st.markdown("---")

    # Chat controls
    st.markdown("### 💬 Chat")
    if st.button("🔄 Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.markdown("---")
    st.markdown(
        '<div style="font-size:0.75rem;color:#6b7280;text-align:center;">'
        'MediAssist AI v1.0<br>Built with ❤️ for NeoStats<br>'
        '⚠️ For educational use only'
        '</div>',
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────
# MAIN CONTENT
# ─────────────────────────────────────────────

# Header
st.markdown(f"""
<div class="main-header">
    <h1>{APP_ICON} {APP_TITLE}</h1>
    <p>{APP_SUBTITLE} — Powered by RAG + Live Web Search</p>
</div>
""", unsafe_allow_html=True)

# Active feature badges
col1, col2, col3 = st.columns([1, 1, 1])
with col1:
    rag_status = "✅ RAG Active" if (st.session_state.use_rag and not st.session_state.vector_store.is_empty()) else "⭕ RAG (no docs)" if st.session_state.use_rag else "❌ RAG Off"
    st.markdown(f'<span class="status-badge badge-rag">{rag_status}</span>', unsafe_allow_html=True)
with col2:
    web_status = "✅ Web Search On" if st.session_state.use_web_search else "❌ Web Search Off"
    st.markdown(f'<span class="status-badge badge-web">{web_status}</span>', unsafe_allow_html=True)
with col3:
    st.markdown(f'<span class="status-badge badge-ai">🤖 {st.session_state.llm_provider.upper()} · {st.session_state.response_mode}</span>', unsafe_allow_html=True)

st.markdown("")

# Disclaimer
st.markdown(
    '<div class="disclaimer">⚕️ <strong>Medical Disclaimer:</strong> '
    'MediAssist AI provides educational information only and is not a substitute '
    'for professional medical advice, diagnosis, or treatment. Always consult a qualified healthcare provider.</div>',
    unsafe_allow_html=True,
)

st.markdown("---")

# Welcome message (shown only when no chat history)
if not st.session_state.messages:
    with st.chat_message("assistant", avatar="🏥"):
        st.markdown("""
**Welcome to MediAssist AI!** 👋

I'm your intelligent medical knowledge companion. Here's what I can help you with:

- 🔬 **Medical concepts** — symptoms, diseases, treatments, medications
- 📄 **Document analysis** — upload medical PDFs and ask questions about them
- 🌐 **Latest research** — get real-time information from the web
- 💊 **Drug information** — understand medications and interactions
- 🏥 **Healthcare guidance** — general wellness and preventive care

**Quick start tips:**
- Upload medical documents in the sidebar to enable RAG
- Toggle "Live Web Search" for current medical news
- Switch between **Concise** and **Detailed** response modes

*What medical topic can I help you understand today?*
        """)

# ─────────────────────────────────────────────
# RENDER CHAT HISTORY
# ─────────────────────────────────────────────
for message in st.session_state.messages:
    role = message["role"]
    avatar = "🏥" if role == "assistant" else "👤"
    with st.chat_message(role, avatar=avatar):
        st.markdown(message["content"])
        # Show sources if available
        if role == "assistant" and message.get("sources"):
            with st.expander("📚 Sources used", expanded=False):
                st.markdown(
                    f'<div class="source-box">{message["sources"]}</div>',
                    unsafe_allow_html=True,
                )


# ─────────────────────────────────────────────
# CHAT INPUT & RESPONSE GENERATION
# ─────────────────────────────────────────────
if user_input := st.chat_input("Ask a medical question..."):

    # Display user message
    with st.chat_message("user", avatar="👤"):
        st.markdown(user_input)

    st.session_state.messages.append({"role": "user", "content": user_input})

    # Generate response
    with st.chat_message("assistant", avatar="🏥"):
        rag_context = ""
        web_results = ""
        sources_used = []

        with st.status("🔍 Thinking...", expanded=False) as status:

            # ── Step 1: RAG retrieval ──
            if st.session_state.use_rag and not st.session_state.vector_store.is_empty():
                try:
                    status.update(label="📄 Searching documents...")
                    rag_context = retrieve_context(
                        user_input,
                        st.session_state.vector_store,
                        top_k=TOP_K_RESULTS,
                    )
                    if rag_context:
                        sources_used.append("📄 Document Knowledge Base")
                        logger.info("RAG context retrieved successfully.")
                except Exception as e:
                    logger.error(f"RAG retrieval error: {e}")
                    st.warning("⚠️ Document search encountered an issue.")

            # ── Step 2: Web search ──
            perform_search = False
            if st.session_state.use_web_search:
                if st.session_state.auto_web_search:
                    perform_search = should_search_web(user_input, rag_context)
                else:
                    perform_search = True

            if perform_search:
                try:
                    status.update(label="🌐 Searching the web...")
                    web_results = web_search(user_input, max_results=4)
                    if web_results and "unavailable" not in web_results.lower():
                        sources_used.append("🌐 Live Web Search")
                        logger.info("Web search completed successfully.")
                except Exception as e:
                    logger.error(f"Web search error: {e}")
                    st.warning("⚠️ Web search encountered an issue.")

            # ── Step 3: Build prompt ──
            status.update(label="🤖 Generating response...")
            system_prompt = build_system_prompt(st.session_state.response_mode)
            enriched_user_msg = build_user_message(user_input, rag_context, web_results)

            # Build message history for LLM
            history = format_chat_history(
                [m for m in st.session_state.messages[:-1]],  # exclude current user msg
                max_turns=8,
            )
            history.append({"role": "user", "content": enriched_user_msg})

            status.update(label="✅ Done!", state="complete")

        # ── Step 4: Call LLM ──
        try:
            response_text = get_llm_response(
                messages=history,
                provider=st.session_state.llm_provider,
                system_prompt=system_prompt,
                temperature=0.4 if st.session_state.response_mode == "Concise" else 0.7,
                max_tokens=512 if st.session_state.response_mode == "Concise" else 1536,
            )
        except Exception as e:
            logger.error(f"LLM response error: {e}")
            response_text = (
                f"⚠️ I encountered an error generating a response.\n\n"
                f"**Error:** `{str(e)}`\n\n"
                f"Please check that your API key for **{st.session_state.llm_provider}** "
                f"is set correctly in your environment variables."
            )

        # Display response
        st.markdown(response_text)

        # Show sources
        sources_str = ""
        if sources_used:
            sources_str = "**Sources used in this response:**\n" + "\n".join(f"- {s}" for s in sources_used)
            if web_results:
                sources_str +=f"\\n\\n{web_results}"
            if rag_context:
                sources_str += f"\n\n**Retrieved context preview:**\n```\n{rag_context[:400]}...\n```"
            with st.expander("📚 Sources used", expanded=False):
                st.markdown(f'<div class="source-box">{sources_str}</div>', unsafe_allow_html=True)

        # Persist to session
        st.session_state.messages.append({
            "role": "assistant",
            "content": response_text,
            "sources": sources_str,
        })
