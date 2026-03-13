# 🏥 MediAssist AI — NeoStats AI Engineer Case Study

> *An intelligent medical knowledge chatbot with RAG, live web search, and adaptive response modes.*

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.35+-red)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 🎯 Use Case

**Problem:** Patients and healthcare professionals often need quick, reliable answers to medical questions but face barriers — information overload, inability to search proprietary medical documents, and outdated AI knowledge.

**Solution:** MediAssist AI is a conversational medical knowledge assistant that:
- Searches your own uploaded medical PDFs/documents via **RAG**
- Pulls real-time medical news and research via **live web search**
- Adapts verbosity via **Concise / Detailed** response modes
- Supports **OpenAI, Groq, and Google Gemini** as LLM backends

---

## 🏗️ Project Structure

```
project/
├── config/
│   ├── __init__.py
│   └── config.py          ← All API keys & settings (loaded from env)
│
├── models/
│   ├── __init__.py
│   ├── llm.py             ← Multi-provider LLM abstraction (OpenAI/Groq/Gemini)
│   └── embeddings.py      ← Sentence-transformers embedding model
│
├── utils/
│   ├── __init__.py
│   ├── rag_utils.py       ← Document loading, chunking, vector store, retrieval
│   ├── search_utils.py    ← Live web search (Tavily + DuckDuckGo fallback)
│   └── prompt_utils.py    ← Prompt construction, chat history management
│
├── data/                  ← Vector store persistence (auto-created)
├── .streamlit/
│   ├── config.toml        ← Streamlit theme & server config
│   └── secrets.toml       ← Secrets template (DO NOT commit with real keys)
│
├── app.py                 ← Main Streamlit UI
├── requirements.txt
├── .env.example           ← Environment variable template
└── .gitignore
```

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 🤖 **Multi-LLM Support** | Switch between OpenAI GPT, Groq LLaMA, or Google Gemini |
| 📄 **RAG Integration** | Upload PDFs/TXT files; answers grounded in your documents |
| 🌐 **Live Web Search** | Real-time Tavily search with free DuckDuckGo fallback |
| ⚡ **Response Modes** | Toggle between Concise (≤3 sentences) and Detailed (comprehensive) |
| 🔍 **Auto Web Search** | AI auto-detects when fresh web data is needed |
| 📚 **Source Attribution** | See exactly which documents/web results informed each answer |
| 💬 **Chat History** | Full multi-turn conversation with context window management |

---

## 🚀 Getting Started

### 1. Clone the repo
```bash
git clone https://github.com/your-username/mediassist-ai.git
cd mediassist-ai
```

### 2. Create a virtual environment
```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Set up API keys
```bash
cp .env.example .env
# Edit .env and fill in your API keys
```

**Required:** At least one LLM provider key (Groq is free and fast).
**Optional:** Tavily key for better web search (DuckDuckGo works without it).

Get free keys:
- Groq: https://console.groq.com
- Tavily: https://tavily.com
- OpenAI: https://platform.openai.com
- Gemini: https://aistudio.google.com

### 5. Run the app
```bash
streamlit run app.py
```

---

## ☁️ Deploying to Streamlit Cloud

1. Push your code to GitHub (without `.env` — it's in `.gitignore`)
2. Go to [https://streamlit.io/cloud](https://streamlit.io/cloud)
3. Click **New app** → select your repo → set `app.py` as the main file
4. In **Advanced settings → Secrets**, add your keys:
```toml
GROQ_API_KEY = "gsk_..."
TAVILY_API_KEY = "tvly-..."
```
5. Click **Deploy** 🎉

---

## 🔒 Security

- API keys are **never hardcoded** — always loaded from environment variables or Streamlit secrets
- `.env` is in `.gitignore` to prevent accidental commits
- All functional code is wrapped in `try/except` blocks with proper logging

---

## ⚠️ Medical Disclaimer

MediAssist AI provides educational information only. It is **not** a substitute for professional medical advice, diagnosis, or treatment. Always consult a qualified healthcare provider.
