# GenAI Assistant with RAG

A production-grade AI chat assistant powered by **Retrieval-Augmented Generation (RAG)** using Google Gemini. The system ingests documents, generates embeddings, performs vector similarity search, and uses retrieved context to generate grounded, accurate responses.

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115-green?logo=fastapi)
![Gemini](https://img.shields.io/badge/Google-Gemini-orange?logo=google)
![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector_Store-purple)

---

## 📋 Table of Contents

- [Architecture](#architecture)
- [RAG Workflow](#rag-workflow)
- [Embedding Strategy](#embedding-strategy)
- [Similarity Search](#similarity-search)
- [Prompt Design](#prompt-design)
- [Tech Stack](#tech-stack)
- [Setup Instructions](#setup-instructions)
- [API Documentation](#api-documentation)
- [Project Structure](#project-structure)
- [Deployment](#deployment)

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────┐
│                   Frontend (HTML/CSS/JS)             │
│         Chat UI → Session Management → API Calls    │
└───────────────────────┬─────────────────────────────┘
                        │ POST /api/chat
                        ▼
┌─────────────────────────────────────────────────────┐
│                 FastAPI Backend                      │
│  ┌──────────┐  ┌──────────┐  ┌──────────────────┐  │
│  │ Validator │→│ RAG Svc  │→│ Conversation Mem  │  │
│  └──────────┘  └────┬─────┘  └──────────────────┘  │
│                     │                                │
│         ┌───────────┼───────────┐                   │
│         ▼           ▼           ▼                   │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐           │
│  │ Embedding│ │ ChromaDB │ │ Gemini   │           │
│  │ Service  │ │ Vector   │ │ LLM      │           │
│  └──────────┘ │ Store    │ └──────────┘           │
│               └──────────┘                          │
└─────────────────────────────────────────────────────┘
```

---

## 🔄 RAG Workflow

### Indexing Phase (Application Startup)

1. **Load Documents** — Read `docs.json` knowledge base
2. **Chunk Documents** — Split into 300-500 token chunks with 50-token overlap
3. **Generate Embeddings** — Google `embedding-001` model (768 dimensions)
4. **Store Vectors** — Persist in ChromaDB with metadata

### Query Phase (Runtime)

1. **Receive Query** — User sends message via chat
2. **Embed Query** — Generate query embedding using `retrieval_query` task type
3. **Similarity Search** — Find Top-3 most relevant chunks via cosine similarity
4. **Threshold Check** — If max similarity < 0.35, return fallback response
5. **Build Prompt** — Inject retrieved context + conversation history + question
6. **Generate Response** — Send to Gemini (temperature=0.2) for grounded answer
7. **Store History** — Save message pair in conversation memory

---

## 📐 Embedding Strategy

| Parameter | Value | Rationale |
|---|---|---|
| **Model** | `models/embedding-001` | Google's optimized text embedding model |
| **Dimensions** | 768 | Rich semantic representation |
| **Document Task** | `retrieval_document` | Optimized for document indexing |
| **Query Task** | `retrieval_query` | Optimized for search queries |
| **Chunk Size** | 400 tokens | Balances context richness vs. specificity |
| **Chunk Overlap** | 50 tokens | Preserves cross-boundary context |

### Chunking Strategy

- **Sentence-aware splitting** — Never breaks mid-sentence
- **Token estimation** — ~1 token per 0.75 words (English heuristic)
- **Overlap** — Last few sentences from previous chunk are prepended to maintain context continuity across chunk boundaries

---

## 🔍 Similarity Search

### Cosine Similarity

Cosine similarity measures the angle between two vectors in high-dimensional space:

```
similarity = cos(θ) = (A · B) / (||A|| × ||B||)
```

- **Range**: 0 (orthogonal/unrelated) to 1 (identical direction)
- **Advantage**: Magnitude-independent — focuses on semantic direction, not document length
- **ChromaDB**: Internally uses cosine distance (1 - similarity), which we convert back

### Threshold Logic

- **Threshold**: 0.35 (configurable via `.env`)
- **Above threshold**: Include chunk in context for LLM
- **Below threshold**: Exclude chunk; if all chunks below threshold, return fallback response
- **Purpose**: Prevents hallucination by ensuring only semantically relevant content reaches the LLM

---

## 💬 Prompt Design

### Strategy

```
[System Prompt]      → Constrains LLM to use ONLY provided context
[Retrieved Context]  → Numbered sources with relevance scores
[Conversation History] → Last 5 message pairs for continuity
[User Question]      → Current query
[Instructions]       → Explicit grounding instruction
```

### Design Reasoning

1. **System Prompt**: Strict rules prevent the LLM from generating information outside the knowledge base
2. **Numbered Sources**: Enables traceability — the LLM knows which source each fact comes from
3. **Relevance Scores**: Helps the LLM prioritize higher-quality matches
4. **History Limit**: 5 pairs balances context continuity vs. token budget
5. **Low Temperature (0.2)**: Ensures factual, deterministic responses — minimal creativity
6. **Explicit Grounding**: Final instruction reinforces "say so if you don't know"

---

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| Backend | FastAPI (Python 3.10+) |
| LLM | Google Gemini (`gemini-2.0-flash`) |
| Embeddings | Google `embedding-001` (768D) |
| Vector Store | ChromaDB (persistent, cosine similarity) |
| Frontend | Vanilla HTML/CSS/JavaScript |
| Auth (Bonus) | JWT with python-jose |

---

## 🚀 Setup Instructions

### Prerequisites

- Python 3.10+
- Google Gemini API key ([Get one here](https://aistudio.google.com/apikey))

### Installation

```bash
# 1. Clone the repository
git clone <repository-url>
cd project

# 2. Create virtual environment
python -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment
cp .env.example .env
# Edit .env and add your GEMINI_API_KEY

# 5. Run the application
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### Access

- **Chat UI**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health

---

## 📡 API Documentation

### POST /api/chat

Send a message and receive a RAG-powered response.

**Request:**
```json
{
  "sessionId": "abc123-uuid",
  "message": "How can I reset my password?"
}
```

**Response:**
```json
{
  "reply": "Users can reset their password from Settings > Security > Reset Password...",
  "tokensUsed": 245,
  "retrievedChunks": 3
}
```

### GET /health

Check system status.

**Response:**
```json
{
  "status": "healthy",
  "totalChunks": 12,
  "timestamp": "2024-01-01T00:00:00"
}
```

---

## 📁 Project Structure

```
project/
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI app, startup, CORS
│   ├── config.py             # Pydantic settings from .env
│   ├── routes/
│   │   ├── chat.py           # POST /api/chat
│   │   └── health.py         # GET /health
│   ├── services/
│   │   ├── embedding.py      # Google embedding generation
│   │   ├── llm.py            # Gemini LLM integration
│   │   ├── rag.py            # RAG orchestrator
│   │   └── conversation.py   # Conversation memory
│   ├── models/
│   │   └── schemas.py        # Pydantic models
│   ├── vectorstore/
│   │   └── chroma_store.py   # ChromaDB wrapper
│   ├── prompts/
│   │   └── templates.py      # Prompt templates
│   └── utils/
│       ├── chunker.py        # Document chunking
│       └── logger.py         # Structured logging
├── frontend/
│   ├── index.html            # Chat UI
│   ├── styles.css            # Styling
│   └── app.js                # Client logic
├── docs.json                 # Knowledge base
├── requirements.txt          # Dependencies
├── .env.example              # Environment template
└── README.md                 # This file
```

---

## 📊 Error Handling

| Scenario | Response |
|---|---|
| Missing message field | `400: {"error": "Message field is required"}` |
| Invalid API key | `400: {"detail": "Invalid Gemini API key"}` |
| LLM timeout | `504: {"detail": "Request timed out"}` |
| Rate limit | Automatic retry with exponential backoff |
| No relevant context | Fallback: "I could not find enough information..." |
| Server error | `500: {"error": "An unexpected error occurred"}` |

---

## 📄 License

MIT License
