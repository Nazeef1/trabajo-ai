# Trabajo AI

Agentic resume-to-job-description matching with explainable gap analysis.

Upload a resume and one or more job descriptions. A pipeline of four LangGraph
agents — **Parser → Retriever → Reasoner → Gap Analyzer** — works through each
job description in sequence, ranks them by fit, and explains specifically
what's missing and why.

## Architecture

```
Resume + JD  ─▶  Parser Agent     (structures raw text into skills/experience/etc.)
                       │
                       ▼
             Retriever Agent      (RAG: embeds resume into ChromaDB, retrieves
                       │            chunks most relevant to this JD's requirements)
                       ▼
              Reasoner Agent      (scores the match using structured data +
                       │            retrieved evidence, not just raw text)
                       ▼
           Gap Analyzer Agent     (produces categorized, severity-ranked,
                       │            specific gaps)
                       ▼
                Ranked Results
```

Each agent has a distinct responsibility and its own prompt. The graph state
is passed between them via LangGraph, so each step's structured output feeds
the next — this is what makes it a genuine multi-agent pipeline rather than
one long prompt.

The RAG layer is doing real work here: rather than dumping the whole resume
into every prompt, the resume is chunked into semantic sections and the
Retriever Agent pulls only the chunks relevant to each JD's specific
requirements before the Reasoner agent scores the match.

## Stack

- **Orchestration:** LangGraph
- **LLM:** Groq (Llama 3.3 70B) — free tier
- **Vector store:** ChromaDB (in-memory, per-request)
- **Backend:** FastAPI
- **Frontend:** plain HTML/CSS/JS (no build step)

## Setup

### 1. Install dependencies

```bash
cd trabajo-ai
pip install -r requirements.txt
```

### 2. Get a free Groq API key

Go to [console.groq.com](https://console.groq.com), sign up (free), then
**API Keys → Create API Key**.

### 3. Configure environment

```bash
cp backend/.env.example backend/.env
```

Edit `backend/.env` and paste your key:

```
GROQ_API_KEY=gsk_your_actual_key_here
```

### 4. Run the server

```bash
python -m uvicorn backend.main:app --reload --port 8000
```

Open **http://localhost:8000** in your browser.

## Usage

1. Upload a resume (PDF).
2. Paste one or more job descriptions (use "+ Add another job description"
   for multiple).
3. Click **Run analysis**.
4. Results are ranked best-fit-first, each showing: match score, matched vs
   missing skills, and a categorized gap analysis with severity levels.

## Project structure

```
trabajo-ai/
├── backend/
│   ├── main.py         # FastAPI app, routes
│   ├── agents.py        # LangGraph pipeline: 4 agents
│   ├── parser.py        # PDF/text extraction
│   ├── vectorstore.py    # ChromaDB wrapper (RAG layer)
│   ├── models.py        # Pydantic schemas
│   └── .env.example
├── frontend/
│   └── index.html       # Single-page UI
├── requirements.txt
└── README.md
```

## Notes / known limitations

- Each JD is processed sequentially through the full agent pipeline (4 LLM
  calls per JD). For many JDs at once, this means noticeable latency —
  parallelizing across JDs would be a natural next step.
- Experience-year estimation from resume text is approximate; the Parser
  Agent infers it rather than relying on explicit date parsing.
- The vector store is ephemeral and scoped to a single request — nothing is
  persisted between analyses, by design (no need for it in this use case).
