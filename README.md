# Trabajo AI

Agentic resume-to-job-description matching with explainable gap analysis.

Upload a resume and one or more job descriptions. Four LangGraph agents —
parser, retriever, reasoner, and gap analyzer — run in sequence to rank job
fit and explain what's missing.

**Live demo:** [trabajo-ai.vercel.app](https://trabajo-ai.vercel.app)

## Architecture

```
Resume + JD → Parser Agent → Retriever Agent (RAG) → Reasoner Agent → Gap Analyzer
```

- **Parser** structures raw resume/JD text into skills, experience, and qualifications.
- **Retriever** embeds the resume into ChromaDB and retrieves the chunks most relevant to each JD's specific requirements.
- **Reasoner** scores the match using structured data plus retrieved evidence.
- **Gap Analyzer** produces categorized, severity-ranked, specific gaps.

## Stack

LangGraph · Groq (Llama 3.3 70B) · ChromaDB · FastAPI · vanilla JS frontend

## Project structure

```
trabajo-ai/
├── backend/        # FastAPI app, agent pipeline, RAG layer
├── frontend/        # Single-page UI
└── render.yaml      # Backend deployment config
```

## Known limitations

- JDs are processed sequentially (4 LLM calls each), so multiple JDs add up in latency.
- Experience-years is an LLM estimate, not parsed from explicit dates.
- Backend free tier sleeps after inactivity; first request after idle can take ~30-50s.
