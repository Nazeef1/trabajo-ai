"""
Trabajo AI FastAPI backend.

Endpoints:
  POST /api/analyze  -> upload one resume (PDF or text) + multiple JDs (text),
                         run the agent pipeline for each JD, return ranked results.
  GET  /health        -> basic liveness check
"""
import os
from typing import List

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from dotenv import load_dotenv
from groq import RateLimitError, APIStatusError, APIConnectionError

from .parser import extract_text_from_upload
from .vectorstore import ResumeVectorStore
from .agents import run_pipeline_for_jd
from .models import AnalysisResponse

load_dotenv()

app = FastAPI(title="Trabajo AI API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    groq_configured = bool(os.environ.get("GROQ_API_KEY"))
    return {"status": "ok", "groq_configured": groq_configured}


@app.post("/api/analyze", response_model=AnalysisResponse)
async def analyze(
    resume_file: UploadFile = File(...),
    jd_texts: List[str] = Form(...),
):
    """
    resume_file: PDF or text file upload
    jd_texts: list of raw JD text blocks (one per job description)
    """
    if not jd_texts or all(not jd.strip() for jd in jd_texts):
        raise HTTPException(status_code=400, detail="At least one job description is required.")

    if not os.environ.get("GROQ_API_KEY"):
        raise HTTPException(
            status_code=500,
            detail="Server is missing GROQ_API_KEY. Add it to backend/.env and restart.",
        )

    resume_bytes = await resume_file.read()
    resume_raw = extract_text_from_upload(resume_file.filename, resume_bytes)

    if not resume_raw.strip():
        raise HTTPException(status_code=400, detail="Could not extract text from resume file.")

    vector_store = ResumeVectorStore()
    vector_store.index_resume(resume_raw)

    results = []
    try:
        for idx, jd_raw in enumerate(jd_texts):
            if not jd_raw.strip():
                continue
            result = run_pipeline_for_jd(
                resume_raw=resume_raw,
                jd_raw=jd_raw,
                jd_index=idx,
                jd_title_fallback=f"Job Description {idx + 1}",
                vector_store=vector_store,
            )
            results.append(result)
    except RateLimitError:
        raise HTTPException(
            status_code=429,
            detail=(
                "The free Groq tier's rate limit was hit (too many requests, "
                "or daily token quota reached). Wait a minute and try again "
                "with fewer job descriptions, or try again tomorrow once the "
                "daily quota resets."
            ),
        )
    except APIConnectionError:
        raise HTTPException(
            status_code=503,
            detail="Could not reach the Groq API. Check your internet connection and try again.",
        )
    except APIStatusError as e:
        if e.status_code == 401:
            detail = "Groq API key is invalid or missing. Check your .env file."
        else:
            detail = f"Groq API returned an error (status {e.status_code}). Try again shortly."
        raise HTTPException(status_code=502, detail=detail)
    finally:
        vector_store.cleanup()

    if not results:
        raise HTTPException(status_code=400, detail="No valid job descriptions to analyze.")

    results.sort(key=lambda r: r.match_score, reverse=True)
    best_match_index = results[0].jd_index

    return AnalysisResponse(results=results, best_match_index=best_match_index)


# Serve the frontend
frontend_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "frontend")
if os.path.isdir(frontend_dir):
    app.mount("/static", StaticFiles(directory=frontend_dir), name="static")

    @app.get("/")
    def serve_index():
        return FileResponse(os.path.join(frontend_dir, "index.html"))