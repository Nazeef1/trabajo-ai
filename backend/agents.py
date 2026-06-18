"""
Trabajo AI multi-agent pipeline, built with LangGraph.
Pipeline (per JD, run sequentially as a graph):
  parse_node       -> structures raw resume + JD text into typed fields
  retrieve_node    -> RAG: pulls the resume chunks most relevant to this JD's
                      requirements from the vector store
  reason_node      -> scores the match using structured data + retrieved context
  gap_node         -> produces a specific, explainable gap analysis
"""
import json
import os
from typing import TypedDict, List, Optional

from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, END

from .models import ParsedResume, ParsedJD, GapItem, MatchResult
from .vectorstore import ResumeVectorStore

GROQ_MODEL = "llama-3.3-70b-versatile"


def get_llm(temperature: float = 0.1) -> ChatGroq:
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY is not set. Add it to your .env file. "
            "Get a free key at https://console.groq.com"
        )
    return ChatGroq(model=GROQ_MODEL, temperature=temperature, api_key=api_key)


def _extract_json(text: str) -> dict:
    """LLMs sometimes wrap JSON in markdown fences or add preamble text.
    Strip that defensively before parsing."""
    text = text.strip()
    if "```" in text:
        # Grab content between the first pair of triple backticks
        parts = text.split("```")
        for part in parts:
            part = part.strip()
            if part.startswith("json"):
                part = part[4:].strip()
            if part.startswith("{") or part.startswith("["):
                text = part
                break
    # Fallback: find first { and last }
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        text = text[start:end + 1]
    return json.loads(text)


class GraphState(TypedDict):
    resume_raw: str
    jd_raw: str
    jd_index: int
    parsed_resume: Optional[dict]
    parsed_jd: Optional[dict]
    retrieved_context: Optional[List[str]]
    match_score: Optional[int]
    matched_skills: Optional[List[str]]
    missing_skills: Optional[List[str]]
    rationale: Optional[str]
    gaps: Optional[List[dict]]
    vector_store: Optional[object]  # ResumeVectorStore, excluded from serialization


# ---------------------------------------------------------------------------
# Agent 1: Parser — structures raw resume & JD text into typed fields
# ---------------------------------------------------------------------------
def parse_node(state: GraphState) -> dict:
    llm = get_llm()
    prompt = f"""You are a precise information extraction system. Extract structured
data from the resume and job description below. Respond with ONLY a JSON object,
no preamble, no markdown fences.

RESUME:
{state['resume_raw'][:6000]}

JOB DESCRIPTION:
{state['jd_raw'][:4000]}

Return JSON with this exact shape:
{{
  "resume": {{
    "skills": ["list of technical skills/tools/languages mentioned"],
    "experience_years": <number, estimate of total relevant professional/internship experience in years, or null>,
    "education": ["degree and institution strings"],
    "job_titles": ["titles held"],
    "summary": "1-2 sentence summary of the candidate"
  }},
  "jd": {{
    "title": "job title",
    "required_skills": ["explicitly required skills/qualifications"],
    "preferred_skills": ["nice-to-have / preferred skills"],
    "min_experience_years": <number or null>,
    "qualifications": ["other qualifications like degree, CGPA, availability"]
  }}
}}"""
    response = llm.invoke(prompt)
    data = _extract_json(response.content)
    return {
        "parsed_resume": data.get("resume", {}),
        "parsed_jd": data.get("jd", {}),
    }


# ---------------------------------------------------------------------------
# Agent 2: Retriever — RAG step, pulls relevant resume chunks per JD requirement
# ---------------------------------------------------------------------------
def retrieve_node(state: GraphState) -> dict:
    vs: ResumeVectorStore = state["vector_store"]
    parsed_jd = state["parsed_jd"] or {}
    required = parsed_jd.get("required_skills", []) or []
    preferred = parsed_jd.get("preferred_skills", []) or []
    query_terms = required + preferred
    if not query_terms:
        query_terms = [state["jd_raw"][:200]]

    seen = set()
    retrieved = []
    for term in query_terms[:8]:  # cap to keep latency reasonable
        chunks = vs.retrieve(term, k=3)
        for c in chunks:
            if c not in seen:
                seen.add(c)
                retrieved.append(c)

    return {"retrieved_context": retrieved[:12]}  # cap total context size


# ---------------------------------------------------------------------------
# Agent 3: Reasoner — scores the match using structured data + retrieved context
# ---------------------------------------------------------------------------
def reason_node(state: GraphState) -> dict:
    llm = get_llm()
    context_block = "\n".join(f"- {c}" for c in (state["retrieved_context"] or []))
    prompt = f"""You are an expert technical recruiter evaluating candidate fit.
Respond with ONLY a JSON object, no preamble, no markdown fences.

CANDIDATE PROFILE (structured):
{json.dumps(state['parsed_resume'], indent=2)}

JOB REQUIREMENTS (structured):
{json.dumps(state['parsed_jd'], indent=2)}

RELEVANT RESUME EVIDENCE (retrieved via semantic search against the JD's requirements):
{context_block}

Evaluate the match. Be honest and discriminating — do not give high scores by default.
Weigh required skills more heavily than preferred skills. Consider experience level fit.

Return JSON with this exact shape:
{{
  "match_score": <integer 0-100>,
  "matched_skills": ["skills/requirements the candidate clearly satisfies"],
  "missing_skills": ["required or preferred skills the candidate appears to lack"],
  "rationale": "2-3 sentence explanation of the score, referencing specific evidence"
}}"""
    response = llm.invoke(prompt)
    data = _extract_json(response.content)
    return {
        "match_score": data.get("match_score", 0),
        "matched_skills": data.get("matched_skills", []),
        "missing_skills": data.get("missing_skills", []),
        "rationale": data.get("rationale", ""),
    }


# ---------------------------------------------------------------------------
# Agent 4: Gap Analyzer — specific, categorized, severity-ranked gap analysis
# ---------------------------------------------------------------------------
def gap_node(state: GraphState) -> dict:
    llm = get_llm()
    prompt = f"""You are a career coach helping a candidate understand exactly what
separates them from a strong fit for this role. Respond with ONLY a JSON object,
no preamble, no markdown fences.

CANDIDATE PROFILE:
{json.dumps(state['parsed_resume'], indent=2)}

JOB REQUIREMENTS:
{json.dumps(state['parsed_jd'], indent=2)}

MISSING SKILLS IDENTIFIED:
{json.dumps(state['missing_skills'], indent=2)}

For each meaningful gap, classify it and explain it specifically and actionably
(not generic advice). Return JSON with this exact shape:
{{
  "gaps": [
    {{
      "category": "skill" | "experience" | "qualification",
      "description": "specific, actionable description of the gap",
      "severity": "critical" | "moderate" | "minor"
    }}
  ]
}}
If there are no meaningful gaps, return an empty list."""
    response = llm.invoke(prompt)
    data = _extract_json(response.content)
    return {"gaps": data.get("gaps", [])}


def build_graph():
    graph = StateGraph(GraphState)
    graph.add_node("parse", parse_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("reason", reason_node)
    graph.add_node("gap_analysis", gap_node)

    graph.set_entry_point("parse")
    graph.add_edge("parse", "retrieve")
    graph.add_edge("retrieve", "reason")
    graph.add_edge("reason", "gap_analysis")
    graph.add_edge("gap_analysis", END)

    return graph.compile()


_compiled_graph = None


def get_compiled_graph():
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = build_graph()
    return _compiled_graph


def run_pipeline_for_jd(
    resume_raw: str,
    jd_raw: str,
    jd_index: int,
    jd_title_fallback: str,
    vector_store: ResumeVectorStore,
) -> MatchResult:
    """Run the full agent graph for a single JD and return a MatchResult."""
    graph = get_compiled_graph()
    initial_state: GraphState = {
        "resume_raw": resume_raw,
        "jd_raw": jd_raw,
        "jd_index": jd_index,
        "parsed_resume": None,
        "parsed_jd": None,
        "retrieved_context": None,
        "match_score": None,
        "matched_skills": None,
        "missing_skills": None,
        "rationale": None,
        "gaps": None,
        "vector_store": vector_store,
    }
    final_state = graph.invoke(initial_state)

    parsed_jd = final_state.get("parsed_jd") or {}
    title = parsed_jd.get("title") or jd_title_fallback

    gaps = [
        GapItem(
            category=g.get("category", "skill"),
            description=g.get("description", ""),
            severity=g.get("severity", "moderate"),
        )
        for g in (final_state.get("gaps") or [])
    ]

    return MatchResult(
        jd_title=title,
        jd_index=jd_index,
        match_score=final_state.get("match_score", 0),
        matched_skills=final_state.get("matched_skills") or [],
        missing_skills=final_state.get("missing_skills") or [],
        gaps=gaps,
        rationale=final_state.get("rationale", ""),
    )
