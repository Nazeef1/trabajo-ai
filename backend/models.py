"""
Pydantic models used across the Trabajo AI pipeline.
"""
from typing import List, Optional
from pydantic import BaseModel, Field


class ParsedResume(BaseModel):
    skills: List[str] = Field(default_factory=list)
    experience_years: Optional[float] = None
    education: List[str] = Field(default_factory=list)
    job_titles: List[str] = Field(default_factory=list)
    summary: str = ""
    raw_text: str = ""


class ParsedJD(BaseModel):
    title: str = ""
    required_skills: List[str] = Field(default_factory=list)
    preferred_skills: List[str] = Field(default_factory=list)
    min_experience_years: Optional[float] = None
    qualifications: List[str] = Field(default_factory=list)
    raw_text: str = ""


class GapItem(BaseModel):
    category: str  # e.g. "skill", "experience", "qualification"
    description: str
    severity: str  # "critical", "moderate", "minor"


class MatchResult(BaseModel):
    jd_title: str
    jd_index: int
    match_score: int  # 0-100
    matched_skills: List[str] = Field(default_factory=list)
    missing_skills: List[str] = Field(default_factory=list)
    gaps: List[GapItem] = Field(default_factory=list)
    rationale: str = ""


class AnalysisResponse(BaseModel):
    results: List[MatchResult]
    best_match_index: int
