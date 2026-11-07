"""Pydantic schemas shared across the pipeline."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

Source = Literal["careers", "nofluffjobs", "profession.hu"]


class JobPosting(BaseModel):
    """Normalized job posting, independent of the board it came from.

    `job_id` is namespaced by source (e.g. ``nofluffjobs:devops-engineer-acme-budapest``)
    so IDs from different boards can never collide in the state table.
    """

    model_config = ConfigDict(str_strip_whitespace=True, frozen=True)

    job_id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    company: str = "Unknown company"
    url: str
    description: str = ""
    source: Source

    @field_validator("url")
    @classmethod
    def _url_must_be_http(cls, value: str) -> str:
        if not value.startswith(("http://", "https://")):
            raise ValueError(f"url must be absolute http(s), got {value!r}")
        return value

    @field_validator("company", mode="before")
    @classmethod
    def _default_company(cls, value: object) -> object:
        return value or "Unknown company"


class MatchResult(BaseModel):
    """Structured output contract for the LLM scorer (JSON mode via Groq).

    JSON mode only guarantees syntactically valid JSON, not this shape, so every response
    is validated here and bounds are enforced in Python after parsing.
    """

    match_score_percentage: int
    missing_skills: list[str]
    intro_sentence_for_application: str

    @field_validator("match_score_percentage")
    @classmethod
    def _clamp_score(cls, value: int) -> int:
        return max(0, min(100, value))

    @field_validator("missing_skills")
    @classmethod
    def _clean_skills(cls, value: list[str]) -> list[str]:
        seen: list[str] = []
        for skill in value:
            skill = skill.strip()
            if skill and skill.lower() not in {s.lower() for s in seen}:
                seen.append(skill)
        return seen


class ScoredJob(BaseModel):
    job: JobPosting
    match: MatchResult
