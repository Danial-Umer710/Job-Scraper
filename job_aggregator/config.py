"""Runtime configuration loaded from environment variables (and an optional .env file)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# Fallback only: set GROQ_MODEL in .env (or the environment) to override without code changes.
DEFAULT_GROQ_MODEL = "openai/gpt-oss-120b"

# Company career pages crawled by the "careers" source (job_aggregator/ats_crawler.py).
# Override without code changes via CAREER_PAGE_URLS="https://a.example/jobs,https://b.example/careers".
# Each URL was checked on 2026-11-06 to return real job links. robots.txt is enforced per RFC 9309
# (job_aggregator/robots.py) before every listing or job page is opened.
CAREER_PAGE_URLS: list[str] = [
    "https://careers.epam.com/en/jobs/hungary/budapest",
    "https://www.ibm.com/careers/search?field_keyword_05[0]=Hungary",
    "https://blackrock.wd1.myworkdayjobs.com/BlackRock_Professional?q=Budapest",
    "https://careers.msci.com/job-search?production__mscicare2201__sort-rank%5BrefinementList%5D%5Btown_city_country%5D%5B0%5D=Budapest%20%7C%20Hungary",
    # SAP and Telekom ignore the Budapest filter; non-Budapest locations are skipped before scoring.
    "https://jobs.sap.com/search/?q=&locationsearch=Budapest",
    "https://telekom.jobs/search?location=Budapest",
    # Eightfold sites: robots.txt is "Disallow: /" + "Allow: /careers"; allowed under RFC 9309.
    "https://jobs.ericsson.com/careers?location=Budapest%2C%20Hungary",
    "https://morganstanley.eightfold.ai/careers?location=Budapest%2C%20Hungary",
]

# Not crawled (kept here so they're easy to revisit):
# - Nokia (jobs.nokia.com): Oracle HCM page renders jobs without real links, so nothing can be extracted.
# - Bosch (jobs.bosch.com, SmartRecruiters redirects there): robots.txt disallows /en/ for crawlers.

DEFAULT_CANDIDATE_PROFILE = """\
Candidate: Computer Science Engineering student based in Budapest, Hungary.
Target roles: Data Engineering, DevOps, Cloud / Platform engineering (internship, trainee or junior level).
Key skills:
- Programming & scripting: Python, SQL, Bash
- Cloud: AWS
- Infrastructure as Code: Terraform
- Containers & orchestration: Docker, Kubernetes
- Observability: Prometheus, Grafana
- Data: ETL pipeline design and implementation
Languages: English (professional working proficiency).
"""


def _bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _list(name: str, default: str) -> list[str]:
    return [item.strip() for item in os.getenv(name, default).split(",") if item.strip()]


def _career_page_urls() -> list[str]:
    raw = os.getenv("CAREER_PAGE_URLS")
    if raw is None:  # not set -> use the list in this file
        return list(CAREER_PAGE_URLS)
    return [u.strip() for u in raw.split(",") if u.strip()]


def _load_profile() -> str:
    path = os.getenv("CANDIDATE_PROFILE_FILE")
    if path:
        return Path(path).read_text(encoding="utf-8")
    return DEFAULT_CANDIDATE_PROFILE


@dataclass(frozen=True)
class Settings:
    groq_api_key: str = field(default_factory=lambda: os.getenv("GROQ_API_KEY", "").strip())
    groq_model: str = field(
        default_factory=lambda: os.getenv("GROQ_MODEL", "").strip() or DEFAULT_GROQ_MODEL
    )
    # Spacing between scoring calls. 2 s = 30 requests/minute, Groq's free-tier request limit.
    llm_min_seconds_between_calls: float = field(
        default_factory=lambda: float(os.getenv("LLM_MIN_SECONDS_BETWEEN_CALLS") or "2.0")
    )
    discord_webhook_url: str = field(default_factory=lambda: os.getenv("DISCORD_WEBHOOK_URL", ""))

    state_backend: str = field(default_factory=lambda: os.getenv("STATE_BACKEND", "dynamodb").lower())
    dynamodb_table: str = field(default_factory=lambda: os.getenv("DYNAMODB_TABLE", "job_scraper_state"))
    aws_region: str = field(default_factory=lambda: os.getenv("AWS_REGION", "eu-central-1"))
    local_state_path: str = field(
        default_factory=lambda: os.getenv("LOCAL_STATE_PATH", ".state/processed_jobs.json")
    )
    state_ttl_days: int = field(default_factory=lambda: int(os.getenv("STATE_TTL_DAYS", "180")))

    career_page_urls: list[str] = field(default_factory=_career_page_urls)

    match_threshold: int = field(default_factory=lambda: int(os.getenv("MATCH_THRESHOLD", "65")))
    max_jobs_per_run: int = field(default_factory=lambda: int(os.getenv("MAX_JOBS_PER_RUN", "40")))
    dry_run: bool = field(default_factory=lambda: _bool("DRY_RUN"))
    sources: list[str] = field(default_factory=lambda: _list("SOURCES", "careers"))

    http_timeout_seconds: float = field(default_factory=lambda: float(os.getenv("HTTP_TIMEOUT", "30")))
    candidate_profile: str = field(default_factory=_load_profile)

    def validate(self) -> None:
        if not self.groq_api_key:
            raise ValueError(
                "GROQ_API_KEY is missing or empty. Set it in .env (local) or as a GitHub Actions "
                "secret; get a key at https://console.groq.com/keys"
            )
        missing = []
        if not self.dry_run and not self.discord_webhook_url:
            missing.append("DISCORD_WEBHOOK_URL")
        if self.state_backend not in {"dynamodb", "local"}:
            raise ValueError(f"STATE_BACKEND must be 'dynamodb' or 'local', got {self.state_backend!r}")
        bad_urls = [u for u in self.career_page_urls if not u.startswith(("http://", "https://"))]
        if "careers" in self.sources and bad_urls:
            raise ValueError(f"CAREER_PAGE_URLS must be plain http(s) URLs, got: {bad_urls}")
        if missing:
            raise ValueError(f"Missing required environment variables: {', '.join(missing)}")
