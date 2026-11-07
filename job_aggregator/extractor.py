"""API/RSS extraction and normalization into `JobPosting` objects.

No HTML scraping: NoFluffJobs is read from its public JSON API and Profession.hu from its
RSS feeds (parsed with the standard-library XML parser).
"""

from __future__ import annotations

import html
import json
import logging
import re
import unicodedata
import xml.etree.ElementTree as ET
from abc import ABC, abstractmethod
from urllib.parse import quote

import requests
from pydantic import ValidationError
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from .models import JobPosting

log = logging.getLogger(__name__)

USER_AGENT = "budapest-job-aggregator/1.0 (+personal job alert pipeline)"

_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"[ \t\r\f\v]+")
_BLOCK_TAGS_RE = re.compile(r"</?(p|div|br|li|ul|ol|h[1-6])[^>]*>", re.IGNORECASE)

# Seniority words (English + Hungarian) used to filter keyword-based feeds.
SENIORITY_RE = re.compile(
    r"\b(junior|jr\.?|intern(ship)?|trainee|graduate|entry[- ]level|student|"
    r"gyakornok|pályakezdő|diák(munka)?)\b",
    re.IGNORECASE,
)
# Domain words for Data Engineering / DevOps / Cloud roles.
DOMAIN_RE = re.compile(
    r"(devops|dev ops|devsecops|\bsre\b|site reliability|cloud|platform|infrastructure|"
    r"infrastruktúra|data engineer|adatmérnök|\betl\b|\bdata\b|\badat|big data|\baws\b|azure|\bgcp\b|"
    r"kubernetes|docker|terraform|linux|python|\bsql\b|rendszergazda|üzemeltet|system administrator|"
    r"sysadmin|automation|automatizál|mlops|dataops|backend)",
    re.IGNORECASE,
)


def html_to_text(raw: str | None) -> str:
    """Convert a small HTML fragment from an API field into readable plain text."""
    if not raw:
        return ""
    text = re.sub(r"\s*\n\s*", " ", raw)  # source line wraps are not meaningful in HTML
    text = _BLOCK_TAGS_RE.sub("\n", text)
    text = _TAG_RE.sub("", text)
    text = html.unescape(text)
    lines = [_WS_RE.sub(" ", line).strip() for line in text.splitlines()]
    return "\n".join(line for line in lines if line)


def build_session(timeout_retries: int = 3) -> requests.Session:
    session = requests.Session()
    retry = Retry(
        total=timeout_retries,
        backoff_factor=1.5,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset({"GET", "POST"}),
        respect_retry_after_header=True,
    )
    session.mount("https://", HTTPAdapter(max_retries=retry))
    session.headers.update({"User-Agent": USER_AGENT, "Accept-Language": "en-GB,en;q=0.9,hu;q=0.8"})
    return session


class BaseExtractor(ABC):
    source_name: str

    def __init__(self, session: requests.Session | None = None, timeout: float = 30) -> None:
        self.session = session or build_session()
        self.timeout = timeout

    @abstractmethod
    def list_jobs(self) -> list[JobPosting]:
        """Return lightweight postings (description may be partial)."""

    def hydrate(self, job: JobPosting) -> JobPosting:
        """Return the posting with its full description. Called only for net-new jobs."""
        return job

    def hydrate_many(self, jobs: list[JobPosting]) -> list[JobPosting]:
        """Hydrate a batch of net-new jobs. Jobs left out of the result are retried next run."""
        return [self.hydrate(job) for job in jobs]


class NoFluffJobsExtractor(BaseExtractor):
    """NoFluffJobs public search API (the same JSON API its web front-end uses)."""

    source_name = "nofluffjobs"
    SEARCH_URL = "https://nofluffjobs.com/api/search/posting"
    DETAIL_URL = "https://nofluffjobs.com/api/posting/{posting_id}"
    PUBLIC_URL = "https://nofluffjobs.com/hu/job/{slug}"

    # NoFluffJobs category slugs relevant to Data Engineering / DevOps / Cloud.
    DEFAULT_CATEGORIES = (
        "devops",
        "data",
        "bigData",
        "sysAdministrator",
        "backend",
        "architecture",
        "artificialIntelligence",
    )

    def __init__(
        self,
        city: str = "budapest",
        seniority: tuple[str, ...] = ("trainee", "junior"),
        categories: tuple[str, ...] = DEFAULT_CATEGORIES,
        max_pages: int = 10,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.city = city
        self.seniority = seniority
        self.categories = categories
        self.max_pages = max_pages

    def _search_page(self, page: int) -> dict:
        response = self.session.post(
            self.SEARCH_URL,
            params={
                "page": page,
                "salaryCurrency": "HUF",
                "salaryPeriod": "month",
                "region": "hu",
                "language": "en-GB",
            },
            headers={
                "Content-Type": "application/infiniteSearch+json",
                "Accept": "application/json",
            },
            json={
                "criteriaSearch": {
                    "city": [self.city],
                    "seniority": list(self.seniority),
                    "category": list(self.categories),
                },
                "page": page,
            },
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json()

    def list_jobs(self) -> list[JobPosting]:
        jobs: dict[str, JobPosting] = {}
        page, total_pages = 1, 1
        wanted_seniority = {s.lower() for s in self.seniority}

        while page <= min(total_pages, self.max_pages):
            payload = self._search_page(page)
            total_pages = int(payload.get("totalPages") or 1)
            for raw in payload.get("postings", []):
                seniority = {s.lower() for s in raw.get("seniority", [])}
                if wanted_seniority and not seniority & wanted_seniority:
                    continue
                job = self._normalize_listing(raw)
                if job:
                    jobs.setdefault(job.job_id, job)
            page += 1

        log.info("nofluffjobs: %d postings after filtering", len(jobs))
        return list(jobs.values())

    def _normalize_listing(self, raw: dict) -> JobPosting | None:
        try:
            posting_id = raw["id"]
            slug = raw.get("url") or posting_id.lower()
            tiles = ", ".join(
                t.get("value", "") for t in raw.get("tiles", {}).get("values", []) if t.get("value")
            )
            return JobPosting(
                job_id=f"{self.source_name}:{posting_id}",
                title=raw["title"],
                company=raw.get("name", ""),
                url=self.PUBLIC_URL.format(slug=slug),
                description=f"Category: {raw.get('category', '')}. Tech: {tiles}".strip(),
                source="nofluffjobs",
            )
        except (KeyError, ValidationError) as exc:
            log.warning("nofluffjobs: skipping malformed posting %s: %s", raw.get("id"), exc)
            return None

    def hydrate(self, job: JobPosting) -> JobPosting:
        posting_id = job.job_id.split(":", 1)[1]
        try:
            response = self.session.get(
                self.DETAIL_URL.format(posting_id=quote(posting_id, safe="")),
                headers={"Accept": "application/json"},
                timeout=self.timeout,
            )
            response.raise_for_status()
            detail = response.json()
        except (requests.RequestException, ValueError) as exc:
            log.warning("nofluffjobs: detail fetch failed for %s (%s); using listing data", job.job_id, exc)
            return job

        reqs = detail.get("requirements") or {}
        musts = [m.get("value", "").strip() for m in reqs.get("musts", []) if m.get("value")]
        nices = [n.get("value", "").strip() for n in reqs.get("nices", []) if n.get("value")]
        tasks = (detail.get("specs") or {}).get("dailyTasks") or []

        sections = [
            ("About the role", html_to_text((detail.get("details") or {}).get("description"))),
            ("Requirements", html_to_text(reqs.get("description"))),
            ("Must-have", ", ".join(musts)),
            ("Nice-to-have", ", ".join(nices)),
            ("Daily tasks", "\n".join(f"- {t}" for t in tasks)),
        ]
        description = "\n\n".join(f"{name}:\n{body}" for name, body in sections if body)
        return job.model_copy(update={"description": description or job.description})


class ProfessionHuExtractor(BaseExtractor):
    """Profession.hu keyword RSS feeds restricted to Budapest."""

    source_name = "profession.hu"
    FEED_URL = "https://www.profession.hu/allasok/budapest/1,0,1,{keyword}?rss=1"
    DEFAULT_KEYWORDS = (
        "junior devops",
        "devops gyakornok",
        "junior data engineer",
        "data engineer gyakornok",
        "junior cloud engineer",
        "cloud gyakornok",
        "devops intern",
        "data engineering intern",
        "junior sre",
        "junior python",
    )
    _ID_RE = re.compile(r"-(\d+)(?:[/?#].*)?$")
    _SLUG_RE = re.compile(r"/allas/([^/?#]+)")

    def __init__(self, keywords: tuple[str, ...] = DEFAULT_KEYWORDS, **kwargs) -> None:
        super().__init__(**kwargs)
        self.keywords = keywords

    def _fetch_feed(self, keyword: str) -> bytes:
        url = self.FEED_URL.format(keyword=quote(keyword))
        response = self.session.get(
            url, headers={"Accept": "application/rss+xml, application/xml"}, timeout=self.timeout
        )
        response.raise_for_status()
        return response.content

    def list_jobs(self) -> list[JobPosting]:
        jobs: dict[str, JobPosting] = {}
        for keyword in self.keywords:
            try:
                xml_bytes = self._fetch_feed(keyword)
            except requests.RequestException as exc:
                log.warning("profession.hu: feed %r failed: %s", keyword, exc)
                continue
            for job in self.parse_feed(xml_bytes):
                jobs.setdefault(job.job_id, job)
        log.info("profession.hu: %d postings after filtering", len(jobs))
        return list(jobs.values())

    def parse_feed(self, xml_bytes: bytes) -> list[JobPosting]:
        try:
            root = ET.fromstring(xml_bytes)
        except ET.ParseError as exc:
            log.warning("profession.hu: invalid RSS: %s", exc)
            return []

        results: list[JobPosting] = []
        for item in root.iterfind("./channel/item"):
            title = (item.findtext("title") or "").strip()
            link = (item.findtext("guid") or item.findtext("link") or "").strip()
            link = link.split("?", 1)[0]
            if not title or not link:
                continue
            if not (SENIORITY_RE.search(title) and DOMAIN_RE.search(title)):
                continue

            id_match = self._ID_RE.search(link)
            native_id = id_match.group(1) if id_match else link
            try:
                results.append(
                    JobPosting(
                        job_id=f"{self.source_name}:{native_id}",
                        title=title,
                        company=self._company_hint(link, title),
                        url=link,
                        description=self._parse_description(item.findtext("description")),
                        source="profession.hu",
                    )
                )
            except ValidationError as exc:
                log.warning("profession.hu: skipping malformed item %s: %s", link, exc)
        return results

    @staticmethod
    def _parse_description(raw: str | None) -> str:
        """Profession.hu puts a JSON array of bullet strings in <description>."""
        if not raw:
            return ""
        raw = html.unescape(raw.strip())
        try:
            bullets = json.loads(raw)
            if isinstance(bullets, list):
                return "\n".join(f"- {html_to_text(str(b))}" for b in bullets if str(b).strip())
        except json.JSONDecodeError:
            pass
        return html_to_text(raw)

    @staticmethod
    def _ascii_tokens(text: str) -> list[str]:
        folded = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode().lower()
        return re.findall(r"[a-z0-9]+", folded)

    @classmethod
    def _company_hint(cls, link: str, title: str) -> str:
        """RSS has no company field; derive a best-effort hint from the URL slug.

        Slugs look like ``<title-words>-<company-words>-<city>-<id>``, so leading slug
        tokens that also occur in the title are dropped, plus a trailing ``budapest``.
        """
        match = cls._SLUG_RE.search(link)
        if not match:
            return ""
        slug_tokens = match.group(1).split("-")
        if slug_tokens and slug_tokens[-1].isdigit():
            slug_tokens.pop()
        title_tokens = set(cls._ascii_tokens(title))
        while slug_tokens and slug_tokens[0] in title_tokens:
            slug_tokens.pop(0)
        if slug_tokens and slug_tokens[-1] == "budapest":
            slug_tokens.pop()
        return " ".join(slug_tokens).title()


def build_extractors(settings) -> list[BaseExtractor]:
    """Build the extractors named in settings.sources (careers, nofluffjobs, profession)."""
    session = build_session()
    timeout = settings.http_timeout_seconds

    def careers() -> BaseExtractor:
        from .ats_crawler import CareerPageExtractor  # lazy: crawl4ai/Chromium only when used

        return CareerPageExtractor(
            settings.career_page_urls,
            api_key=settings.groq_api_key,
            model=settings.groq_model,
            session=session,
            timeout=timeout,
        )

    registry = {
        "careers": careers,
        "nofluffjobs": lambda: NoFluffJobsExtractor(session=session, timeout=timeout),
        "profession": lambda: ProfessionHuExtractor(session=session, timeout=timeout),
    }
    unknown = set(settings.sources) - registry.keys()
    if unknown:
        raise ValueError(f"Unknown SOURCES: {', '.join(sorted(unknown))}")
    return [registry[name]() for name in settings.sources]
