"""Adapter for career sites hosted on Eightfold (e.g. Ericsson, Morgan Stanley).

Eightfold listing and job pages render as a large embedded config blob and draw job cards with
click handlers instead of links. So the LLM link extraction in ats_crawler.py finds nothing
on them, and chunking the blob would cost ~6 LLM calls per page. Instead, this module reads the
same JSON endpoints the Eightfold front-end uses: no browser and no LLM calls.

Their robots.txt explicitly allows them ("Allow: /api/pcsx"), and every request is still
checked with RobotsPolicy (RFC 9309) first.
"""

from __future__ import annotations

import logging
import re
import time
from urllib.parse import parse_qs, urlencode, urljoin, urlsplit

import requests

from .extractor import html_to_text
from .robots import RobotsPolicy

log = logging.getLogger(__name__)

# Eightfold sites on a company's own domain: host -> Eightfold "domain" parameter.
# Sites on <company>.eightfold.ai are detected automatically (domain = "<company>.com").
EIGHTFOLD_SITES = {
    "jobs.ericsson.com": "ericsson.com",
}
JOB_PATH_RE = re.compile(r"/careers/job/(\d+)")
PAGE_SIZE = 10  # fixed by the Eightfold API

# The search endpoint returns every role (finance, HR, ...), unlike the LLM extraction used for other
# sites, which only keeps tech roles. Match title + department against tech terms so non-tech roles
# don't spend Groq free-tier scoring calls.
TECH_ROLE_RE = re.compile(
    r"\b(software|developer|engineer(ing)?|devops|devsecops|sre|site reliability|cloud|platform|"
    r"infrastructure|data|machine learning|ml|ai|python|java|golang|linux|network|security|cyber|"
    r"it|technology|tech|qa|test(ing)?|automation|kubernetes|aws|azure|gcp|r&d|product development)\b",
    re.IGNORECASE,
)


def eightfold_domain(url: str) -> str | None:
    """Return the Eightfold domain parameter if `url` is on an Eightfold-hosted site."""
    host = urlsplit(url).netloc.lower().split(":")[0]
    if host in EIGHTFOLD_SITES:
        return EIGHTFOLD_SITES[host]
    if host.endswith(".eightfold.ai"):
        return host.split(".")[0] + ".com"
    return None


class EightfoldClient:
    def __init__(
        self,
        session: requests.Session,
        robots: RobotsPolicy,
        timeout: float = 30,
        max_pages: int = 6,
        delay_seconds: float = 1.0,
    ) -> None:
        self.session = session
        self.robots = robots
        self.timeout = timeout
        self.max_pages = max_pages
        self.delay_seconds = delay_seconds

    def _get_json(self, url: str) -> dict | None:
        if not self.robots.allowed(url):
            log.warning("eightfold: %s is disallowed by robots.txt; skipping", url)
            return None
        try:
            response = self.session.get(url, headers={"Accept": "application/json"}, timeout=self.timeout)
            response.raise_for_status()
            payload = response.json()
        except (requests.RequestException, ValueError) as exc:
            log.warning("eightfold: %s failed: %s", url, exc)
            return None
        return payload.get("data", payload) if isinstance(payload, dict) else None

    def search(self, listing_url: str) -> list[dict]:
        """Jobs for a listing URL such as https://jobs.ericsson.com/careers?location=Budapest%2C%20Hungary.

        Returns dicts with title, url and location (the shape of ats_crawler.ExtractedJob).
        """
        domain = eightfold_domain(listing_url)
        if not domain:
            return []
        parts = urlsplit(listing_url)
        origin = f"{parts.scheme}://{parts.netloc}"
        params = {k: v[0] for k, v in parse_qs(parts.query).items()}

        jobs: dict[str, dict] = {}
        skipped_non_tech = 0
        start, total = 0, None
        for page in range(self.max_pages):
            query = urlencode({
                "domain": domain,
                "query": params.get("query", ""),
                "location": params.get("location", ""),
                "start": start,
                "sort_by": "timestamp",  # newest first, so capped runs still see new postings
            })
            data = self._get_json(f"{origin}/api/pcsx/search?{query}")
            if not data:
                break
            positions = data.get("positions") or []
            total = data.get("count", total)
            for position in positions:
                if not TECH_ROLE_RE.search(f"{position.get('name', '')} {position.get('department', '')}"):
                    skipped_non_tech += 1
                    continue
                path = position.get("positionUrl") or f"/careers/job/{position.get('id')}"
                url = urljoin(origin, path)
                locations = position.get("locations") or []
                jobs.setdefault(url, {
                    "title": (position.get("name") or "").strip(),
                    "url": url,
                    "location": "; ".join(str(loc) for loc in locations),
                })
            start += len(positions)
            if not positions or (total is not None and start >= total):
                break
            if page + 1 < self.max_pages:
                time.sleep(self.delay_seconds)

        log.info(
            "eightfold: %s -> %d tech job(s), %d non-tech skipped (site reports %s)",
            listing_url, len(jobs), skipped_non_tech, total,
        )
        return [job for job in jobs.values() if job["title"]]

    def description(self, job_url: str) -> str | None:
        domain = eightfold_domain(job_url)
        match = JOB_PATH_RE.search(urlsplit(job_url).path)
        if not domain or not match:
            return None
        parts = urlsplit(job_url)
        query = urlencode({"position_id": match.group(1), "domain": domain, "hl": "en"})
        data = self._get_json(f"{parts.scheme}://{parts.netloc}/api/pcsx/position_details?{query}")
        if not data:
            return None
        text = html_to_text(data.get("jobDescription") or "")
        if not text:
            return None
        header = [
            f"Location: {'; '.join(map(str, data.get('locations') or [])) or data.get('location', '')}",
            f"Work location option: {data.get('workLocationOption') or data.get('locationFlexibility') or 'n/a'}",
            f"Department: {data.get('department') or 'n/a'}",
        ]
        return "\n".join(header) + "\n\n" + text
