"""Discord webhook delivery using rich embeds."""

from __future__ import annotations

import json
import logging
import time

import requests

from .models import ScoredJob

log = logging.getLogger(__name__)

EMBEDS_PER_MESSAGE = 10  # Discord hard limit
MAX_TOTAL_EMBED_CHARS = 6000  # Discord hard limit per message


def _truncate(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _color(score: int) -> int:
    if score >= 85:
        return 0x2ECC71  # green
    if score >= 75:
        return 0xF1C40F  # yellow
    return 0xE67E22  # orange


def build_embed(scored: ScoredJob) -> dict:
    job, match = scored.job, scored.match
    missing = ", ".join(match.missing_skills) if match.missing_skills else "None, strong match"
    return {
        "title": _truncate(f"{job.title} @ {job.company}", 256),
        "url": job.url,
        "color": _color(match.match_score_percentage),
        "fields": [
            {"name": "Match score", "value": f"**{match.match_score_percentage}%**", "inline": True},
            {"name": "Source", "value": job.source, "inline": True},
            {"name": "Apply", "value": _truncate(f"[Open posting]({job.url})", 1024), "inline": True},
            {"name": "Missing skills", "value": _truncate(missing, 1024), "inline": False},
            {
                "name": "Suggested intro sentence",
                "value": _truncate(match.intro_sentence_for_application or "-", 1024),
                "inline": False,
            },
        ],
        "footer": {"text": "Budapest Job Aggregator"},
    }


def _embed_size(embed: dict) -> int:
    size = len(embed.get("title", "")) + len(embed.get("footer", {}).get("text", ""))
    size += sum(len(f["name"]) + len(f["value"]) for f in embed.get("fields", []))
    return size


def _chunk(embeds: list[dict]) -> list[list[dict]]:
    batches: list[list[dict]] = []
    current: list[dict] = []
    current_size = 0
    for embed in embeds:
        size = _embed_size(embed)
        if current and (len(current) >= EMBEDS_PER_MESSAGE or current_size + size > MAX_TOTAL_EMBED_CHARS):
            batches.append(current)
            current, current_size = [], 0
        current.append(embed)
        current_size += size
    if current:
        batches.append(current)
    return batches


class DiscordNotifier:
    def __init__(self, webhook_url: str, dry_run: bool = False, timeout: float = 15) -> None:
        self.webhook_url = webhook_url
        self.dry_run = dry_run
        self.timeout = timeout
        self.session = requests.Session()

    def _post(self, payload: dict, max_attempts: int = 5) -> None:
        if self.dry_run:
            print(json.dumps(payload, indent=2, ensure_ascii=False))
            return
        for attempt in range(1, max_attempts + 1):
            response = self.session.post(self.webhook_url, json=payload, timeout=self.timeout)
            if response.status_code == 429:
                retry_after = float(response.json().get("retry_after", 1.0))
                log.warning("discord: rate limited, sleeping %.2fs", retry_after)
                time.sleep(retry_after + 0.25)
                continue
            if response.status_code >= 500 and attempt < max_attempts:
                time.sleep(2**attempt)
                continue
            response.raise_for_status()
            return
        raise RuntimeError("discord: exhausted retries")

    def send(self, scored_jobs: list[ScoredJob]) -> list[ScoredJob]:
        """Send alerts. Returns the jobs that were delivered successfully."""
        ordered = sorted(scored_jobs, key=lambda s: s.match.match_score_percentage, reverse=True)
        delivered: list[ScoredJob] = []
        index = 0
        for batch in _chunk([build_embed(s) for s in ordered]):
            jobs_in_batch = ordered[index : index + len(batch)]
            index += len(batch)
            payload = {
                "username": "Budapest Job Radar",
                "content": f"{len(batch)} new matching role(s) in Budapest",
                "embeds": batch,
                "allowed_mentions": {"parse": []},
            }
            try:
                self._post(payload)
                delivered.extend(jobs_in_batch)
            except (requests.RequestException, RuntimeError) as exc:
                log.error("discord: failed to deliver batch of %d: %s", len(batch), exc)
        return delivered
