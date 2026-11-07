"""Pipeline entry point: extract -> dedupe -> hydrate -> score -> alert -> persist state."""

from __future__ import annotations

import logging
import sys

from .ai_scorer import LLMScorer, ScorerAuthError, ScorerQuotaError
from .config import Settings
from .discord_notifier import DiscordNotifier
from .extractor import BaseExtractor, build_extractors
from .models import JobPosting, ScoredJob
from .state import build_state_store

log = logging.getLogger("job_aggregator")


def collect(extractors: list[BaseExtractor]) -> tuple[list[tuple[BaseExtractor, JobPosting]], int]:
    """Run every extractor; one failing board must not break the others."""
    collected: dict[str, tuple[BaseExtractor, JobPosting]] = {}
    failures = 0
    for extractor in extractors:
        try:
            for job in extractor.list_jobs():
                collected.setdefault(job.job_id, (extractor, job))
        except ScorerAuthError:
            raise  # bad key/model would break every source: abort the run
        except Exception:  # noqa: BLE001 - isolate source failures
            failures += 1
            log.exception("extractor %s failed", extractor.source_name)
    return list(collected.values()), failures


def run(settings: Settings) -> int:
    settings.validate()

    extractors = build_extractors(settings)
    state = build_state_store(
        settings.state_backend,
        table=settings.dynamodb_table,
        region=settings.aws_region,
        local_path=settings.local_state_path,
        ttl_days=settings.state_ttl_days,
    )
    scorer = LLMScorer(
        api_key=settings.groq_api_key,
        candidate_profile=settings.candidate_profile,
        model=settings.groq_model,
        min_seconds_between_calls=settings.llm_min_seconds_between_calls,
    )
    notifier = DiscordNotifier(settings.discord_webhook_url, dry_run=settings.dry_run)

    # Phase 1: extraction & normalization
    candidates, failures = collect(extractors)
    log.info("extracted %d unique postings from %d source(s)", len(candidates), len(extractors))
    if failures == len(extractors):
        log.error("all sources failed")
        return 2

    # Phase 2: state check
    new_ids = state.filter_new([job.job_id for _, job in candidates])
    new_jobs = [(ex, job) for ex, job in candidates if job.job_id in new_ids]
    log.info("%d net-new postings", len(new_jobs))
    if len(new_jobs) > settings.max_jobs_per_run:
        log.info("capping this run at %d; the rest will be picked up next run", settings.max_jobs_per_run)
        new_jobs = new_jobs[: settings.max_jobs_per_run]

    # Phase 3: fetch descriptions for NET-NEW jobs only (one batch per source), then score
    hydrated: list[JobPosting] = []
    for extractor in extractors:
        batch = [job for ex, job in new_jobs if ex is extractor]
        if not batch:
            continue
        try:
            hydrated.extend(extractor.hydrate_many(batch))
        except Exception:  # noqa: BLE001 - retry this source's jobs next run
            log.exception("fetching job descriptions from %s failed", extractor.source_name)
    log.info("%d/%d new postings have descriptions", len(hydrated), len(new_jobs))

    matches: list[ScoredJob] = []
    below_threshold: list[ScoredJob] = []
    for job in hydrated:
        result = scorer.score(job)
        if result is None:
            log.warning("skipping %s this run (scoring failed; will retry next run)", job.job_id)
            continue
        scored = ScoredJob(job=job, match=result)
        log.info("%3d%%  %s @ %s", result.match_score_percentage, job.title, job.company)
        if result.match_score_percentage >= settings.match_threshold:
            matches.append(scored)
        else:
            below_threshold.append(scored)

    # Phase 4: delivery
    delivered = notifier.send(matches) if matches else []
    log.info("%d/%d matches delivered to Discord", len(delivered), len(matches))

    # Persist only what is final: delivered matches and rejected jobs.
    # Undelivered matches stay "new" so the next run retries them.
    if settings.dry_run:
        log.info("dry run: state not updated")
    else:
        for scored in delivered:
            state.mark_processed(scored.job, scored.match.match_score_percentage, notified=True)
        for scored in below_threshold:
            state.mark_processed(scored.job, scored.match.match_score_percentage, notified=False)

    return 1 if len(delivered) < len(matches) else 0


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        stream=sys.stdout,
    )
    for noisy in ("httpx", "httpx2", "openai", "botocore", "urllib3", "LiteLLM", "litellm"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    try:
        sys.exit(run(Settings()))
    except ValueError as exc:
        log.error("configuration error: %s", exc)
        sys.exit(2)
    except ScorerQuotaError as exc:
        log.error(
            "Stopping: %s. Nothing was lost; unscored jobs are not saved to state, so the next run "
            "picks them up. Groq's daily limits roll over within 24 hours; upgrading to the Groq "
            "Developer tier raises them.",
            str(exc).split(":", 1)[0],
        )
        sys.exit(2)
    except ScorerAuthError as exc:
        log.error("LLM scoring is misconfigured (check GROQ_API_KEY / GROQ_MODEL): %s", exc)
        sys.exit(2)


if __name__ == "__main__":
    main()
