"""Processed-job state: DynamoDB in production, a JSON file for local testing."""

from __future__ import annotations

import json
import logging
import time
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from pathlib import Path

import boto3
from botocore.exceptions import ClientError

from .models import JobPosting

log = logging.getLogger(__name__)


class StateStore(ABC):
    @abstractmethod
    def filter_new(self, job_ids: list[str]) -> set[str]:
        """Return the subset of `job_ids` that have NOT been processed yet."""

    @abstractmethod
    def mark_processed(self, job: JobPosting, score: int | None, notified: bool) -> None:
        """Record a job so it is never scored or alerted again."""


class DynamoDBStateStore(StateStore):
    """Table `job_scraper_state`, partition key `job_id` (S), TTL attribute `expires_at`."""

    BATCH_GET_LIMIT = 100

    def __init__(self, table_name: str, region: str, ttl_days: int = 180) -> None:
        self.table_name = table_name
        self.ttl_days = ttl_days
        self.resource = boto3.resource("dynamodb", region_name=region)
        self.table = self.resource.Table(table_name)

    def filter_new(self, job_ids: list[str]) -> set[str]:
        unique_ids = list(dict.fromkeys(job_ids))
        seen: set[str] = set()

        for start in range(0, len(unique_ids), self.BATCH_GET_LIMIT):
            chunk = unique_ids[start : start + self.BATCH_GET_LIMIT]
            request = {
                self.table_name: {
                    "Keys": [{"job_id": job_id} for job_id in chunk],
                    "ProjectionExpression": "job_id",
                }
            }
            attempt = 0
            while request:
                response = self.resource.batch_get_item(RequestItems=request)
                for item in response.get("Responses", {}).get(self.table_name, []):
                    seen.add(item["job_id"])
                request = response.get("UnprocessedKeys") or {}
                if request:
                    attempt += 1
                    if attempt > 8:
                        raise RuntimeError("DynamoDB batch_get_item kept returning UnprocessedKeys")
                    time.sleep(min(2**attempt * 0.1, 5))

        return set(unique_ids) - seen

    def mark_processed(self, job: JobPosting, score: int | None, notified: bool) -> None:
        now = int(time.time())
        item = {
            "job_id": job.job_id,
            "source": job.source,
            "title": job.title[:500],
            "company": job.company[:200],
            "url": job.url,
            "notified": notified,
            "processed_at": datetime.now(timezone.utc).isoformat(),
            "expires_at": now + self.ttl_days * 86400,
        }
        if score is not None:
            item["match_score"] = score
        try:
            self.table.put_item(Item=item, ConditionExpression="attribute_not_exists(job_id)")
        except ClientError as exc:
            if exc.response["Error"]["Code"] == "ConditionalCheckFailedException":
                log.info("state: %s was already recorded by a concurrent run", job.job_id)
                return
            raise


class LocalJsonStateStore(StateStore):
    """File-based store with the same semantics, for local development."""

    def __init__(self, path: str) -> None:
        self.path = Path(path)
        self._data: dict[str, dict] = {}
        if self.path.exists():
            self._data = json.loads(self.path.read_text(encoding="utf-8"))

    def filter_new(self, job_ids: list[str]) -> set[str]:
        return {job_id for job_id in job_ids if job_id not in self._data}

    def mark_processed(self, job: JobPosting, score: int | None, notified: bool) -> None:
        self._data[job.job_id] = {
            "title": job.title,
            "company": job.company,
            "url": job.url,
            "match_score": score,
            "notified": notified,
            "processed_at": datetime.now(timezone.utc).isoformat(),
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self._data, indent=2, ensure_ascii=False), encoding="utf-8")
        tmp.replace(self.path)


def build_state_store(backend: str, *, table: str, region: str, local_path: str, ttl_days: int) -> StateStore:
    if backend == "dynamodb":
        return DynamoDBStateStore(table, region, ttl_days)
    return LocalJsonStateStore(local_path)
