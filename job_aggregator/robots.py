"""robots.txt evaluation per RFC 9309 (the Robots Exclusion Protocol standard).

Why not crawl4ai's built-in check: it applies the FIRST matching rule, so a file like

    User-agent: *
    Disallow: /
    Allow: /careers

blocks /careers. RFC 9309 section 2.2.2 says the MOST SPECIFIC (longest) matching rule wins,
and an allow wins a tie, so /careers is allowed. This module implements the standard:

- Group selection: the group whose user-agent token matches ours (case-insensitive), else "*".
  Multiple groups for the same agent are merged.
- Rules support "*" (any sequence) and "$" (end of URL) and are matched against path + query.
- Missing robots.txt (HTTP 4xx): everything allowed. Server error or unreachable (5xx, network):
  everything disallowed, as the RFC requires.
"""

from __future__ import annotations

import logging
import re
import threading
from dataclasses import dataclass, field
from urllib.parse import unquote, urlsplit

import requests

log = logging.getLogger(__name__)

ROBOTS_USER_AGENT_TOKEN = "budapest-job-aggregator"
_FETCH_HEADERS = {"User-Agent": f"{ROBOTS_USER_AGENT_TOKEN}/1.0 (+personal job alert pipeline)"}
MAX_ROBOTS_BYTES = 500 * 1024  # RFC 9309: parse at least 500 KiB


@dataclass
class _Rule:
    allow: bool
    pattern: str
    regex: re.Pattern = field(repr=False)


def _compile(pattern: str) -> re.Pattern:
    anchored = pattern.endswith("$")
    body = pattern[:-1] if anchored else pattern
    regex = ".*".join(re.escape(part) for part in body.split("*"))
    return re.compile(regex + ("$" if anchored else ""))


def _normalize(path: str) -> str:
    # Compare decoded forms so "/careers%2Fjob" and "/careers/job" are treated consistently.
    return unquote(path)


class RobotsRules:
    """Parsed rules for one user agent."""

    def __init__(self, rules: list[_Rule] | None = None, allow_all: bool = False, disallow_all: bool = False):
        self.rules = rules or []
        self.allow_all = allow_all
        self.disallow_all = disallow_all

    @classmethod
    def parse(cls, text: str, user_agent: str = ROBOTS_USER_AGENT_TOKEN) -> "RobotsRules":
        groups: list[tuple[list[str], list[_Rule]]] = []
        agents: list[str] = []
        rules: list[_Rule] = []
        in_rules = False

        for raw_line in text.splitlines():
            line = raw_line.split("#", 1)[0].strip()
            if ":" not in line:
                continue
            key, value = (part.strip() for part in line.split(":", 1))
            key = key.lower()
            if key == "user-agent":
                if in_rules:  # a user-agent line after rules starts a new group
                    groups.append((agents, rules))
                    agents, rules, in_rules = [], [], False
                agents.append(value.lower())
            elif key in ("allow", "disallow"):
                if not agents:
                    continue  # rules before any user-agent line are ignored
                in_rules = True
                if value == "":
                    continue  # "Disallow:" with no path matches nothing
                rules.append(_Rule(allow=key == "allow", pattern=value, regex=_compile(_normalize(value))))
        if agents:
            groups.append((agents, rules))

        token = user_agent.lower()
        specific = [r for agent_list, group in groups if token in agent_list for r in group]
        if any(token in agent_list for agent_list, _ in groups):
            return cls(specific)
        return cls([r for agent_list, group in groups if "*" in agent_list for r in group])

    def allowed(self, url: str) -> bool:
        if self.disallow_all:
            return False
        if self.allow_all:
            return True
        parts = urlsplit(url)
        target = _normalize((parts.path or "/") + (f"?{parts.query}" if parts.query else ""))
        best: _Rule | None = None
        for rule in self.rules:
            if not rule.regex.match(target):
                continue
            # Longest pattern wins; on equal length, allow wins (RFC 9309 section 2.2.2).
            if best is None or len(rule.pattern) > len(best.pattern) or (
                len(rule.pattern) == len(best.pattern) and rule.allow and not best.allow
            ):
                best = rule
        return True if best is None else best.allow


class RobotsPolicy:
    """Fetches and caches robots.txt per origin for the lifetime of one pipeline run."""

    def __init__(self, timeout: float = 15, session: requests.Session | None = None) -> None:
        self.timeout = timeout
        self.session = session or requests.Session()
        self._cache: dict[str, RobotsRules] = {}
        self._lock = threading.Lock()

    def _fetch(self, origin: str) -> RobotsRules:
        url = f"{origin}/robots.txt"
        try:
            response = self.session.get(url, headers=_FETCH_HEADERS, timeout=self.timeout, allow_redirects=True)
        except requests.RequestException as exc:
            log.warning("robots: %s unreachable (%s); treating site as disallowed", url, exc)
            return RobotsRules(disallow_all=True)
        if 400 <= response.status_code < 500:
            return RobotsRules(allow_all=True)
        if response.status_code >= 500:
            log.warning("robots: %s returned %s; treating site as disallowed", url, response.status_code)
            return RobotsRules(disallow_all=True)
        return RobotsRules.parse(response.content[:MAX_ROBOTS_BYTES].decode("utf-8", errors="replace"))

    def allowed(self, url: str) -> bool:
        parts = urlsplit(url)
        origin = f"{parts.scheme}://{parts.netloc}".lower()
        with self._lock:
            rules = self._cache.get(origin)
        if rules is None:
            rules = self._fetch(origin)
            with self._lock:
                self._cache[origin] = rules
        return rules.allowed(url)
