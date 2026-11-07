"""Offline unit tests: no network, AWS, LLM or Discord calls."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from job_aggregator.discord_notifier import _chunk, build_embed
from job_aggregator.extractor import NoFluffJobsExtractor, ProfessionHuExtractor, html_to_text
from job_aggregator.models import JobPosting, MatchResult, ScoredJob
from job_aggregator.state import LocalJsonStateStore

RSS = b"""<?xml version="1.0"?>
<rss version="2.0"><channel><title>t</title>
<item>
  <title>Junior DevOps Engineer</title>
  <link>https://www.profession.hu/allas/junior-devops-engineer-acme-kft-budapest-3040001?utm_source=rss </link>
  <guid>https://www.profession.hu/allas/junior-devops-engineer-acme-kft-budapest-3040001</guid>
  <description>["Maintain CI/CD pipelines","Work with Kubernetes \\u00e9s AWS"]</description>
</item>
<item>
  <title>Senior DevOps Engineer</title>
  <guid>https://www.profession.hu/allas/senior-devops-engineer-foo-3040002</guid>
  <description>[]</description>
</item>
<item>
  <title>Junior Sales Representative</title>
  <guid>https://www.profession.hu/allas/junior-sales-representative-bar-3040003</guid>
  <description>[]</description>
</item>
</channel></rss>"""


def make_job(**overrides) -> JobPosting:
    data = dict(
        job_id="nofluffjobs:x",
        title="Junior Data Engineer",
        company="Acme",
        url="https://nofluffjobs.com/hu/job/x",
        description="Python, SQL",
        source="nofluffjobs",
    )
    data.update(overrides)
    return JobPosting(**data)


def test_profession_feed_filters_seniority_and_domain():
    jobs = ProfessionHuExtractor().parse_feed(RSS)
    assert [j.title for j in jobs] == ["Junior DevOps Engineer"]
    job = jobs[0]
    assert job.job_id == "profession.hu:3040001"
    assert job.url.endswith("3040001")
    assert job.company == "Acme Kft"
    assert "Kubernetes és AWS" in job.description


def test_nofluffjobs_listing_normalization():
    raw = {
        "id": "junior-devops-acme-Budapest",
        "title": "Junior DevOps",
        "name": "Acme",
        "url": "junior-devops-acme-budapest",
        "category": "devops",
        "seniority": ["Junior"],
        "tiles": {"values": [{"value": "AWS", "type": "requirement"}]},
    }
    job = NoFluffJobsExtractor()._normalize_listing(raw)
    assert job is not None
    assert job.job_id == "nofluffjobs:junior-devops-acme-Budapest"
    assert job.url == "https://nofluffjobs.com/hu/job/junior-devops-acme-budapest"
    assert "AWS" in job.description


def test_html_to_text():
    assert html_to_text("<ul><li>Python &amp; SQL</li><li>Docker</li></ul>") == "Python & SQL\nDocker"


def test_job_posting_rejects_relative_url():
    with pytest.raises(ValidationError):
        make_job(url="/hu/job/x")


def test_match_result_clamps_and_dedupes():
    result = MatchResult(
        match_score_percentage=140,
        missing_skills=["Azure", "azure", " ", "Spark"],
        intro_sentence_for_application="Hi",
    )
    assert result.match_score_percentage == 100
    assert result.missing_skills == ["Azure", "Spark"]


def test_embed_and_chunking():
    scored = ScoredJob(
        job=make_job(),
        match=MatchResult(match_score_percentage=80, missing_skills=[], intro_sentence_for_application="I..."),
    )
    embed = build_embed(scored)
    assert embed["url"] == scored.job.url
    assert any(f["value"] == "**80%**" for f in embed["fields"])
    assert [len(b) for b in _chunk([embed] * 23)] == [10, 10, 3]


def test_local_state_store_roundtrip(tmp_path):
    path = tmp_path / "state.json"
    store = LocalJsonStateStore(str(path))
    job = make_job()
    assert store.filter_new([job.job_id, "other"]) == {job.job_id, "other"}
    store.mark_processed(job, 70, notified=True)
    assert LocalJsonStateStore(str(path)).filter_new([job.job_id, "other"]) == {"other"}


# ---------------------------------------------------------------------------
# LLM scorer (Groq) with a fake client: no network
# ---------------------------------------------------------------------------
import httpx  # noqa: E402
import openai  # noqa: E402

from job_aggregator import ai_scorer  # noqa: E402


def _completion(content: str | None):
    message = type("M", (), {"content": content})()
    choice = type("C", (), {"message": message, "finish_reason": "stop"})()
    return type("R", (), {"choices": [choice]})()


def _status_error(cls, status: int, headers: dict | None = None):
    request = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
    response = httpx.Response(status, request=request, headers=headers or {})
    return cls("boom", response=response, body=None)


@pytest.fixture
def scorer(monkeypatch):
    monkeypatch.setattr(ai_scorer.time, "sleep", lambda _s: None)
    return ai_scorer.LLMScorer("test-key", "profile", min_seconds_between_calls=0, max_retries=3)


def _script(scorer, *outcomes):
    calls = []

    def fake(_prompt):
        outcome = outcomes[len(calls)]
        calls.append(1)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome

    scorer._complete = fake
    return calls


def test_scorer_parses_fenced_json(scorer):
    _script(scorer, _completion('```json\n{"match_score_percentage": 77, "missing_skills": ["Azure"], '
                                '"intro_sentence_for_application": "Hi"}\n```'))
    result = scorer.score(make_job())
    assert result.match_score_percentage == 77 and result.missing_skills == ["Azure"]


def test_scorer_retries_rate_limit_then_invalid_json_then_succeeds(scorer):
    calls = _script(
        scorer,
        _status_error(openai.RateLimitError, 429, {"retry-after": "2"}),
        _completion('{"oops": true}'),
        _completion('{"match_score_percentage": 66, "missing_skills": [], "intro_sentence_for_application": "x"}'),
    )
    assert scorer.score(make_job()).match_score_percentage == 66
    assert len(calls) == 3


def test_scorer_gives_up_on_repeated_timeouts(scorer):
    request = httpx.Request("POST", "https://api.groq.com")
    calls = _script(scorer, *[openai.APITimeoutError(request=request)] * 3)
    assert scorer.score(make_job()) is None
    assert len(calls) == 3


@pytest.mark.parametrize("cls,status", [(openai.AuthenticationError, 401), (openai.NotFoundError, 404)])
def test_scorer_aborts_run_on_config_errors(scorer, cls, status):
    _script(scorer, _status_error(cls, status))
    with pytest.raises(ai_scorer.ScorerAuthError):
        scorer.score(make_job())


def test_scorer_hard_timeout(scorer, monkeypatch):
    import threading

    monkeypatch.setattr(ai_scorer, "CALL_TIMEOUT_SECONDS", 0.2)
    scorer.max_retries = 2
    calls = []
    release = threading.Event()

    def hang(_prompt):  # blocks for real (time.sleep is patched out by the fixture)
        calls.append(1)
        release.wait(5)
        return _completion('{"match_score_percentage": 99, "missing_skills": [], "intro_sentence_for_application": "x"}')

    scorer._complete = hang
    try:
        assert scorer.score(make_job()) is None
        assert len(calls) == 2
    finally:
        release.set()


# ---------------------------------------------------------------------------
# Career-page crawler (pure helpers; no browser or LLM)
# ---------------------------------------------------------------------------
import json as _json  # noqa: E402

from job_aggregator.ats_crawler import (  # noqa: E402
    SENIOR_TITLE_RE,
    _raise_if_fatal,
    canonical_url,
    company_from_url,
    job_id_for,
    parse_extracted,
)


def test_canonical_url_and_job_id_are_stable():
    a = "https://Careers.EPAM.com/en/vacancy/devops-intern-123_en/?utm_source=x&b=2&a=1#apply"
    b = "https://careers.epam.com/en/vacancy/devops-intern-123_en?a=1&b=2"
    assert canonical_url(a) == canonical_url(b)
    assert job_id_for(a) == job_id_for(b)
    assert job_id_for(a).startswith("careers:")


@pytest.mark.parametrize(
    "url,company",
    [
        ("https://careers.epam.com/en/jobs/hungary/budapest", "EPAM"),
        ("https://jobs.smartrecruiters.com/BoschGroup/?city=Budapest", "Bosch"),
        ("https://telekom.jobs/search?location=Budapest", "Deutsche Telekom"),
        ("https://careers.example.org/jobs", "Example"),
    ],
)
def test_company_from_url(url, company):
    assert company_from_url(url) == company


def test_parse_extracted_drops_hallucinated_and_relative_resolved():
    page = "https://careers.epam.com/en/jobs/hungary/budapest"
    links = {
        canonical_url("https://careers.epam.com/en/vacancy/devops-intern-1_en"),
        canonical_url("https://careers.epam.com/en/vacancy/junior-python-2_en"),
    }
    raw = _json.dumps([
        {"title": "DevOps Intern", "url": "/en/vacancy/devops-intern-1_en", "location": "Budapest", "error": False},
        {"title": "Junior Python", "url": "https://careers.epam.com/en/vacancy/junior-python-2_en"},
        {"title": "Invented", "url": "https://careers.epam.com/en/vacancy/does-not-exist"},
        {"title": "Listing page", "url": page},
        {"index": 1, "error": True, "tags": ["error"], "content": "timeout"},
    ])
    jobs, errors = parse_extracted(raw, page, links)
    assert [j.title for j in jobs] == ["DevOps Intern", "Junior Python"]
    assert jobs[0].url == "https://careers.epam.com/en/vacancy/devops-intern-1_en"
    assert errors == ["timeout"]


def test_fatal_llm_errors_abort():
    from job_aggregator.ai_scorer import ScorerAuthError

    with pytest.raises(ScorerAuthError):
        _raise_if_fatal(["litellm.AuthenticationError: OpenAIException - 401 User not found"])
    _raise_if_fatal(["Request timed out"])  # transient: no exception


def test_senior_title_prefilter():
    assert SENIOR_TITLE_RE.search("Senior Cloud Engineer")
    assert SENIOR_TITLE_RE.search("Engineering Manager")
    assert not SENIOR_TITLE_RE.search("DevOps Intern")
    assert not SENIOR_TITLE_RE.search("Junior Python Developer")


@pytest.mark.parametrize(
    "url,company",
    [
        ("https://blackrock.wd1.myworkdayjobs.com/BlackRock_Professional?q=Budapest", "BlackRock"),
        ("https://www.ibm.com/careers/search?field_keyword_05[0]=Hungary", "IBM"),
        ("https://careers.msci.com/job-search?x=1", "MSCI"),
        ("https://acme.wd3.myworkdayjobs.com/External", "Acme"),
    ],
)
def test_company_from_ats_subdomain(url, company):
    assert company_from_url(url) == company


@pytest.mark.parametrize(
    "location,ok",
    [("", True), ("Budapest, Hungary", True), ("HU - Remote", True), ("Debrecen, HU", True),
     ("Košice", False), ("Walldorf, Germany", False), ("Chumash, CA", False)],
)
def test_location_prefilter(location, ok):
    from job_aggregator.ats_crawler import location_is_relevant

    assert location_is_relevant(location) is ok


# ---------------------------------------------------------------------------
# robots.txt (RFC 9309)
# ---------------------------------------------------------------------------
from job_aggregator.robots import RobotsPolicy, RobotsRules  # noqa: E402

EIGHTFOLD_ROBOTS = """User-agent: *
Disallow: /
Allow: /$
Allow: /careers
Allow: /api/apply
"""


@pytest.mark.parametrize(
    "path,ok",
    [
        ("/careers?location=Budapest%2C%20Hungary", True),
        ("/careers/job/563465374941425", True),
        ("/", True),           # "Allow: /$" beats "Disallow: /" (same length, allow wins the tie)
        ("/api/pcsx", False),
        ("/careerhub", False),
    ],
)
def test_robots_longest_match_wins(path, ok):
    assert RobotsRules.parse(EIGHTFOLD_ROBOTS).allowed("https://jobs.ericsson.com" + path) is ok


def test_robots_group_selection_and_wildcards():
    text = """User-agent: *
Disallow: /en/
Disallow: /*.pdf$

User-agent: Mindbreeze-P
Allow: /en/
"""
    rules = RobotsRules.parse(text)
    assert not rules.allowed("https://jobs.bosch.com/en/?country=hu")
    assert not rules.allowed("https://x.test/files/cv.pdf")
    assert rules.allowed("https://x.test/files/cv.pdf?download=1")
    assert rules.allowed("https://jobs.bosch.com/de/")
    assert RobotsRules.parse(text, user_agent="Mindbreeze-P").allowed("https://jobs.bosch.com/en/")


def test_robots_specific_agent_group_overrides_star():
    text = "User-agent: *\nDisallow: /\n\nUser-agent: budapest-job-aggregator\nDisallow: /private\n"
    rules = RobotsRules.parse(text)
    assert rules.allowed("https://x.test/careers")
    assert not rules.allowed("https://x.test/private/a")


def test_robots_empty_disallow_allows_everything():
    assert RobotsRules.parse("User-agent: *\nDisallow:\n").allowed("https://x.test/anything")


class _FakeResponse:
    def __init__(self, status, body=b""):
        self.status_code, self.content = status, body


class _FakeSession:
    def __init__(self, response):
        self.response, self.calls = response, 0

    def get(self, *_a, **_k):
        self.calls += 1
        if isinstance(self.response, Exception):
            raise self.response
        return self.response


def test_robots_policy_http_status_handling_and_cache():
    import requests as _requests

    missing = RobotsPolicy(session=_FakeSession(_FakeResponse(404)))
    assert missing.allowed("https://a.test/x")
    erroring = RobotsPolicy(session=_FakeSession(_FakeResponse(503)))
    assert not erroring.allowed("https://a.test/x")
    unreachable = RobotsPolicy(session=_FakeSession(_requests.ConnectionError("down")))
    assert not unreachable.allowed("https://a.test/x")

    session = _FakeSession(_FakeResponse(200, EIGHTFOLD_ROBOTS.encode()))
    cached = RobotsPolicy(session=session)
    assert cached.allowed("https://a.test/careers") and not cached.allowed("https://a.test/admin")
    assert session.calls == 1


# ---------------------------------------------------------------------------
# Eightfold adapter (fake HTTP session)
# ---------------------------------------------------------------------------
from job_aggregator.eightfold import EightfoldClient, eightfold_domain  # noqa: E402


def test_eightfold_domain_detection():
    assert eightfold_domain("https://jobs.ericsson.com/careers?location=Budapest") == "ericsson.com"
    assert eightfold_domain("https://morganstanley.eightfold.ai/careers") == "morganstanley.com"
    assert eightfold_domain("https://careers.epam.com/en/jobs") is None


class _JsonResponse:
    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self.payload


class _EightfoldSession:
    def __init__(self):
        self.urls = []

    def get(self, url, **_kw):
        self.urls.append(url)
        if "position_details" in url:
            return _JsonResponse({"data": {"jobDescription": "<p>Run <b>CI/CD</b></p>", "locations": ["Budapest"]}})
        start = int(url.split("start=")[1].split("&")[0])
        positions = [{"id": str(start + i), "name": f"DevOps Engineer {start + i}", "locations": ["Budapest,Budapest,Hungary"],
                      "positionUrl": f"/careers/job/{start + i}"} for i in range(10 if start == 0 else 2)]
        return _JsonResponse({"data": {"positions": positions, "count": 12}})


class _AllowAll:
    def allowed(self, _url):
        return True


def test_eightfold_search_paginates_and_describes():
    session = _EightfoldSession()
    client = EightfoldClient(session, _AllowAll(), delay_seconds=0)
    jobs = client.search("https://jobs.ericsson.com/careers?location=Budapest%2C%20Hungary")
    assert len(jobs) == 12 and len([u for u in session.urls if "/api/pcsx/search" in u]) == 2
    assert jobs[0]["url"] == "https://jobs.ericsson.com/careers/job/0"
    assert "location=Budapest%2C+Hungary" in session.urls[0] and "domain=ericsson.com" in session.urls[0]
    text = client.description(jobs[0]["url"])
    assert "Run CI/CD" in text and text.startswith("Location: Budapest")


def test_eightfold_respects_robots():
    class _DenyAll:
        def allowed(self, _url):
            return False

    session = _EightfoldSession()
    assert EightfoldClient(session, _DenyAll()).search("https://jobs.ericsson.com/careers?location=X") == []
    assert session.urls == []


@pytest.mark.parametrize(
    "title,department,tech",
    [
        ("DevOps Engineer", "76 Product Development", True),
        ("Model Governance & Controls Associate", "Risk, Assurance, Governance & Control", False),
        ("Resource Planner", "Finance Business Management", False),
        ("Associate", "Technology", True),
        ("People Business Partner Hungary", "87 HR", False),
    ],
)
def test_eightfold_tech_filter(title, department, tech):
    from job_aggregator.eightfold import TECH_ROLE_RE

    assert bool(TECH_ROLE_RE.search(f"{title} {department}")) is tech


# ---------------------------------------------------------------------------
# Daily quota exhaustion stops the run (no pointless retries)
# ---------------------------------------------------------------------------
def test_daily_quota_detection():
    from job_aggregator.ai_scorer import is_daily_quota_error

    assert is_daily_quota_error("Rate limit reached ... on requests per day (RPD): Limit 1000")
    assert not is_daily_quota_error("Rate limit reached ... on requests per minute (RPM): Limit 30")


def test_scorer_raises_on_daily_quota(scorer):
    calls = _script(
        scorer,
        _status_error(openai.RateLimitError, 429),
        _completion('{"match_score_percentage": 66, "missing_skills": [], "intro_sentence_for_application": "x"}'),
    )

    def quota(_prompt):
        calls.append(1)
        request = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
        response = httpx.Response(429, request=request)
        raise openai.RateLimitError("Rate limit reached ... on tokens per day (TPD): Limit 100000", response=response, body=None)

    scorer._complete = quota
    with pytest.raises(ai_scorer.ScorerQuotaError):
        scorer.score(make_job())
    assert len(calls) == 1  # no retries


def test_extraction_quota_error_aborts():
    from job_aggregator.ai_scorer import ScorerQuotaError

    with pytest.raises(ScorerQuotaError):
        _raise_if_fatal(["litellm.RateLimitError: GroqException - Rate limit reached ... on tokens per day (TPD)"])


# ---------------------------------------------------------------------------
# Groq configuration and error handling
# ---------------------------------------------------------------------------
def test_config_requires_groq_key(monkeypatch):
    from job_aggregator.config import DEFAULT_GROQ_MODEL, Settings

    monkeypatch.setenv("GROQ_API_KEY", "")
    monkeypatch.delenv("GROQ_MODEL", raising=False)
    monkeypatch.delenv("LLM_MIN_SECONDS_BETWEEN_CALLS", raising=False)
    settings = Settings()
    assert settings.groq_model == DEFAULT_GROQ_MODEL  # whatever config.py defines
    assert settings.llm_min_seconds_between_calls == 2.0
    with pytest.raises(ValueError, match="GROQ_API_KEY"):
        settings.validate()


def test_groq_model_env_override(monkeypatch):
    from job_aggregator.config import Settings

    monkeypatch.setenv("GROQ_MODEL", "llama-3.1-8b-instant")
    assert Settings().groq_model == "llama-3.1-8b-instant"
    scorer = ai_scorer.LLMScorer("k", "p")
    assert scorer.model == "llama-3.1-8b-instant"
    assert str(scorer.client.base_url).rstrip("/") == "https://api.groq.com/openai/v1"
    assert scorer.min_interval == 2.0


def test_scorer_retries_groq_json_validate_failed(scorer):
    calls = _script(
        scorer,
        openai.BadRequestError(
            "Failed to generate JSON. code: json_validate_failed",
            response=httpx.Response(400, request=httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")),
            body=None,
        ),
        _completion('{"match_score_percentage": 80, "missing_skills": [], "intro_sentence_for_application": "x"}'),
    )
    assert scorer.score(make_job()).match_score_percentage == 80
    assert len(calls) == 2


def test_scorer_other_400_skips_job(scorer):
    _script(scorer, _status_error(openai.BadRequestError, 400))
    assert scorer.score(make_job()) is None


def test_groq_daily_quota_message_detected():
    from job_aggregator.ai_scorer import is_daily_quota_error

    assert is_daily_quota_error(
        "Rate limit reached for model `llama-3.3-70b-versatile` in organization `org_x` service tier "
        "`on_demand` on tokens per day (TPD): Limit 100000, Used 99500, Requested 1200."
    )
    assert is_daily_quota_error("... on requests per day (RPD): Limit 1000, Used 1000, Requested 1.")
    assert not is_daily_quota_error(
        "Rate limit reached for model `llama-3.3-70b-versatile` ... on tokens per minute (TPM): Limit 12000"
    )


def test_extraction_strategy_uses_groq_provider(monkeypatch):
    from job_aggregator.ats_crawler import build_llm_strategy

    monkeypatch.setenv("GROQ_API_KEY", "env-key")
    monkeypatch.setenv("GROQ_MODEL", "llama-3.3-70b-versatile")
    strategy = build_llm_strategy()
    assert strategy.llm_config.provider == "groq/llama-3.3-70b-versatile"
    assert strategy.llm_config.api_token == "env-key"


def test_extraction_requests_fit_groq_free_tier(monkeypatch):
    """input chunk + output budget must stay well under the smallest free-tier TPM (8,000)."""
    from job_aggregator.ats_crawler import build_llm_strategy

    monkeypatch.setenv("GROQ_API_KEY", "k")
    strategy = build_llm_strategy(model="openai/gpt-oss-120b")
    assert strategy.chunk_token_threshold + strategy.extra_args["max_tokens"] <= 6000
    assert strategy.extra_args["reasoning_effort"] == "low"
    assert strategy.llm_config.backoff_base_delay >= 15
    assert "reasoning_effort" not in build_llm_strategy(model="llama-3.3-70b-versatile").extra_args
