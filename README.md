# Budapest Job Aggregator & AI Scorer

A serverless ETL pipeline that runs every 8 hours on GitHub Actions. It:

1. Crawls company career pages (`CAREER_PAGE_URLS` in `config.py`) with **crawl4ai** in headless Chromium. An LLM on Groq extracts the tech job titles and links from each page.
2. Normalizes each role into a Pydantic `JobPosting`, and drops clearly senior or manager titles before any further work.
3. Skips roles already stored in DynamoDB (`job_scraper_state`), then fetches the job page **only for new roles**.
4. Scores new roles against your profile with an LLM on **Groq** (JSON mode).
5. Posts roles scoring 65% or more to **Discord** as embeds.

```
budapest-job-aggregator/
├── .github/workflows/scraper.yml   # cron every 8h + manual dispatch (with dry-run toggle)
├── infra/main.tf                   # DynamoDB (PAY_PER_REQUEST, TTL) + GitHub OIDC IAM role
├── job_aggregator/
│   ├── config.py                   # env-driven settings + candidate profile
│   ├── models.py                   # JobPosting, MatchResult, ScoredJob
│   ├── ats_crawler.py              # career pages: crawl4ai + LLM link extraction (source "careers")
│   ├── extractor.py                # base class + NoFluffJobs API / Profession.hu RSS (optional sources)
│   ├── state.py                    # DynamoDB store (+ local JSON store for testing)
│   ├── ai_scorer.py                # Groq scoring (openai SDK, JSON mode)
│   ├── discord_notifier.py         # embeds, batching, rate-limit handling
│   └── main.py                     # orchestration
├── tests/test_pipeline.py          # offline unit tests
├── requirements.txt / requirements-dev.txt
└── .env.example
```

## How a job flows through the pipeline

- **Job IDs include the source**, for example `careers:<hash of the job URL>`, `nofluffjobs:<slug>` or `profession.hu:<numeric id>`, so IDs never collide.
  - Career-page URLs are normalized before hashing (lowercased host, no tracking parameters or `#fragment`), so the same job keeps the same ID.
- **The full description is fetched only for new jobs**, in one browser session per run.
- **State is written only for finished jobs:**
  - Jobs that were delivered to Discord, or that scored below the threshold, are recorded.
  - If a Discord post fails or scoring fails, the job stays new and is retried on the next run.
- **Records expire after 180 days** through a DynamoDB TTL on `expires_at`, so the table stays inside the free tier.
- **Limits:** `MAX_JOBS_PER_RUN` caps LLM usage per run, and `LLM_MIN_SECONDS_BETWEEN_CALLS` spaces calls out to respect free-tier rate limits.

## Career-page crawling

- **Two steps:**
  1. Each listing page is rendered, and the LLM extracts job titles and URLs. Pages are crawled one at a time to keep the LLM request rate low.
  2. For new jobs only, each job page is rendered and its main content becomes the description. This step uses no LLM.
- **Eightfold-hosted sites (Ericsson, Morgan Stanley) skip the browser and the LLM.** Their pages render job cards without real links, so [`eightfold.py`](job_aggregator/eightfold.py) reads the JSON search and job-details endpoints that the site itself uses. robots.txt explicitly allows them (`Allow: /api/pcsx`).
  - Because that search returns every role, a title/department filter keeps only tech roles before scoring.
- **Invented URLs are discarded:** a URL the LLM returns is kept only if it actually appears among the page's links.
- **robots.txt is respected, following the standard (RFC 9309).** [`robots.py`](job_aggregator/robots.py) checks every listing and job page before it's opened. Disallowed pages are skipped and logged.
  - The most specific (longest) matching rule wins, and an `Allow` wins a tie. crawl4ai's built-in check applies the first matching rule instead, which wrongly blocks sites such as Ericsson's (`Disallow: /` followed by `Allow: /careers`).
  - If robots.txt is missing (HTTP 4xx), everything is allowed. If the server errors or can't be reached, the whole site is treated as disallowed.
  - `config.py` lists the companies that are left out and why: Nokia and Bosch.
- **Cheap pre-filters run before any further work:** clearly senior or manager titles, and listed locations outside Budapest or Hungary, are skipped. SAP's and Telekom's lists aren't limited to Budapest.
- **Pages that never finish loading** (IBM, for example) are retried with a lighter "page loaded" wait instead of waiting for the network to go quiet.
- **One broken site doesn't stop the run.** A page that errors or returns HTTP 4xx is logged and skipped.
- **Only the first page of results is read.** Most sites show 10–25 jobs per page, and pagination isn't followed.
- **Free-tier usage:** each run makes about 1–2 LLM calls per LLM-extracted career page, plus one call per new job scored. Groq's free tier caps requests and tokens per minute and per day for each model. The current numbers are on Groq's rate-limits page.
  - `LLM_MIN_SECONDS_BETWEEN_CALLS` (default 2 s) keeps scoring at the 30-requests-per-minute limit, and `MAX_JOBS_PER_RUN` caps scoring calls per run.
  - A scoring call uses roughly 2,000–4,000 tokens, so the per-minute token limit is usually hit before the request limit. Those 429s are retried after the wait Groq asks for.
  - If a daily cap runs out, the run stops cleanly and the remaining jobs wait for the next run.
- **Heavier CI:** this needs Playwright's Chromium, which the workflow installs and caches.
- **Job boards still work:** add `SOURCES=careers,nofluffjobs,profession` to also pull from NoFluffJobs and Profession.hu.

## LLM scoring and extraction (Groq)

- **Scoring:** uses the official `openai` Python SDK pointed at Groq's OpenAI compatibility endpoint (`https://api.groq.com/openai/v1`).
- **Link extraction:** crawl4ai calls Groq through litellm, using the provider string `groq/<model>`.
- **Model:** the default is `openai/gpt-oss-120b` (`DEFAULT_GROQ_MODEL` in `config.py`), used for both scoring and link extraction. To use another model, set `GROQ_MODEL`. The models your key can use are listed at `https://api.groq.com/openai/v1/models`.
- **Output:** scoring requests use JSON mode, and every response is checked against `MatchResult`. Invalid output, including Groq's `json_validate_failed` error, is retried.
- **Errors that stop the run:** a missing or invalid `GROQ_API_KEY`, an unknown model, or an exhausted daily cap ends the run with exit code `2`, instead of failing every job.

## Infrastructure

### Option A: Terraform (recommended)

This creates the table and an IAM role that GitHub Actions can assume through OIDC, so no AWS keys are stored in GitHub.

```bash
cd infra
terraform init
terraform apply -var="github_repository=<your-github-user>/budapest-job-aggregator"
```

If your AWS account already has the GitHub OIDC provider, add `-var="create_github_oidc_provider=false"`.

Copy the `github_actions_role_arn` output into a GitHub secret named `AWS_ROLE_ARN`.

### Option B: AWS CLI (table only)

```bash
aws dynamodb create-table --table-name job_scraper_state --attribute-definitions AttributeName=job_id,AttributeType=S --key-schema AttributeName=job_id,KeyType=HASH --billing-mode PAY_PER_REQUEST --region eu-central-1
```

```bash
aws dynamodb update-time-to-live --table-name job_scraper_state --time-to-live-specification "Enabled=true,AttributeName=expires_at" --region eu-central-1
```

If you use Option B, GitHub Actions needs credentials:
1. Create an IAM role or user allowed to run `dynamodb:BatchGetItem`, `dynamodb:GetItem` and `dynamodb:PutItem` on the table.
2. Either set up OIDC for it, or replace the `configure-aws-credentials` step with `aws-access-key-id` / `aws-secret-access-key` secrets.

## GitHub setup

| Kind     | Name                  | Value                                   |
|----------|-----------------------|-----------------------------------------|
| Secret   | `GROQ_API_KEY`        | From https://console.groq.com/keys      |
| Secret   | `DISCORD_WEBHOOK_URL` | Channel → Integrations → Webhooks       |
| Secret   | `AWS_ROLE_ARN`        | Terraform output                        |
| Variable | `AWS_REGION`          | Optional, default `eu-central-1`        |
| Variable | `GROQ_MODEL`          | Optional, default `openai/gpt-oss-120b` |

To trigger a run by hand, go to **Actions → Budapest job aggregator → Run workflow**. Tick `dry_run` to see the Discord payloads in the log instead of posting them.

## Running locally (Windows PowerShell)

```powershell
cd D:\Projects\budapest-job-aggregator
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements-dev.txt
python -m playwright install chromium   # headless browser for crawl4ai
pytest -q
Copy-Item .env.example .env   # then edit .env: set GROQ_API_KEY
```

**1. Dry run.** This uses the local JSON state, prints Discord payloads instead of posting them, and writes no state. In `.env`, set `STATE_BACKEND=local` and `DRY_RUN=true`, then run:

```powershell
python -m job_aggregator.main
```

**2. Real Discord alerts with local state.** In `.env`, set `DRY_RUN=false` and `DISCORD_WEBHOOK_URL=...`. Then run:

```powershell
python -m job_aggregator.main
```

To start over, delete `.state\processed_jobs.json`.

**3. Against real DynamoDB.** In `.env`, set `STATE_BACKEND=dynamodb` and `AWS_REGION`. Configure your AWS credentials (`aws configure` or `AWS_PROFILE`), then run:

```powershell
python -m job_aggregator.main
```

On macOS or Linux, activate the environment with `source .venv/bin/activate` instead.

## Customizing

- **Use your full CV:** save it as `cv.md` and set `CANDIDATE_PROFILE_FILE=cv.md`. This replaces the built-in profile in the prompt.
- **Change the threshold:** set `MATCH_THRESHOLD`.
- **Change what is searched:**
  - NoFluffJobs categories: `NoFluffJobsExtractor.DEFAULT_CATEGORIES`
  - Profession.hu search keywords: `ProfessionHuExtractor.DEFAULT_KEYWORDS`
  - Title filters: `SENIORITY_RE` and `DOMAIN_RE` in `extractor.py`
- **Change which companies are crawled:** edit `CAREER_PAGE_URLS` in `config.py`, or set `CAREER_PAGE_URLS=url1,url2` in `.env`. Use a page that lists jobs already filtered to Budapest.
- **Choose sources:** `SOURCES=careers` (the default), or any mix of `careers`, `nofluffjobs` and `profession`.

## Exit codes

| Code | Meaning |
|------|---------|
| `0`  | Success |
| `1`  | Some matching jobs could not be delivered to Discord. They are retried on the next run. |
| `2`  | Configuration or auth error, or every source failed. |

Codes `1` and `2` make the GitHub Actions run show as failed, so you get notified.
