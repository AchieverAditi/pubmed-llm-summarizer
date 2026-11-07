#!/usr/bin/env python3
"""
pubmed_summarizer.py

Cleans messy PubMed-style JSON, asks a local Llama 3.3 70B (served by vLLM,
OpenAI-compatible API) to summarize each study's clinical outcome in one
sentence, and survives timeouts / dropped connections.

Usage:
    python pubmed_summarizer.py                    # uses the embedded sample data
    python pubmed_summarizer.py --input data.json  # your own file
    python pubmed_summarizer.py --dry-run          # clean + build payloads, no HTTP
"""
from __future__ import annotations

import argparse
import json
import logging
import random
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Optional

import requests

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
API_URL = "http://192.168.1.100:8000/v1/chat/completions"
MODEL_NAME = "llama-3.3-70b"  # must match --served-model-name on the vLLM server
PROMPT = "Summarize the clinical outcome of this study in one sentence."

CONNECT_TIMEOUT_S = 5      # LAN: if we can't connect in 5s, the server is down
READ_TIMEOUT_S = 120       # a 70B model on a DGX Spark can be slow; 45s must NOT fail
MAX_RETRIES = 3
BACKOFF_BASE_S = 2.0
RETRYABLE_STATUS = {429, 500, 502, 503, 504}

log = logging.getLogger("pubmed")

SAMPLE_DATA = """
[
  {"id": "PMID:349281",
   "ArticleTitle": "Efficacy of Drug X in severe cardiac failure.",
   "abstract_text": "We tested 500 patients. The drug worked in 40% of cases but caused nausea.",
   "authorList": ["Smith, J", "Doe, A"]},
  {"id": 8829102,
   "title": null,
   "Abstract": "Null hypothesis rejected. No significant improvement noted in cohort B vs placebo.",
   "authors": "Unknown"},
  {"id": "ERR_NOT_FOUND",
   "ArticleTitle": "Corrupted Entry",
   "abstract_text": null}
]
"""

# --------------------------------------------------------------------------- #
# Cleaning
# --------------------------------------------------------------------------- #
# The legacy DB uses different key names per record, so we try each alias.
ID_KEYS = ("id", "pmid", "PMID")
TITLE_KEYS = ("ArticleTitle", "title", "Title")
ABSTRACT_KEYS = ("abstract_text", "Abstract", "abstract", "AbstractText")
AUTHOR_KEYS = ("authorList", "authors", "Authors")

PLACEHOLDER_STRINGS = {"", "null", "none", "n/a", "na", "unknown", "undefined"}


@dataclass
class Article:
    id: str
    title: str
    abstract: str
    authors: list[str] = field(default_factory=list)


def _clean_str(value: Any) -> Optional[str]:
    """Return a stripped string, or None for null / empty / placeholder values."""
    if value is None:
        return None
    text = str(value).strip()
    return None if text.lower() in PLACEHOLDER_STRINGS else text


def _first(record: dict, keys: tuple[str, ...]) -> Any:
    for k in keys:
        if k in record and record[k] is not None:
            return record[k]
    return None


def _clean_authors(raw: Any) -> list[str]:
    if isinstance(raw, list):
        return [a for a in (_clean_str(x) for x in raw) if a]
    single = _clean_str(raw)  # "Unknown" -> None -> []
    return [single] if single else []


def clean_record(record: Any, index: int) -> Optional[Article]:
    """Return a normalized Article, or None if the record is unusable."""
    if not isinstance(record, dict):
        log.warning("Record %d skipped: not a JSON object (%s)", index, type(record).__name__)
        return None

    raw_id = _clean_str(_first(record, ID_KEYS))
    if raw_id is None or raw_id.upper().startswith("ERR"):
        log.warning("Record %d skipped: missing or error ID (%r)", index, raw_id)
        return None

    abstract = _clean_str(_first(record, ABSTRACT_KEYS))
    if abstract is None:
        # Nothing to summarize -> never waste a 70B inference call on it.
        log.warning("Record %s skipped: no abstract", raw_id)
        return None

    title = _clean_str(_first(record, TITLE_KEYS)) or "Untitled"
    authors = _clean_authors(_first(record, AUTHOR_KEYS))
    return Article(id=raw_id, title=title, abstract=abstract, authors=authors)


def load_and_clean(raw_text: str) -> list[Article]:
    try:
        data = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        log.error("Input is not valid JSON: %s", exc)
        return []
    if isinstance(data, dict):  # tolerate a single object
        data = [data]
    if not isinstance(data, list):
        log.error("Expected a JSON array, got %s", type(data).__name__)
        return []

    articles = [a for i, r in enumerate(data) if (a := clean_record(r, i))]
    log.info("Cleaned %d/%d records", len(articles), len(data))
    return articles


# --------------------------------------------------------------------------- #
# LLM request
# --------------------------------------------------------------------------- #
def build_payload(article: Article) -> dict:
    user_content = (
        f"{PROMPT}\n\n"
        f"Title: {article.title}\n"
        f"Abstract: {article.abstract}"
    )
    return {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system",
             "content": "You are a careful biomedical assistant. Use only the "
                        "information given. Do not invent results."},
            {"role": "user", "content": user_content},
        ],
        "temperature": 0.0,
        "max_tokens": 100,  # one sentence; also caps KV-cache use per request
    }


class LLMError(Exception):
    """Raised when a summary could not be obtained after all retries."""


def summarize(session: requests.Session, article: Article) -> str:
    payload = build_payload(article)
    last_err: Exception | None = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = session.post(
                API_URL, json=payload, timeout=(CONNECT_TIMEOUT_S, READ_TIMEOUT_S)
            )
            if resp.status_code in RETRYABLE_STATUS:
                raise requests.HTTPError(f"retryable HTTP {resp.status_code}", response=resp)
            resp.raise_for_status()  # other 4xx -> not retryable, handled below

            body = resp.json()
            text = body["choices"][0]["message"]["content"].strip()
            if not text:
                raise ValueError("empty completion")
            return text

        except (requests.ConnectionError, requests.Timeout) as exc:
            last_err = exc  # server dropped / overloaded / still computing
        except requests.HTTPError as exc:
            status = exc.response.status_code if exc.response is not None else None
            if status not in RETRYABLE_STATUS:
                raise LLMError(f"non-retryable HTTP {status}: {exc}") from exc
            last_err = exc
        except (KeyError, IndexError, ValueError) as exc:  # malformed JSON body
            last_err = exc

        if attempt < MAX_RETRIES:
            delay = BACKOFF_BASE_S * 2 ** (attempt - 1) + random.uniform(0, 1)
            log.warning("[%s] attempt %d/%d failed (%s); retrying in %.1fs",
                        article.id, attempt, MAX_RETRIES, last_err, delay)
            time.sleep(delay)

    raise LLMError(f"gave up after {MAX_RETRIES} attempts: {last_err}")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", help="path to JSON file (default: embedded sample)")
    ap.add_argument("--output", default="summaries.json")
    ap.add_argument("--dry-run", action="store_true", help="clean data, print payloads, no HTTP")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if args.input:
        with open(args.input, encoding="utf-8") as f:
            raw = f.read()
    else:
        raw = SAMPLE_DATA

    articles = load_and_clean(raw)
    if not articles:
        log.error("No usable articles; nothing to do.")
        return 1

    if args.dry_run:
        for a in articles:
            print(json.dumps(build_payload(a), indent=2))
        return 0

    results = []
    with requests.Session() as session:
        for a in articles:
            try:
                summary, status = summarize(session, a), "ok"
            except LLMError as exc:
                # One bad record must never kill the whole batch.
                log.error("[%s] failed: %s", a.id, exc)
                summary, status = None, f"error: {exc}"
            results.append({"id": a.id, "title": a.title, "authors": a.authors,
                            "summary": summary, "status": status})
            log.info("[%s] %s", a.id, status)

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    log.info("Wrote %s", args.output)
    return 0 if all(r["status"] == "ok" for r in results) else 2


if __name__ == "__main__":
    sys.exit(main())
