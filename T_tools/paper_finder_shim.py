#!/usr/bin/env python3
"""Drop-in replacement for the missing paper_finder.py from the deep-research skill.

Backend: https://wenhanacademia-ai-paper-finder.hf.space (public, no auth)
  Endpoints reverse-engineered from /static/app.js:
    GET  /api/venues  -> {venue_to_years: {VENUE: [year, ...]}}
    POST /api/search  -> {records: [{title, authors, venue_display, link, ...}]}

CLI matches the original signature documented in the skill:
    python paper_finder_shim.py --mode scrape   --config <yaml>
    python paper_finder_shim.py --mode download --jsonl <results.jsonl>
    python paper_finder_shim.py --list-venues

Stdlib + PyYAML only. No gradio_client (the Space front-end is plain HTML/JS).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

import yaml

API_BASE = "https://wenhanacademia-ai-paper-finder.hf.space"
MAX_VENUE_YEAR_SELECTIONS = 15  # backend hard cap (see frontend MAX_TOTAL_SELECTIONS)
DEFAULT_TIMEOUT = 90


def http_get(url: str, timeout: int = DEFAULT_TIMEOUT) -> dict:
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def http_post_json(url: str, payload: dict, timeout: int = DEFAULT_TIMEOUT) -> dict:
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json", "Accept": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def fetch_venues() -> dict[str, list[str]]:
    return http_get(f"{API_BASE}/api/venues")["venue_to_years"]


def normalize_venue_key(supported: dict[str, list[str]], key: str) -> str | None:
    """Match user-input venue key (any case) against supported venues."""
    lower = {v.lower(): v for v in supported}
    return lower.get(key.lower())


def parse_year_from_venue_display(s: str) -> int | None:
    m = re.search(r"\b(20\d{2})\b", s or "")
    return int(m.group(1)) if m else None


def parse_venue_from_display(s: str) -> str:
    """'NeurIPS 2024 Poster' -> 'NeurIPS'."""
    return (s or "").split(" ")[0]


def extract_openreview_id(bibtex: str) -> str | None:
    m = re.search(r"openreview\.net/forum\?id=([A-Za-z0-9_-]+)", bibtex or "")
    return m.group(1) if m else None


def to_paper_db_record(rec: dict) -> dict:
    """Map ai-paper-finder record -> paper_db.jsonl schema used by deep-research skill."""
    venue_display = rec.get("venue_display", "")
    venue = parse_venue_from_display(venue_display)
    return {
        "title": rec.get("title", ""),
        "authors": rec.get("authors", []),
        "abstract": rec.get("abstract_md", ""),
        "year": parse_year_from_venue_display(venue_display),
        "venue": venue,
        "venue_normalized": venue.lower(),
        "peer_reviewed": True,  # this backend only indexes peer-reviewed conferences
        "citationCount": None,  # not provided by this API
        "paperId": extract_openreview_id(rec.get("bibtex", "")),
        "arxiv_id": None,  # not provided
        "pdf_url": rec.get("link", ""),
        "tags": rec.get("keywords", []),
        "source": "ai-paper-finder",
        "bibtex": rec.get("bibtex", ""),
        "similarity": rec.get("similarity"),
        "venue_display": venue_display,
    }


def build_venue_years_payload(
    venue_filter: dict[str, list[int]],
    supported: dict[str, list[str]],
) -> dict[str, list[str]]:
    """Validate + normalize user-specified venue/year filter against backend support."""
    out: dict[str, list[str]] = {}
    for raw_key, years in venue_filter.items():
        canonical = normalize_venue_key(supported, raw_key)
        if canonical is None:
            print(f"[warn] venue '{raw_key}' not supported by backend; skipping. "
                  f"Run --list-venues to see available.", file=sys.stderr)
            continue
        avail = set(supported[canonical])
        keep = [str(y) for y in years if str(y) in avail]
        skipped = [y for y in years if str(y) not in avail]
        if skipped:
            print(f"[warn] {canonical}: years {skipped} not indexed; available={sorted(avail)}",
                  file=sys.stderr)
        if keep:
            out[canonical] = keep
    return out


def search_one(
    query: str,
    venue_years: dict[str, list[str]],
    total_results: int,
    max_retries: int = 3,
) -> list[dict]:
    """Call POST /api/search with retry on 429."""
    payload = {"query": query, "total_results": total_results, "venue_years": venue_years}
    delay = 5
    for attempt in range(1, max_retries + 1):
        try:
            data = http_post_json(f"{API_BASE}/api/search", payload)
            return data.get("records", [])
        except urllib.error.HTTPError as e:
            if e.code == 429 and attempt < max_retries:
                # backend returns remaining_seconds in body; use generous default
                wait = delay * attempt
                print(f"[rate-limit] waiting {wait}s (attempt {attempt}/{max_retries})",
                      file=sys.stderr)
                time.sleep(wait)
                continue
            raise


def cmd_list_venues() -> int:
    venues = fetch_venues()
    print(f"Backend: {API_BASE}")
    print(f"Total venues: {len(venues)}")
    print()
    for v in sorted(venues):
        years = sorted(venues[v])
        print(f"  {v:<10s}  {', '.join(years)}")
    print()
    print(f"NOTE: backend caps each search at {MAX_VENUE_YEAR_SELECTIONS} venue-year pairs total.")
    return 0


def cmd_scrape(config_path: Path) -> int:
    cfg = yaml.safe_load(config_path.read_text())
    searches = cfg.get("searches", [])
    output_cfg = cfg.get("output", {})
    out_root = Path(output_cfg.get("root", "search_results"))
    overwrite = bool(output_cfg.get("overwrite", False))
    out_root.mkdir(parents=True, exist_ok=True)

    supported = fetch_venues()
    total_papers = 0

    for i, s in enumerate(searches):
        query = s["query"]
        num_results = int(s.get("num_results", 50))
        venue_filter = s.get("venues", {})

        venue_years = build_venue_years_payload(venue_filter, supported)
        if not venue_years:
            print(f"[skip] search {i}: no valid venue/year after normalization", file=sys.stderr)
            continue

        n_pairs = sum(len(v) for v in venue_years.values())
        if n_pairs > MAX_VENUE_YEAR_SELECTIONS:
            print(f"[error] search {i}: {n_pairs} venue-year pairs exceeds backend cap "
                  f"of {MAX_VENUE_YEAR_SELECTIONS}; trim the config.", file=sys.stderr)
            return 2

        slug = re.sub(r"\W+", "_", query.lower())[:60].strip("_")
        out_file = out_root / f"{i:02d}_{slug}.jsonl"
        if out_file.exists() and not overwrite:
            print(f"[skip] {out_file} exists (overwrite: false)")
            continue

        print(f"[search {i}] query={query!r} venues={venue_years} num={num_results}")
        records = search_one(query, venue_years, num_results)
        print(f"           -> {len(records)} hits")

        with out_file.open("w") as f:
            for r in records:
                f.write(json.dumps(to_paper_db_record(r), ensure_ascii=False) + "\n")
        total_papers += len(records)

        # be polite — backend has rate limit
        if i + 1 < len(searches):
            time.sleep(2)

    print(f"[done] {total_papers} papers across {len(searches)} searches -> {out_root}")
    return 0


def cmd_download(jsonl_path: Path, output_dir: Path | None = None) -> int:
    """Download PDFs listed in JSONL. Skips entries without pdf_url."""
    output_dir = output_dir or jsonl_path.parent / "papers"
    output_dir.mkdir(parents=True, exist_ok=True)

    n_total = n_ok = n_skip = 0
    with jsonl_path.open() as f:
        for line in f:
            rec = json.loads(line)
            n_total += 1
            url = rec.get("pdf_url")
            if not url:
                n_skip += 1
                continue
            pid = rec.get("paperId") or rec.get("title", "untitled")[:40]
            safe = re.sub(r"\W+", "_", pid).strip("_")
            dst = output_dir / f"{safe}.pdf"
            if dst.exists():
                n_ok += 1
                continue
            try:
                urllib.request.urlretrieve(url, dst)
                n_ok += 1
                print(f"[ok] {dst.name}")
            except Exception as e:
                n_skip += 1
                print(f"[fail] {pid}: {e}", file=sys.stderr)
    print(f"[done] {n_ok}/{n_total} downloaded ({n_skip} skipped) -> {output_dir}")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--mode", choices=["scrape", "download"],
                   help="scrape: run yaml config; download: fetch PDFs from JSONL")
    p.add_argument("--config", type=Path, help="YAML config (for --mode scrape)")
    p.add_argument("--jsonl", type=Path, help="JSONL file (for --mode download)")
    p.add_argument("--output-dir", type=Path, help="PDF output dir (default: <jsonl-dir>/papers)")
    p.add_argument("--list-venues", action="store_true", help="Print supported venues+years")
    args = p.parse_args()

    if args.list_venues:
        return cmd_list_venues()
    if args.mode == "scrape":
        if not args.config:
            p.error("--mode scrape requires --config")
        return cmd_scrape(args.config)
    if args.mode == "download":
        if not args.jsonl:
            p.error("--mode download requires --jsonl")
        return cmd_download(args.jsonl, args.output_dir)
    p.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
