"""Download Open Targets Platform 25.06 credible_set + study parquet shards.

Open Targets 25.06 release (Jun 2026) provides 2.6M GWAS / eQTL / pQTL
credible sets from SuSiE/PICS fine-mapping, parquet-format on EBI FTP.

Schema notes (verified on part-00000, see R_raw/open_targets/probe.md):
  - 25 parts × ~100 MB each, ~2.5 GB total credible_set
  - Each row = one (study, locus) credible set, with `locus` array of
    {variantId, posteriorProbability, logBF, beta, standardError, r2Overall}
  - studyType ∈ {gwas, eqtl, tuqtl, sqtl, sceqtl, pqtl}; we filter to gwas
  - chromosome partitioning: part-00000 was chr1 in our probe.

Usage:
    # Download all 25 parts (~30 min on a 3 MB/s link)
    python scripts/40_open_targets_download.py \\
        --release 25.06 \\
        --datasets credible_set study \\
        --out-dir data/raw/open_targets

    # Smoke test: just 3 parts
    python scripts/40_open_targets_download.py --n-parts 3 ...
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import requests
from tqdm.auto import tqdm

FTP_BASE = "https://ftp.ebi.ac.uk/pub/databases/opentargets/platform"
HREF_RE = re.compile(r'href="(part-\d+-[^"]+\.snappy\.parquet)"')


def list_parts(release: str, dataset: str, timeout: int = 30) -> list[str]:
    """Scrape the FTP index page for part-XXXXX filenames in this dataset."""
    url = f"{FTP_BASE}/{release}/output/{dataset}/"
    r = requests.get(url, timeout=timeout)
    r.raise_for_status()
    parts = HREF_RE.findall(r.text)
    if not parts:
        raise RuntimeError(f"No parquet parts at {url}")
    return sorted(parts)


def _expected_size(url: str, timeout: int = 30) -> int | None:
    """HEAD the URL to discover the canonical file size for skip / verify."""
    try:
        r = requests.head(url, allow_redirects=True, timeout=timeout)
        r.raise_for_status()
        return int(r.headers["content-length"])
    except Exception:
        return None


def download_one(url: str, dest: Path, timeout: int = 600,
                 max_retries: int = 5) -> None:
    """Stream-download with HTTP Range resume + retry on chunked-encoding errors.

    Idempotent: if dest exists at the expected size, skip; if smaller, resume
    from the byte offset.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    expected = _expected_size(url, timeout)

    for attempt in range(1, max_retries + 1):
        if dest.exists() and expected is not None:
            cur = dest.stat().st_size
            if cur == expected:
                print(f"[skip] {dest.name} already complete ({cur:,} B)")
                return
            if cur > expected:
                print(f"[reset] {dest.name} size {cur} > expected {expected}, re-downloading")
                dest.unlink()
                cur = 0
        else:
            cur = dest.stat().st_size if dest.exists() else 0

        headers = {"Range": f"bytes={cur}-"} if cur > 0 else {}
        try:
            with requests.get(url, stream=True, timeout=timeout,
                              headers=headers) as r:
                if cur > 0 and r.status_code == 200:
                    # Server ignored Range; restart from scratch.
                    dest.unlink()
                    cur = 0
                r.raise_for_status()
                total = expected or (int(r.headers.get("content-length", 0))
                                     + cur)
                mode = "ab" if cur > 0 else "wb"
                with dest.open(mode) as f, tqdm(
                    desc=dest.name, total=total, initial=cur,
                    unit="B", unit_scale=True
                ) as pbar:
                    for chunk in r.iter_content(chunk_size=1 << 20):
                        if chunk:
                            f.write(chunk)
                            pbar.update(len(chunk))
            # Post-write size check
            if expected is not None and dest.stat().st_size != expected:
                raise RuntimeError(
                    f"size mismatch: got {dest.stat().st_size}, expected {expected}")
            return
        except (requests.exceptions.ChunkedEncodingError,
                requests.exceptions.ConnectionError,
                requests.exceptions.ReadTimeout,
                RuntimeError) as e:
            print(f"[retry {attempt}/{max_retries}] {dest.name}: {type(e).__name__}: {e}")
            if attempt == max_retries:
                raise


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--release", default="25.06")
    ap.add_argument("--datasets", nargs="+",
                    default=["credible_set", "study"],
                    help="Subdirectories under output/ to fetch.")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--n-parts", type=int, default=0,
                    help="Cap parts per dataset (0 = all). Smoke-test with 3.")
    args = ap.parse_args()

    for ds in args.datasets:
        print(f"\n=== {ds} ===")
        try:
            parts = list_parts(args.release, ds)
        except Exception as e:
            print(f"[error] cannot list {ds}: {e}", file=sys.stderr)
            continue
        if args.n_parts > 0:
            parts = parts[:args.n_parts]
        print(f"[plan] {len(parts)} parts")
        ds_dir = args.out_dir / ds
        for fname in parts:
            url = f"{FTP_BASE}/{args.release}/output/{ds}/{fname}"
            # Strip the SuSie/Spark UUID for cleaner local filenames.
            clean = re.sub(r"-[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}",
                           "", fname)
            download_one(url, ds_dir / clean)

    print("\n[done]", args.out_dir)


if __name__ == "__main__":
    main()
