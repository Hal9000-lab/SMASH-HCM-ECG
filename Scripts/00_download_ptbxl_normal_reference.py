#!/usr/bin/env python3
"""Download the selected PTB-XL Normal Reference Cohort waveforms.

Default SMASH-HCM layout:
    _SMASH-HCM/
    |-- Data/
    |   |-- PTBXL_normal_reference_manifest.csv
    |   `-- PTBXL_Normal/
    |       `-- records500/
    `-- Scripts/
        `-- download_ptbxl_normal_reference.py

The project root is inferred from this script's location, so the script can be
launched directly from VS Code without specifying --manifest or --out.
Existing non-empty files are skipped, and partial downloads can be resumed.
No third-party Python packages are required.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

USER_AGENT = "PTBXL-normal-reference-downloader/1.1"


def get_default_paths() -> tuple[Path, Path, Path]:
    script_dir = Path(__file__).resolve().parent
    # Expected location: <project_root>/Scripts/this_script.py
    project_root = script_dir.parent
    data_dir = project_root / "Data"
    manifest = data_dir / "PTBXL_normal_reference_manifest.csv"
    out_root = data_dir / "PTBXL_Normal" / "records500"
    return project_root, manifest, out_root


def download(url: str, dest: Path, retries: int = 5, timeout: int = 60) -> str:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        return "skipped"

    part = dest.with_suffix(dest.suffix + ".part")
    for attempt in range(1, retries + 1):
        try:
            headers = {"User-Agent": USER_AGENT}
            start = part.stat().st_size if part.exists() else 0
            if start > 0:
                headers["Range"] = f"bytes={start}-"

            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=timeout) as response:
                status = getattr(response, "status", 200)
                mode = "ab" if start > 0 and status == 206 else "wb"
                with open(part, mode) as f:
                    while True:
                        chunk = response.read(1024 * 1024)
                        if not chunk:
                            break
                        f.write(chunk)

            os.replace(part, dest)
            return "downloaded"

        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as exc:
            if attempt == retries:
                raise
            wait = min(2 ** attempt, 30)
            print(
                f"  retry {attempt}/{retries} after error: {exc}; waiting {wait}s",
                file=sys.stderr,
            )
            time.sleep(wait)

    return "failed"


def main() -> int:
    project_root, default_manifest, default_out = get_default_paths()

    parser = argparse.ArgumentParser(
        description="Download selected PTB-XL 500-Hz normal-reference records."
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=default_manifest,
        help=f"Manifest CSV (default: {default_manifest})",
    )
    parser.add_argument(
        "--out",
        "--output",
        dest="out",
        type=Path,
        default=default_out,
        help=f"Output records500 directory (default: {default_out})",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Download only the first N records (recommended for an initial test).",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.05,
        help="Pause between records in seconds (default: 0.05).",
    )
    args = parser.parse_args()

    manifest = args.manifest.resolve()
    out_root = args.out.resolve()

    print(f"Project root : {project_root}")
    print(f"Manifest     : {manifest}")
    print(f"Output       : {out_root}")

    if not manifest.is_file():
        print("\nERROR: manifest not found.", file=sys.stderr)
        print(f"Expected: {manifest}", file=sys.stderr)
        print(
            "Place PTBXL_normal_reference_manifest.csv inside the project's Data folder "
            "or pass --manifest <path>.",
            file=sys.stderr,
        )
        return 2

    out_root.mkdir(parents=True, exist_ok=True)

    with manifest.open(newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))

    required = {"ecg_id", "filename_hr", "header_url", "signal_url"}
    if not rows:
        print("ERROR: manifest is empty.", file=sys.stderr)
        return 2
    missing = required.difference(rows[0].keys())
    if missing:
        print(
            "ERROR: manifest is missing required columns: " + ", ".join(sorted(missing)),
            file=sys.stderr,
        )
        return 2

    if args.limit is not None:
        if args.limit <= 0:
            print("ERROR: --limit must be greater than zero.", file=sys.stderr)
            return 2
        rows = rows[: args.limit]

    print(f"Records to process: {len(rows)}")
    downloaded = skipped = failed = 0

    for i, row in enumerate(rows, 1):
        rel = row["filename_hr"].replace("\\", "/")
        rel_no_prefix = rel[len("records500/") :] if rel.startswith("records500/") else rel
        base_dest = out_root / Path(rel_no_prefix)

        files = (
            (row["header_url"], base_dest.with_suffix(".hea")),
            (row["signal_url"], base_dest.with_suffix(".dat")),
        )

        print(f"[{i}/{len(rows)}] ECG {row['ecg_id']} -> {rel_no_prefix}")
        for url, dest in files:
            try:
                state = download(url, dest)
                if state == "downloaded":
                    downloaded += 1
                else:
                    skipped += 1
            except Exception as exc:
                failed += 1
                print(f"  FAILED {url}: {exc}", file=sys.stderr)

        if args.sleep > 0:
            time.sleep(args.sleep)

    print(
        f"Done. Files downloaded: {downloaded}; skipped: {skipped}; failed: {failed}"
    )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
