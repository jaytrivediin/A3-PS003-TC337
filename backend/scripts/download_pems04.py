"""Download and validate the real PEMS04 archive.

Usage (from the backend directory):
    python scripts/download_pems04.py            # download if missing
    python scripts/download_pems04.py --force    # re-download
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.config import get_settings  # noqa: E402
from app.data.pems04 import DatasetError, PEMS04Dataset  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Download the PEMS04 dataset (ASTGCN release).")
    parser.add_argument("--force", action="store_true", help="Re-download even if the file exists.")
    args = parser.parse_args()
    settings = get_settings()
    ds = PEMS04Dataset(settings.dataset_path, settings.pems04_url)
    if ds.file_exists() and not args.force:
        print(f"Archive already present at {ds.path} ({ds.file_size()} bytes). Use --force to re-download.")
    else:
        print(f"Downloading {ds.url} -> {ds.path}")
        try:
            ds.download(progress=False, timeout=120)
        except DatasetError as exc:
            print(f"ERROR: {exc}")
            return 1
    try:
        arr = ds.load()
    except DatasetError as exc:
        print(f"ERROR: downloaded file failed validation: {exc}")
        return 1
    print(f"Validated PEMS04 array: shape={arr.shape} dtype={arr.dtype} (time steps, sensors, features)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
