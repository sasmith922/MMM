"""
download_data.py
----------------
Script to download raw NCAA data files and place them in data/raw/.

Usage
-----
    python scripts/download_data.py

Currently this script serves as a placeholder.  Implement one of the
data-acquisition strategies listed in the TODOs below to populate the
data/raw directory before running any other pipeline scripts.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from madness_model.paths import RAW_DATA_DIR

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download raw NCAA tournament data.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=RAW_DATA_DIR,
        help="Directory to write downloaded files (default: data/raw).",
    )
    return parser.parse_args()


def download(output_dir: Path) -> None:
    """Download raw data files to *output_dir*.

    Parameters
    ----------
    output_dir:
        Destination directory.  Created if it does not exist.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    log.info("Output directory: %s", output_dir)

    # TODO: Option 1 — Kaggle API
    #   import subprocess
    #   subprocess.run([
    #       "kaggle", "competitions", "download",
    #       "-c", "march-machine-learning-mania-2024",
    #       "-p", str(output_dir), "--unzip",
    #   ], check=True)

    # TODO: Option 2 — Direct URL download using requests/urllib
    #   URLS = {
    #       "teams.csv": "https://...",
    #       "regular_season.csv": "https://...",
    #       ...
    #   }
    #   for filename, url in URLS.items():
    #       ...

    log.warning(
        "No data source configured.  "
        "Add a download strategy in scripts/download_data.py."
    )


def main() -> None:
    args = parse_args()
    download(args.output_dir)
    log.info("Done.")


if __name__ == "__main__":
    main()
