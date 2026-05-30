"""Download the SILSO monthly sunspot dataset to a local CSV file.

Run from the repository root:
    python project/src/models/download_dataset.py

If Windows/Python has SSL certificate problems, use:
    python project/src/models/download_dataset.py --allow-insecure-download

The script creates:
    project/data/raw/SN_m_tot_V2.0.csv
"""

from __future__ import annotations

import argparse
import ssl
import urllib.error
import urllib.request
from pathlib import Path


SILSO_URL = "https://www.sidc.be/silso/DATA/SN_m_tot_V2.0.csv"
PROJECT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_PATH = PROJECT_DIR / "data" / "raw" / "SN_m_tot_V2.0.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download SILSO monthly sunspot CSV.")
    parser.add_argument("--url", default=SILSO_URL, help="Dataset URL.")
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Where to save the CSV file.",
    )
    parser.add_argument(
        "--allow-insecure-download",
        action="store_true",
        help=(
            "Retry without SSL certificate verification if normal HTTPS download fails. "
            "Use only for this public dataset."
        ),
    )
    return parser.parse_args()


def download_file(url: str, output_path: Path, allow_insecure_download: bool) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        with urllib.request.urlopen(url, timeout=60) as response:
            content = response.read()
    except urllib.error.URLError as exc:
        message = str(exc)
        is_ssl_error = "CERTIFICATE_VERIFY_FAILED" in message or "SSL" in message
        if not is_ssl_error or not allow_insecure_download:
            raise RuntimeError(
                "Failed to download dataset. If this is an SSL certificate problem, "
                "rerun with --allow-insecure-download or download the file manually."
            ) from exc

        print(
            "WARNING: SSL certificate verification failed. "
            "Retrying without certificate verification for the public SILSO CSV."
        )
        context = ssl._create_unverified_context()
        with urllib.request.urlopen(url, context=context, timeout=60) as response:
            content = response.read()

    if not content:
        raise RuntimeError("Downloaded file is empty.")

    output_path.write_bytes(content)


def main() -> None:
    args = parse_args()
    download_file(args.url, args.output, args.allow_insecure_download)
    print(f"Saved dataset to: {args.output}")
    print("Next step:")
    print(f"python project/src/models/train.py --data-path {args.output}")


if __name__ == "__main__":
    main()
