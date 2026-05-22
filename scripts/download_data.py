"""Helper to download visu-predict input files from the maintainer's Google Drive folder.

Usage:
    pip install gdown
    python scripts/download_data.py --dest ./outputs/inputs

The script downloads the entire shared Drive folder. To copy individual files
only, open the folder URL in a browser and download what you need.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

DRIVE_FOLDER_URL = (
    "https://drive.google.com/drive/folders/1eNGQpeHlxa7SWnpzIjeHFif4Ae15gjgs"
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dest", type=Path, default=Path("./outputs/inputs"),
        help="Directory to download into (default: ./outputs/inputs)",
    )
    parser.add_argument(
        "--quiet", action="store_true", help="Suppress progress output from gdown",
    )
    args = parser.parse_args()

    try:
        import gdown
    except ImportError:
        print(
            "gdown not installed. Run: pip install gdown\n"
            f"Or download manually from: {DRIVE_FOLDER_URL}",
            file=sys.stderr,
        )
        return 1

    args.dest.mkdir(parents=True, exist_ok=True)
    print(f"Downloading shared Drive folder into {args.dest.resolve()} ...")
    gdown.download_folder(
        url=DRIVE_FOLDER_URL,
        output=str(args.dest),
        quiet=args.quiet,
        use_cookies=False,
    )
    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
