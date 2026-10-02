from __future__ import annotations

import argparse
from pathlib import Path

from realwaste.constants import (
    DEFAULT_ARCHIVE,
    DEFAULT_CHECKSUM,
    DEFAULT_DATASET_ROOT,
    REALWASTE_URL,
)
from realwaste.data.download import (
    download_and_extract,
    download_archive,
    sha256_file,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download and verify the RealWaste dataset")
    parser.add_argument("--url", default=REALWASTE_URL)
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--checksum", type=Path, default=DEFAULT_CHECKSUM)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--force-extract", action="store_true")
    parser.add_argument(
        "--record-checksum",
        action="store_true",
        help="Explicitly trust the downloaded UCI archive and write its SHA-256",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.record_checksum:
        if not args.archive.exists():
            download_archive(args.url, args.archive)
        checksum = sha256_file(args.archive)
        args.checksum.parent.mkdir(parents=True, exist_ok=True)
        args.checksum.write_text(f"{checksum}  {args.archive.name}\n", encoding="utf-8")
        print(f"Recorded trusted checksum in {args.checksum}")

    dataset_root = download_and_extract(
        args.archive,
        args.checksum,
        args.dataset_root,
        url=args.url,
        verify_only=args.verify_only,
        force_extract=args.force_extract,
    )
    print(f"RealWaste is ready at {dataset_root}")


if __name__ == "__main__":
    main()
