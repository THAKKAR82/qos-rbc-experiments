"""Stream file checksums as CSV; does not load datasets into memory."""

import argparse
import csv
import hashlib
from pathlib import Path
import sys


def checksum(path):
    """Return path, bytes actually read, and SHA-256 using 1 MiB chunks."""
    path = Path(path)
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
            size += len(chunk)
    return {"path": str(path), "size_bytes": size, "sha256": digest.hexdigest()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", type=Path)
    args = parser.parse_args()
    writer = csv.DictWriter(sys.stdout, fieldnames=["path", "size_bytes", "sha256"])
    writer.writeheader()
    for path in args.paths:
        try:
            writer.writerow(checksum(path))
        except OSError as error:
            parser.exit(1, f"{path}: {error}\n")


if __name__ == "__main__":
    main()
