#!/usr/bin/env python
"""
find_target_header.py

Search a directory tree of DESI reconstruction FITS files for a given
TARGETID and print the header metadata (TARGETID, NIGHT, Z, NORM, etc.)
found in extension 1 of each matching file.

Assumes a directory layout like:
    <base_dir>/<TARGETID>/<NIGHT>/recon_<TARGETID>_<NIGHT>.fits

but does NOT rely on the filename — it opens each FITS file and checks
the actual TARGETID header keyword, so it will still find matches even
if the file naming convention changes or is inconsistent.

Usage:
    python find_target_header.py 39627553509475403
    python find_target_header.py 39627553509475403 --base-dir /path/to/reconstructions
    python find_target_header.py 39627553509475403 --ext 1 --verbose
"""

import argparse
import sys
from pathlib import Path

from astropy.io import fits


DEFAULT_BASE_DIR = "/work/10579/prisha/ls6/desi_project/reconstructions"


def find_matching_files(base_dir, pattern="recon_*.fits"):
    """Recursively yield all FITS files under base_dir matching pattern."""
    base = Path(base_dir)
    if not base.exists():
        print(f"ERROR: base directory does not exist: {base}", file=sys.stderr)
        sys.exit(1)
    yield from base.rglob(pattern)


def get_header_info(filepath, ext=1):
    """
    Open a FITS file and return its extension header as a dict,
    or None if the file can't be read / extension doesn't exist.
    """
    try:
        with fits.open(filepath) as hdul:
            if ext >= len(hdul):
                return None
            return dict(hdul[ext].header)
    except Exception as e:
        print(f"  (skipped {filepath}: {e})", file=sys.stderr)
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Find and print header metadata for a given TARGETID."
    )
    parser.add_argument("targetid", type=int, help="TARGETID to search for")
    parser.add_argument(
        "--base-dir",
        default=DEFAULT_BASE_DIR,
        help=f"Root directory to search (default: {DEFAULT_BASE_DIR})",
    )
    parser.add_argument(
        "--ext",
        type=int,
        default=1,
        help="FITS extension index containing the header to check (default: 1)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print every file checked, not just matches",
    )
    args = parser.parse_args()

    print(f"Searching under: {args.base_dir}")
    print(f"Looking for TARGETID = {args.targetid}\n")

    matches = 0
    checked = 0

    for filepath in find_matching_files(args.base_dir):
        checked += 1
        header = get_header_info(filepath, ext=args.ext)

        if header is None:
            if args.verbose:
                print(f"[skip]  {filepath}")
            continue

        file_targetid = header.get("TARGETID")

        if args.verbose:
            print(f"[check] {filepath}  TARGETID={file_targetid}")

        if file_targetid == args.targetid:
            matches += 1
            print(f"MATCH: {filepath}")
            print("-" * 60)
            for key in ("TARGETID", "NIGHT", "Z", "NORM"):
                if key in header:
                    print(f"  {key:10s} = {header[key]}")
            # print any other non-standard keywords too, in case there are more
            extra_keys = [
                k
                for k in header
                if k not in ("TARGETID", "NIGHT", "Z", "NORM")
                and not k.startswith(("TTYPE", "TFORM", "TUNIT"))
                and k
                not in (
                    "XTENSION",
                    "BITPIX",
                    "NAXIS",
                    "NAXIS1",
                    "NAXIS2",
                    "PCOUNT",
                    "GCOUNT",
                    "TFIELDS",
                )
            ]
            if extra_keys:
                print("  Other header keywords:")
                for k in extra_keys:
                    print(f"    {k:10s} = {header[k]}")
            print()

    print(f"Checked {checked} file(s), found {matches} match(es).")
    if matches == 0:
        print(
            "No matches found. Double-check the TARGETID and --base-dir, "
            "or run with --verbose to see every file that was checked."
        )


if __name__ == "__main__":
    main()