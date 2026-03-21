#!/usr/bin/env python3
# Author: Jakub Cabal <jakubcabal@gmail.com>
# SPDX-License-Identifier: MIT

"""
fake_scan.py - Convert PDFs to image-based PDFs (simulated scans)

Converts digital PDFs (text, vectors) to image-based PDFs.
Each page becomes a JPEG image, multi-page PDFs stay multi-page.

Usage: python fake_scan.py [-d DIR] [-q QUALITY] [-r DPI] [-a]
"""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


def check_deps():
    """Check if required commands are installed."""
    missing = []
    for cmd, pkg in [("pdftoppm", "poppler-utils"), ("convert", "imagemagick")]:
        if not shutil.which(cmd):
            missing.append(f"{cmd} ({pkg})")
    if missing:
        print(f"Error: Missing required commands: {', '.join(missing)}", file=sys.stderr)
        return False
    return True


def process_file(path: Path, quality: int, dpi: int, auto_orient: bool) -> bool:
    """Convert a PDF to image-based PDF (simulated scan)."""
    parent = path.parent
    stem = path.stem
    out_pdf = parent / f"{stem}_scan.pdf"
    
    print(f"Processing: {path.name}")
    
    # Convert PDF pages to JPEG images
    result = subprocess.run([
        "pdftoppm", "-jpeg", "-jpegopt", f"quality={quality}",
        "-r", str(dpi), str(path.resolve()), str(parent / f"{stem}_scan")
    ], capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"  Failed: pdftoppm error", file=sys.stderr)
        if result.stderr.strip():
            print(f"  {result.stderr.strip()}", file=sys.stderr)
        for f in parent.glob(f"{stem}_scan*.jpg"):
            f.unlink()
        return False
    
    images = sorted(parent.glob(f"{stem}_scan-*.jpg"))
    if not images:
        print(f"  Failed: no images created", file=sys.stderr)
        return False
    
    print(f"  Found {len(images)} page(s)")
    
    # Convert images back to PDF
    cmd = ["convert"] + [str(img) for img in images]
    if auto_orient:
        cmd.insert(-1, "-auto-orient")
    cmd.append(str(out_pdf))
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"  Failed: convert error", file=sys.stderr)
        if result.stderr.strip():
            print(f"  {result.stderr.strip()}", file=sys.stderr)
        for img in images:
            img.unlink()
        return False
    
    for img in images:
        img.unlink()
    
    print(f"  Created: {stem}_scan.pdf ({len(images)} pages)")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Convert PDFs to image-based PDFs (simulated scans).",
        epilog="Examples:\n  python fake_scan.py -d ~/Documents -q 90 -r 300 -a",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument("-d", "--dir", type=Path, metavar="DIR",
                        help="Directory with PDF files (default: current)")
    parser.add_argument("-q", "--quality", type=int, default=80, metavar="NUM",
                        help="JPEG quality 1-100 (default: 80)")
    parser.add_argument("-r", "--dpi", type=int, default=200, metavar="NUM",
                        help="Resolution in DPI (default: 200)")
    parser.add_argument("-a", "--auto-orient", action="store_true",
                        help="Auto-correct page orientation")
    
    args = parser.parse_args()
    
    if len(sys.argv) == 1:
        parser.print_help()
        return 0
    
    directory = args.dir or Path(".")
    
    if not directory.is_dir():
        print(f"Error: Not a directory: {directory}", file=sys.stderr)
        return 1
    
    if not 1 <= args.quality <= 100:
        print("Error: Quality must be 1-100", file=sys.stderr)
        return 1
    
    if args.dpi < 1:
        print("Error: DPI must be positive", file=sys.stderr)
        return 1
    
    if not check_deps():
        return 1
    
    # Find all PDF files (handle both .pdf and .PDF extensions)
    pdfs = list({p.resolve(): p for p in directory.glob("*.pdf")}.values())
    pdfs += list({p.resolve(): p for p in directory.glob("*.PDF")}.values())
    pdfs = list({p.resolve(): p for p in pdfs}.values())  # Remove duplicates
    
    if not pdfs:
        print("No PDF files found")
        return 0
    
    info = f" +auto-orient" if args.auto_orient else ""
    print(f"Processing {len(pdfs)} file(s) in: {directory.resolve()}")
    print(f"Settings: quality={args.quality}, dpi={args.dpi}{info}\n")
    
    ok = sum(1 for p in pdfs if process_file(p, args.quality, args.dpi, args.auto_orient))
    fail = len(pdfs) - ok
    
    print(f"\nSummary: {ok} processed, {fail} failed")
    return 1 if fail else 0


if __name__ == "__main__":
    sys.exit(main())
