#!/usr/bin/env python3
"""
Fetch all volleyball match XML files from the PVL dashboard and save to data/xml_files.

Examples:
  python fetch_all_matches.py --list
  python fetch_all_matches.py --download --limit 50
  python fetch_all_matches.py --download --year 2025
  python fetch_all_matches.py --download --tournament PVL2024A

Defaults:
  - Output directory defaults to scripts.config.XML_DIR (data/xml_files)
  - Polite delay between downloads
"""
from __future__ import annotations
import argparse
import re
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
import xml.etree.ElementTree as ET

try:
    # When run from project root
    from scripts.config import XML_DIR, XML_DIR_STR, ensure_directories
except Exception:  # When run from scripts/
    from config import XML_DIR, XML_DIR_STR, ensure_directories

import requests

BASE_URL = 'https://dashboard.pvl.ph/assets/match_results/xml/'


class XMLFileFetcher:
    def __init__(self, base_url: str = BASE_URL):
        self.base_url = base_url
        self.xml_files: List[str] = []

    def scrape_directory(self) -> List[str]:
        print(f"Scraping directory: {self.base_url}")
        try:
            resp = requests.get(self.base_url, timeout=30)
            resp.raise_for_status()
            pattern = r'href="([^\"]+\.xml)"'
            matches = re.findall(pattern, resp.text)
            xml_files = [f for f in matches if f.endswith('.xml') and not f.startswith('..')]
            xml_files = sorted(set(xml_files))
            print(f"✓ Found {len(xml_files)} XML files")
            self.xml_files = xml_files
            return xml_files
        except requests.RequestException as e:
            print(f"✗ Error scraping directory: {e}")
            return []

    def download_file(self, filename: str, output_dir: Path, timeout: int = 30) -> bool:
        url = f"{self.base_url}{filename}"
        dest = output_dir / filename
        try:
            r = requests.get(url, timeout=timeout)
            r.raise_for_status()
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(r.content)
            # Validate XML is parseable
            try:
                ET.fromstring(r.content)
            except ET.ParseError:
                print(f"  ⚠ Invalid XML structure for {filename} (saved anyway)")
            return True
        except Exception as e:
            print(f"  ✗ Error downloading {filename}: {e}")
            return False

    def download_all(self, output_dir: Path, limit: Optional[int] = None, delay: float = 0.1) -> Dict[str, int]:
        if not self.xml_files:
            self.scrape_directory()
        output_dir.mkdir(parents=True, exist_ok=True)
        files = self.xml_files[:limit] if limit else self.xml_files
        ok = skipped = failed = 0
        print(f"\nDownloading {len(files)} file(s) to {output_dir}/")
        print("=" * 70)
        for i, fn in enumerate(files, 1):
            dest = output_dir / fn
            if dest.exists():
                print(f"[{i}/{len(files)}] ⊘ {fn} (already exists)")
                skipped += 1
            else:
                print(f"[{i}/{len(files)}] Downloading {fn}...", end=' ')
                if self.download_file(fn, output_dir):
                    size = dest.stat().st_size
                    print(f"✓ ({size} bytes)")
                    ok += 1
                else:
                    print("✗")
                    failed += 1
            if delay and i < len(files):
                time.sleep(delay)
        print("=" * 70)
        print(f"Summary: {ok} downloaded, {skipped} skipped, {failed} failed")
        return {"successful": ok, "skipped": skipped, "failed": failed, "total": len(files)}

    def filter_files(self, pattern: Optional[str] = None, year: Optional[int] = None, tournament: Optional[str] = None) -> List[str]:
        if not self.xml_files:
            self.scrape_directory()
        files = self.xml_files
        if year is not None:
            files = [f for f in files if str(year) in f]
        if tournament:
            t = tournament.upper()
            files = [f for f in files if t in f.upper()]
        if pattern:
            files = [f for f in files if re.search(pattern, f, re.IGNORECASE)]
        self.xml_files = files
        return files


def main():
    parser = argparse.ArgumentParser(description="Fetch volleyball match XML files from PVL dashboard")
    parser.add_argument('--list', action='store_true', help='List available files (after optional filters)')
    parser.add_argument('--download', action='store_true', help='Download files to data/xml_files')
    parser.add_argument('--limit', type=int, help='Limit the number of files')
    parser.add_argument('--year', type=int, help='Filter by year (e.g., 2024, 2025)')
    parser.add_argument('--tournament', type=str, help='Filter by tournament code (e.g., PVL2024A)')
    parser.add_argument('--delay', type=float, default=0.1, help='Delay between downloads (seconds)')

    args = parser.parse_args()

    ensure_directories()
    output_dir = XML_DIR

    fetcher = XMLFileFetcher()
    files = fetcher.scrape_directory()
    if not files:
        raise SystemExit("Failed to scrape the source directory.")

    fetcher.filter_files(year=args.year, tournament=args.tournament)

    if args.list:
        print("\nFirst 50 files:")
        for i, fn in enumerate(fetcher.xml_files[:50], 1):
            print(f"{i:3d}. {fn}")
        print(f"Total: {len(fetcher.xml_files)} files")
        return

    if args.download:
        res = fetcher.download_all(output_dir=output_dir, limit=args.limit, delay=args.delay)
        print(f"\n✓ Download complete. Files saved to: {XML_DIR_STR}")
        if res.get('failed', 0) > 0:
            raise SystemExit(1)
        return

    # Default help hint
    print("""
Usage examples:
  python fetch_all_matches.py --list
  python fetch_all_matches.py --download --limit 50
  python fetch_all_matches.py --download --year 2025
  python fetch_all_matches.py --download --tournament PVL2024A
""")


if __name__ == '__main__':
    main()
