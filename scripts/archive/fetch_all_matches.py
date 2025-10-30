"""
Advanced Match Data Fetcher
Scrapes and processes all volleyball match XML files from PVL dashboard
Can download files OR process directly in memory
"""

import requests
import re
from pathlib import Path
from typing import List, Dict, Any
import xml.etree.ElementTree as ET
from io import StringIO
import time

BASE_URL = 'https://dashboard.pvl.ph/assets/match_results/xml/'


class XMLFileFetcher:
    """Fetch and process XML files from PVL dashboard"""
    
    def __init__(self):
        self.xml_files = []
        self.base_url = BASE_URL
        
    def scrape_directory(self) -> List[str]:
        """Scrape the directory listing to get all XML file names"""
        print(f"Scraping directory: {self.base_url}")
        
        try:
            response = requests.get(self.base_url)
            response.raise_for_status()
            
            # Extract XML file names from HTML
            # Pattern matches: href="filename.xml"
            pattern = r'href="([^"]+\.xml)"'
            matches = re.findall(pattern, response.text)
            
            # Filter out navigation links and keep only actual XML files
            xml_files = [f for f in matches if not f.startswith('..') and '.xml' in f]
            
            # Remove duplicates and sort
            xml_files = sorted(list(set(xml_files)))
            
            print(f"✓ Found {len(xml_files)} XML files")
            self.xml_files = xml_files
            return xml_files
            
        except requests.RequestException as e:
            print(f"✗ Error scraping directory: {e}")
            return []
    
    def download_file(self, filename: str, output_dir: str = '.') -> bool:
        """Download a single XML file"""
        url = f"{self.base_url}{filename}"
        output_path = Path(output_dir) / filename
        
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            
            with open(output_path, 'wb') as f:
                f.write(response.content)
            
            return True
            
        except Exception as e:
            print(f"  ✗ Error downloading {filename}: {e}")
            return False
    
    def download_all(self, output_dir: str = '.', limit: int = None, delay: float = 0.1) -> Dict[str, int]:
        """Download all XML files"""
        if not self.xml_files:
            self.scrape_directory()
        
        Path(output_dir).mkdir(exist_ok=True)
        
        files_to_download = self.xml_files[:limit] if limit else self.xml_files
        successful = 0
        failed = 0
        skipped = 0
        
        print(f"\nDownloading {len(files_to_download)} file(s) to {output_dir}/")
        print(f"{'='*70}")
        
        for i, filename in enumerate(files_to_download, 1):
            # Check if file already exists
            output_path = Path(output_dir) / filename
            if output_path.exists():
                print(f"[{i}/{len(files_to_download)}] ⊘ {filename} (already exists)")
                skipped += 1
                continue
            
            print(f"[{i}/{len(files_to_download)}] Downloading {filename}...", end=' ')
            
            if self.download_file(filename, output_dir):
                size = output_path.stat().st_size
                print(f"✓ ({size} bytes)")
                successful += 1
            else:
                print("✗")
                failed += 1
            
            # Polite delay to avoid overwhelming the server
            if delay > 0 and i < len(files_to_download):
                time.sleep(delay)
        
        print(f"{'='*70}")
        print(f"Summary: {successful} downloaded, {skipped} skipped, {failed} failed")
        
        return {
            'successful': successful,
            'skipped': skipped,
            'failed': failed,
            'total': len(files_to_download)
        }
    
    def fetch_and_parse_in_memory(self, filename: str) -> Dict[str, Any]:
        """Fetch XML file and parse without downloading"""
        url = f"{self.base_url}{filename}"
        
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            
            # Parse XML directly from response
            root = ET.fromstring(response.content)
            
            # Basic parsing (you can expand this)
            data = {
                'filename': filename,
                'tournament': None,
                'teams': [],
                'match_date': None
            }
            
            tournament = root.find('Tournament')
            if tournament is not None:
                data['tournament'] = tournament.get('Code')
            
            for team in root.findall('Team'):
                team_code = team.get('Code')
                if team_code:
                    data['teams'].append(team_code)
            
            match = root.find('Match')
            if match is not None:
                date_elem = match.find('Date')
                if date_elem is not None:
                    data['match_date'] = date_elem.text
            
            return data
            
        except Exception as e:
            print(f"  ✗ Error fetching {filename}: {e}")
            return None
    
    def process_all_in_memory(self, limit: int = None) -> List[Dict[str, Any]]:
        """Process all files in memory without downloading"""
        if not self.xml_files:
            self.scrape_directory()
        
        files_to_process = self.xml_files[:limit] if limit else self.xml_files
        results = []
        
        print(f"\nProcessing {len(files_to_process)} file(s) in memory...")
        print(f"{'='*70}")
        
        for i, filename in enumerate(files_to_process, 1):
            print(f"[{i}/{len(files_to_process)}] Processing {filename}...", end=' ')
            
            data = self.fetch_and_parse_in_memory(filename)
            if data:
                results.append(data)
                teams_str = ' vs '.join(data['teams']) if data['teams'] else 'N/A'
                print(f"✓ ({data['tournament']}: {teams_str})")
            else:
                print("✗")
            
            # Small delay
            if i < len(files_to_process):
                time.sleep(0.05)
        
        print(f"{'='*70}")
        print(f"Successfully processed: {len(results)}/{len(files_to_process)}")
        
        return results
    
    def filter_files(self, pattern: str = None, year: int = None, tournament: str = None) -> List[str]:
        """Filter files by various criteria"""
        if not self.xml_files:
            self.scrape_directory()
        
        filtered = self.xml_files
        
        if year:
            filtered = [f for f in filtered if str(year) in f]
        
        if tournament:
            filtered = [f for f in filtered if tournament.upper() in f.upper()]
        
        if pattern:
            filtered = [f for f in filtered if re.search(pattern, f, re.IGNORECASE)]
        
        return filtered


def main():
    """Example usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Fetch volleyball match XML files')
    parser.add_argument('--list', action='store_true', help='List all available files')
    parser.add_argument('--download', action='store_true', help='Download all files')
    parser.add_argument('--preview', action='store_true', help='Preview files in memory (no download)')
    parser.add_argument('--limit', type=int, help='Limit number of files to process')
    parser.add_argument('--year', type=int, help='Filter by year (e.g., 2024, 2025)')
    parser.add_argument('--tournament', type=str, help='Filter by tournament code (e.g., PVL2024A, PVL2025B)')
    parser.add_argument('--output', type=str, default='.', help='Output directory for downloads')
    parser.add_argument('--delay', type=float, default=0.1, help='Delay between downloads (seconds)')
    
    args = parser.parse_args()
    
    fetcher = XMLFileFetcher()
    
    # Scrape directory
    files = fetcher.scrape_directory()
    
    if not files:
        print("No files found or error occurred")
        return
    
    # Apply filters
    if args.year or args.tournament:
        files = fetcher.filter_files(year=args.year, tournament=args.tournament)
        fetcher.xml_files = files
        print(f"\nFiltered to {len(files)} files")
    
    # List files
    if args.list:
        print(f"\n{'='*70}")
        print("AVAILABLE FILES")
        print(f"{'='*70}")
        for i, filename in enumerate(files[:50], 1):  # Show first 50
            print(f"{i:3d}. {filename}")
        if len(files) > 50:
            print(f"... and {len(files) - 50} more files")
        print(f"\nTotal: {len(files)} files")
        return
    
    # Preview in memory
    if args.preview:
        results = fetcher.process_all_in_memory(limit=args.limit)
        
        print(f"\n{'='*70}")
        print("PREVIEW RESULTS")
        print(f"{'='*70}")
        
        tournaments = {}
        for result in results:
            tournament = result['tournament'] or 'Unknown'
            tournaments[tournament] = tournaments.get(tournament, 0) + 1
        
        print(f"\nMatches by tournament:")
        for tournament, count in sorted(tournaments.items()):
            print(f"  {tournament}: {count} matches")
        
        return
    
    # Download files
    if args.download:
        results = fetcher.download_all(
            output_dir=args.output,
            limit=args.limit,
            delay=args.delay
        )
        
        print(f"\n✓ Download complete!")
        print(f"Files saved to: {Path(args.output).absolute()}")
        return
    
    # Default: show help
    print("\nUsage:")
    print("  python fetch_all_matches.py --list                    # List all files")
    print("  python fetch_all_matches.py --preview --limit 10      # Preview 10 files")
    print("  python fetch_all_matches.py --download --limit 50     # Download 50 files")
    print("  python fetch_all_matches.py --download --year 2025    # Download 2025 files")
    print("  python fetch_all_matches.py --download --tournament PVL2024A")
    print("\nRun with --help for all options")


if __name__ == '__main__':
    main()
