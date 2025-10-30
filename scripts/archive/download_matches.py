"""
Download Multiple PVL Match XML Files
Helper script to download volleyball match data from PVL dashboard
"""

import subprocess
import sys
from pathlib import Path


# Sample match IDs from PVL (you can add more)
SAMPLE_MATCHES = [
    'PVL2023A-W01-AKAvCMF-XML',
    # Add more match IDs here as you find them
    # Example format: 'PVL2023A-W02-XXXvYYY-XML',
]

BASE_URL = 'https://dashboard.pvl.ph/assets/match_results/xml/'


def download_match(match_id: str, output_dir: str = '.') -> bool:
    """Download a single match XML file"""
    if not match_id.endswith('.xml'):
        match_id = match_id + '.xml'
    
    url = f"{BASE_URL}{match_id}"
    output_path = Path(output_dir) / match_id
    
    print(f"Downloading {match_id}...", end=' ')
    
    try:
        result = subprocess.run(
            ['curl', '-o', str(output_path), url],
            capture_output=True,
            text=True,
            check=True
        )
        
        # Check if file was downloaded successfully
        if output_path.exists() and output_path.stat().st_size > 0:
            print(f"✓ ({output_path.stat().st_size} bytes)")
            return True
        else:
            print("✗ (empty file)")
            return False
            
    except subprocess.CalledProcessError as e:
        print(f"✗ (error: {e})")
        return False
    except Exception as e:
        print(f"✗ ({str(e)})")
        return False


def download_all_samples():
    """Download all sample matches"""
    print("="*70)
    print("DOWNLOADING PVL MATCH DATA")
    print("="*70)
    print(f"\nDownloading {len(SAMPLE_MATCHES)} match file(s)...\n")
    
    successful = 0
    failed = 0
    
    for match_id in SAMPLE_MATCHES:
        if download_match(match_id):
            successful += 1
        else:
            failed += 1
    
    print(f"\n{'='*70}")
    print(f"Download Summary: {successful} successful, {failed} failed")
    print(f"{'='*70}")
    
    if successful > 0:
        print(f"\n✓ Downloaded {successful} match file(s)")
        print("\nNext step: Run batch_processor.py to process all matches")
        print("  python batch_processor.py")
    else:
        print("\n⚠️  No files were downloaded successfully")


def download_custom_match():
    """Interactive mode to download a custom match"""
    print("\n" + "="*70)
    print("DOWNLOAD CUSTOM MATCH")
    print("="*70)
    print("\nEnter the match ID (e.g., PVL2023A-W01-AKAvCMF-XML)")
    print("Or press Enter to cancel")
    
    match_id = input("\nMatch ID: ").strip()
    
    if not match_id:
        print("Cancelled.")
        return
    
    download_match(match_id)


def show_help():
    """Show usage instructions"""
    print("\n" + "="*70)
    print("PVL MATCH DOWNLOADER")
    print("="*70)
    print("\nUsage:")
    print("  python download_matches.py                 # Download sample matches")
    print("  python download_matches.py --custom        # Enter custom match ID")
    print("  python download_matches.py --help          # Show this help")
    print("\nMatch ID Format:")
    print("  PVLYYYYC-WXX-TTTvTTT-XML")
    print("  Where:")
    print("    YYYY = Year (e.g., 2023)")
    print("    C    = Conference code (e.g., A for All-Filipino)")
    print("    WXX  = Week number (e.g., W01)")
    print("    TTT  = Team codes (e.g., AKAvCMF)")
    print("\nExamples:")
    print("  PVL2023A-W01-AKAvCMF-XML")
    print("  PVL2023A-W02-PTNvCIG-XML")
    print("\nTo find more match IDs:")
    print("  Visit: https://dashboard.pvl.ph/")
    print("  Look for match results and inspect the XML links")
    print("="*70)


if __name__ == '__main__':
    if len(sys.argv) > 1:
        if sys.argv[1] in ['--help', '-h']:
            show_help()
        elif sys.argv[1] in ['--custom', '-c']:
            download_custom_match()
        else:
            print(f"Unknown option: {sys.argv[1]}")
            print("Use --help for usage information")
    else:
        download_all_samples()
