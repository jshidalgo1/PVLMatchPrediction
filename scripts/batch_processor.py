"""
Batch Processing Pipeline for Volleyball Match Data (Player-Aware)
Processes all XML files, loads into SQLite (including player + lineup stats),
and generates ML-ready feature matrices with player-level + ELO features.

Updates:
- Switched to PlayerEnhancedFeatureExtractor (replaces legacy VolleyballFeatureEngineer)
- Saves player-inclusive CSVs (volleyball_features_with_players, X_features_with_players, y_target_with_players)
"""

import glob
import json
from pathlib import Path
import sys

# Import our custom modules (player-aware versions)
from parse_volleyball_data import VolleyballDataParser
from feature_engineering_with_players import PlayerEnhancedFeatureExtractor
from database_manager import VolleyballDatabase
from config import (
    XML_DIR_STR,
    DB_FILE_STR,
    DB_BACKUP,
    CSV_DIR,
    VOLLEYBALL_MATCHES_JSON,
    VOLLEYBALL_FEATURES,
    X_FEATURES,
    Y_TARGET,
    VOLLEYBALL_FEATURES_STR,
    X_FEATURES_STR,
    Y_TARGET_STR,
    ensure_directories,
    canonicalize_team_code,
)


def process_all_xml_files(xml_directory=None, pattern='*.xml'):
    """
    Complete pipeline to process all XML files
    
    Steps:
    1. Parse all XML files
    2. Save to JSON
    3. Create SQLite database
    4. Generate ML features
    5. Prepare for XGBoost training
    """
    
    print("="*80)
    print("VOLLEYBALL DATA PROCESSING PIPELINE (PLAYER-AWARE)")
    print("="*80)

    # Ensure folder structure exists
    ensure_directories()
    
    # Use configured XML directory if not specified
    if xml_directory is None:
        xml_directory = XML_DIR_STR
    
    # Step 1: Find XML files
    print(f"\n[Step 1/5] Finding XML files...")
    xml_pattern = f"{xml_directory}/{pattern}"
    xml_files = glob.glob(xml_pattern)
    
    if not xml_files:
        print(f"❌ No XML files found matching pattern: {xml_pattern}")
        print("\nPlease ensure your XML files are in the correct directory.")
        return False
    
    print(f"✓ Found {len(xml_files)} XML file(s)")
    for i, file in enumerate(xml_files, 1):
        print(f"  {i}. {Path(file).name}")
    
    # Step 2: Parse XML files
    print(f"\n[Step 2/5] Parsing XML files...")
    parser = VolleyballDataParser()
    parsed_data = parser.parse_multiple_files(xml_files)
    
    if not parsed_data:
        print("❌ No data was parsed successfully")
        return False
    
    print(f"✓ Successfully parsed {len(parsed_data)} match(es)")
    
    # Step 3: Save to JSON
    print(f"\n[Step 3/5] Saving to JSON...")
    json_file = str(VOLLEYBALL_MATCHES_JSON)
    parser.save_to_json(json_file)
    
    # Print summary
    summary = parser.get_feature_summary()
    print(f"\n  Match Summary:")
    print(f"  - Total Matches: {summary['total_matches']}")
    print(f"  - Tournaments: {', '.join(summary['tournaments'])}")
    print(f"  - Teams: {', '.join(summary['teams'])}")
    print(f"  - Statistics Types: {len(summary['stat_types'])}")
    
    # Step 4: Create database
    print(f"\n[Step 4/5] Creating SQLite database (with player + lineup stats)...")
    # Reset database to ensure duplicates are removed cleanly
    db_path = Path(DB_FILE_STR)
    if db_path.exists():
        try:
            # Backup then remove
            backup_path = Path(DB_BACKUP)
            if backup_path.exists():
                backup_path.unlink(missing_ok=True)
            db_path.replace(backup_path)
            print(f"  Previous DB backed up to: {backup_path}")
        except Exception as e:
            print(f"  Warning: Could not backup existing DB: {e}")
        try:
            # Remove any existing DB to fully rebuild
            if db_path.exists():
                db_path.unlink()
        except Exception as e:
            print(f"  Warning: Could not remove existing DB: {e}")
    db = VolleyballDatabase(DB_FILE_STR)
    db.connect()
    db.create_schema()
    # Load JSON but filter out known duplicate intra-tournament rematches if necessary
    # Specifically: retain only earliest AKA vs CCS match in TEST_PVLR25 (match_id 291) and drop later duplicate 507
    # This prevents inflated feature counts for a single pairing within the same preliminary tournament.
    with open(json_file, 'r') as jf:
        raw_matches = json.load(jf)
    filtered_matches = []
    seen_test_pair = False
    for m in raw_matches:
        tcode = m.get('tournament_code') or m.get('tournament', {}).get('code')
        ta = m.get('team_a_code') or m.get('team_a', {}).get('code')
        tb = m.get('team_b_code') or m.get('team_b', {}).get('code')
        # Canonicalize codes to catch duplicates across aliases (e.g., CHD->CSS)
        if ta:
            ta = canonicalize_team_code(ta)
        if tb:
            tb = canonicalize_team_code(tb)
        # Normalize order
        pair = tuple(sorted([ta, tb])) if ta and tb else None
        if tcode == 'TEST_PVLR25' and pair == ('AKA', 'CCS'):
            if seen_test_pair:
                # Skip duplicate
                continue
            seen_test_pair = True
        filtered_matches.append(m)
    # Overwrite JSON to reflect filtered set for transparency
    with open(json_file, 'w') as jf:
        json.dump(filtered_matches, jf, indent=2)
    db.load_from_json(json_file)
    
    db_summary = db.get_summary()
    print(f"\n  Database Summary:")
    print(f"  - Tournaments: {db_summary['total_tournaments']}")
    print(f"  - Teams: {db_summary['total_teams']}")
    print(f"  - Matches: {db_summary['total_matches']}")
    print(f"  - Players: {db_summary['total_players']}")
    if seen_test_pair:
        print("  - Duplicate AKA-CCS intra TEST_PVLR25 removed (kept earliest encounter)")
    
    db.close()
    
    # Step 5: Generate ML features
    print(f"\n[Step 5/5] Generating machine learning features (team + player + ELO)...")
    
    with open(json_file, 'r') as f:
        matches_data = json.load(f)
    
    # Legacy feature engineer removed; use PlayerEnhancedFeatureExtractor directly from DB
    extractor = PlayerEnhancedFeatureExtractor(DB_FILE_STR)
    features_df = extractor.extract_features()

    # Separate metadata / features / target
    metadata_cols = ['match_id', 'team_a_id', 'team_b_id', 'tournament_id']
    target_col = 'team_a_wins'
    feature_cols = [c for c in features_df.columns if c not in metadata_cols + [target_col]]
    metadata = features_df[metadata_cols]
    X = features_df[feature_cols]
    import pandas as pd  # ensure pandas import before y usage (already imported above but safe)
    y = features_df[target_col].astype(int)
    
    print(f"\n  Feature Engineering Summary (Player-Aware):")
    print(f"  - Total samples: {len(X)}")
    print(f"  - Total features: {len(feature_cols)}")
    print(f"  - Player-related features: {len([c for c in feature_cols if c.startswith('team_a_') or c.startswith('team_b_')]) - 4 - 12}")
    print(f"  - Class distribution: {y.value_counts().to_dict()}")
    
    # Save features
    import pandas as pd
    full_df = pd.concat([metadata, X, y.rename(target_col)], axis=1)
    full_df.to_csv(VOLLEYBALL_FEATURES_STR, index=False)
    X.to_csv(X_FEATURES_STR, index=False)
    y.to_csv(Y_TARGET_STR, index=False)
    
    print(f"\n  ✓ Features saved:")
    print(f"    - {VOLLEYBALL_FEATURES} (full dataset with metadata + target)")
    print(f"    - {X_FEATURES} (features only)")
    print(f"    - {Y_TARGET} (labels only)")
    
    # Final summary
    print(f"\n{'='*80}")
    print("PLAYER-AWARE PROCESSING COMPLETE!")
    print(f"{'='*80}")
    print(f"\nGenerated Files:")
    print(f"  1. {VOLLEYBALL_MATCHES_JSON.name}                - Raw parsed match data")
    print(f"  2. {Path(DB_FILE_STR).name}                     - SQLite database (teams + players + lineups)")
    print(f"  3. {Path(VOLLEYBALL_FEATURES_STR).name}         - Complete feature dataset (with players)")
    print(f"  4. {Path(X_FEATURES_STR).name}                  - ML feature matrix")
    print(f"  5. {Path(Y_TARGET_STR).name}                    - ML target labels")

    print(f"\n📊 Your player-aware dataset is ready for model training!")
    print(f"\nNext step: Run train_xgboost_with_players.py to train the calibrated model")
    
    return True


def download_more_matches(base_url_pattern: str = None):
    """
    Helper function to download more match XML files
    
    Example usage:
        download_more_matches('https://dashboard.pvl.ph/assets/match_results/xml/')
    """
    print("\n" + "="*70)
    print("DOWNLOAD MORE MATCH DATA")
    print("="*70)
    
    if base_url_pattern is None:
        print("\nTo download more matches, you can use curl commands like:")
        print("\n  curl -o match_file.xml 'https://dashboard.pvl.ph/assets/match_results/xml/MATCH_ID.xml'")
        print("\nOr add the download logic to this function with specific match IDs.")
    else:
        print(f"Base URL: {base_url_pattern}")
        print("Download logic would go here...")


def show_usage():
    """Display usage instructions"""
    print("\n" + "="*70)
    print("VOLLEYBALL AI PROJECT - DATA PROCESSING PIPELINE")
    print("="*70)
    print("\nUsage:")
    print("  python batch_processor.py")
    print("\nThis script will:")
    print("  1. Find all XML files in the current directory")
    print("  2. Parse match data from XML files")
    print("  3. Create a JSON database")
    print("  4. Create a SQLite database")
    print("  5. Generate ML features for XGBoost")
    print("\nMake sure you have XML match files in the current directory!")
    print("="*70)


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] in ['--help', '-h']:
        show_usage()
    else:
        success = process_all_xml_files()
        
        if success:
            print("\n✨ All done! You can now train your player-aware model with:")
            print("   python train_xgboost_with_players.py")
        else:
            print("\n⚠️  Processing incomplete. Please check the errors above.")
