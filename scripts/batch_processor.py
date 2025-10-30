"""
Batch Processing Pipeline for Volleyball Match Data
Process all XML files and generate ML-ready dataset
"""

import glob
import json
from pathlib import Path
import sys

# Import our custom modules
from parse_volleyball_data import VolleyballDataParser
from feature_engineering import VolleyballFeatureEngineer
from database_manager import VolleyballDatabase
from config import XML_DIR_STR, DB_FILE_STR, CSV_DIR


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
    
    print("="*70)
    print("VOLLEYBALL DATA PROCESSING PIPELINE")
    print("="*70)
    
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
    from config import VOLLEYBALL_MATCHES_JSON
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
    print(f"\n[Step 4/5] Creating SQLite database...")
    db = VolleyballDatabase(DB_FILE_STR)
    db.connect()
    db.create_schema()
    db.load_from_json(json_file)
    
    db_summary = db.get_summary()
    print(f"\n  Database Summary:")
    print(f"  - Tournaments: {db_summary['total_tournaments']}")
    print(f"  - Teams: {db_summary['total_teams']}")
    print(f"  - Matches: {db_summary['total_matches']}")
    print(f"  - Players: {db_summary['total_players']}")
    
    db.close()
    
    # Step 5: Generate ML features
    print(f"\n[Step 5/5] Generating machine learning features...")
    
    with open(json_file, 'r') as f:
        matches_data = json.load(f)
    
    engineer = VolleyballFeatureEngineer(matches_data)
    X, y, metadata, feature_cols = engineer.get_feature_importance_ready_data()
    
    print(f"\n  Feature Engineering Summary:")
    print(f"  - Total samples: {len(X)}")
    print(f"  - Total features: {len(feature_cols)}")
    print(f"  - Class distribution: {y.value_counts().to_dict()}")
    
    # Save features
    import pandas as pd
    full_df = pd.concat([metadata, X, y], axis=1)
    full_df.to_csv(str(CSV_DIR / 'volleyball_features.csv'), index=False)
    X.to_csv(str(CSV_DIR / 'X_features.csv'), index=False)
    y.to_csv(str(CSV_DIR / 'y_target.csv'), index=False)
    
    print(f"\n  ✓ Features saved:")
    print(f"    - {CSV_DIR}/volleyball_features.csv (full dataset with metadata)")
    print(f"    - {CSV_DIR}/X_features.csv (features only)")
    print(f"    - {CSV_DIR}/y_target.csv (labels only)")
    
    # Final summary
    print(f"\n{'='*70}")
    print("PROCESSING COMPLETE!")
    print(f"{'='*70}")
    print(f"\nGenerated Files:")
    print(f"  1. volleyball_matches.json     - Raw parsed match data")
    print(f"  2. volleyball_data.db          - SQLite database")
    print(f"  3. volleyball_features.csv     - Complete feature dataset")
    print(f"  4. X_features.csv              - ML feature matrix")
    print(f"  5. y_target.csv                - ML target labels")
    
    print(f"\n📊 Your dataset is ready for XGBoost training!")
    print(f"\nNext step: Run train_xgboost.py to train your prediction model")
    
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
            print("\n✨ All done! You can now train your model with:")
            print("   python train_xgboost.py")
        else:
            print("\n⚠️  Processing incomplete. Please check the errors above.")
