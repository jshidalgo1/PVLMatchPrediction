"""
Configuration file for Volleyball AI Project
Centralizes all file paths to work with the new organized structure
"""
import os
from pathlib import Path

# Get the project root directory (parent of scripts folder)
PROJECT_ROOT = Path(__file__).parent.parent
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
DOCS_DIR = PROJECT_ROOT / "docs"
CERTIFICATES_DIR = PROJECT_ROOT / "certificates"

# Data subdirectories
XML_DIR = DATA_DIR / "xml_files"
CSV_DIR = DATA_DIR / "csv_files"
DB_DIR = DATA_DIR / "databases"

# Database files
DB_FILE = DB_DIR / "volleyball_data.db"
DB_BACKUP = DB_DIR / "volleyball_data_backup.db"

# CSV/JSON data files
VOLLEYBALL_FEATURES = CSV_DIR / "volleyball_features_with_players.csv"
X_FEATURES = CSV_DIR / "X_features_with_players.csv"
Y_TARGET = CSV_DIR / "y_target_with_players.csv"
FEATURE_IMPORTANCE = CSV_DIR / "feature_importance_with_players.csv"
VOLLEYBALL_MATCHES_JSON = DATA_DIR / "volleyball_matches.json"

# Model files
BEST_MODEL = MODELS_DIR / "best_model_with_players.pkl"
CATBOOST_INFO = MODELS_DIR / "catboost_info"

# Output files
POOL_STANDINGS = OUTPUTS_DIR / "pool_standings.png"
TOURNAMENT_BRACKET = OUTPUTS_DIR / "tournament_bracket.png"
SIMULATION_OUTPUT = OUTPUTS_DIR / "simulation_output.txt"
REQUIREMENTS = OUTPUTS_DIR / "requirements.txt"

# Create directories if they don't exist
def ensure_directories():
    """Create all necessary directories if they don't exist"""
    for directory in [DATA_DIR, XML_DIR, CSV_DIR, DB_DIR, MODELS_DIR, 
                      OUTPUTS_DIR, DOCS_DIR, CERTIFICATES_DIR]:
        directory.mkdir(parents=True, exist_ok=True)

# Convert Path objects to strings for backward compatibility
def get_path_str(path):
    """Convert Path object to string"""
    return str(path)

# Backward compatibility - string versions
DB_FILE_STR = str(DB_FILE)
XML_DIR_STR = str(XML_DIR)
CSV_DIR_STR = str(CSV_DIR)
BEST_MODEL_STR = str(BEST_MODEL)
X_FEATURES_STR = str(X_FEATURES)
Y_TARGET_STR = str(Y_TARGET)
VOLLEYBALL_FEATURES_STR = str(VOLLEYBALL_FEATURES)
FEATURE_IMPORTANCE_STR = str(FEATURE_IMPORTANCE)

if __name__ == "__main__":
    print("Volleyball AI Project - Configuration")
    print("=" * 50)
    print(f"Project Root: {PROJECT_ROOT}")
    print(f"\nDirectories:")
    print(f"  Data:         {DATA_DIR}")
    print(f"  XML Files:    {XML_DIR}")
    print(f"  CSV Files:    {CSV_DIR}")
    print(f"  Databases:    {DB_DIR}")
    print(f"  Models:       {MODELS_DIR}")
    print(f"  Outputs:      {OUTPUTS_DIR}")
    print(f"  Scripts:      {SCRIPTS_DIR}")
    print(f"\nKey Files:")
    print(f"  Database:     {DB_FILE}")
    print(f"  Best Model:   {BEST_MODEL}")
    print(f"  Features:     {X_FEATURES}")
    print(f"  Target:       {Y_TARGET}")
    
    # Check what exists
    print(f"\nFile Status:")
    print(f"  Database exists:      {DB_FILE.exists()}")
    print(f"  Model exists:         {BEST_MODEL.exists()}")
    print(f"  Features exist:       {X_FEATURES.exists()}")
    print(f"  XML files count:      {len(list(XML_DIR.glob('*.xml'))) if XML_DIR.exists() else 0}")
