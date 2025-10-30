#!/usr/bin/env python3
"""
Quick Start Guide for Volleyball AI Project
Interactive setup and training wizard
"""

import subprocess
import sys
from pathlib import Path
import glob


def print_header(text):
    """Print a formatted header"""
    print("\n" + "="*70)
    print(text.center(70))
    print("="*70 + "\n")


def check_requirements():
    """Check if required packages are installed"""
    print_header("CHECKING REQUIREMENTS")
    
    required = ['numpy', 'pandas', 'sklearn', 'xgboost', 'joblib']
    missing = []
    
    for package in required:
        try:
            if package == 'sklearn':
                __import__('sklearn')
            else:
                __import__(package)
            print(f"✓ {package}")
        except ImportError:
            print(f"✗ {package} (missing)")
            missing.append(package if package != 'sklearn' else 'scikit-learn')
    
    if missing:
        print(f"\n⚠️  Missing packages: {', '.join(missing)}")
        print("\nInstall with: pip install " + ' '.join(missing))
        return False
    
    print("\n✓ All requirements satisfied!")
    return True


def check_xml_files():
    """Check if XML files exist"""
    print_header("CHECKING DATA FILES")
    
    xml_files = glob.glob('*.xml')
    
    if not xml_files:
        print("❌ No XML files found in current directory")
        print("\nYou need volleyball match XML files to proceed.")
        print("\nOptions:")
        print("  1. Download sample matches:")
        print("     python download_matches.py")
        print("\n  2. Download manually:")
        print("     curl -o match.xml 'https://dashboard.pvl.ph/assets/match_results/xml/MATCH_ID.xml'")
        return False
    
    print(f"✓ Found {len(xml_files)} XML file(s):")
    for i, file in enumerate(xml_files[:5], 1):
        print(f"  {i}. {Path(file).name}")
    if len(xml_files) > 5:
        print(f"  ... and {len(xml_files) - 5} more")
    
    return True


def run_batch_processor():
    """Run the batch processor"""
    print_header("PROCESSING MATCH DATA")
    
    print("Running batch processor to parse XML files and generate features...\n")
    
    try:
        result = subprocess.run(
            [sys.executable, 'batch_processor.py'],
            check=True,
            text=True
        )
        return True
    except subprocess.CalledProcessError:
        print("\n❌ Batch processing failed")
        return False
    except FileNotFoundError:
        print("\n❌ batch_processor.py not found")
        return False


def check_processed_data():
    """Check if data has been processed"""
    required_files = ['X_features.csv', 'y_target.csv']
    
    for file in required_files:
        if not Path(file).exists():
            return False
    
    return True


def run_training():
    """Run the model training"""
    print_header("TRAINING XGBOOST MODEL")
    
    print("Training XGBoost model on processed data...\n")
    
    try:
        result = subprocess.run(
            [sys.executable, 'train_xgboost.py'],
            check=True,
            text=True
        )
        return True
    except subprocess.CalledProcessError:
        print("\n❌ Training failed")
        return False
    except FileNotFoundError:
        print("\n❌ train_xgboost.py not found")
        return False


def show_next_steps():
    """Show what to do next"""
    print_header("NEXT STEPS")
    
    print("Your volleyball prediction system is ready! 🏐\n")
    print("Here's what you can do next:\n")
    
    print("1. ADD MORE DATA (Recommended)")
    print("   - Download more match XML files")
    print("   - More data = better predictions")
    print("   - Run: python download_matches.py\n")
    
    print("2. IMPROVE THE MODEL")
    print("   - Edit train_xgboost.py to tune hyperparameters")
    print("   - Add custom features in feature_engineering.py")
    print("   - Try different ML algorithms\n")
    
    print("3. MAKE PREDICTIONS")
    print("   - Load the trained model: volleyball_predictor.pkl")
    print("   - Use it to predict new matches")
    print("   - See README.md for code examples\n")
    
    print("4. EXPLORE THE DATA")
    print("   - volleyball_features.csv - Full dataset")
    print("   - volleyball_data.db - SQLite database")
    print("   - feature_importance.csv - Most important features\n")
    
    print("📚 Read README.md for detailed documentation")
    print("="*70)


def main():
    """Main quick start flow"""
    print("""
    ╔══════════════════════════════════════════════════════════════════╗
    ║                                                                  ║
    ║           VOLLEYBALL AI MATCH PREDICTION SYSTEM                  ║
    ║                   Quick Start Wizard                             ║
    ║                                                                  ║
    ╚══════════════════════════════════════════════════════════════════╝
    """)
    
    # Step 1: Check requirements
    if not check_requirements():
        print("\nPlease install missing packages and run this script again.")
        sys.exit(1)
    
    # Step 2: Check XML files
    if not check_xml_files():
        print("\nPlease add XML match files and run this script again.")
        sys.exit(1)
    
    # Step 3: Check if already processed
    if check_processed_data():
        print("\n✓ Data already processed (X_features.csv, y_target.csv found)")
        response = input("\nReprocess data anyway? (y/N): ").strip().lower()
        if response == 'y':
            if not run_batch_processor():
                sys.exit(1)
    else:
        # Run batch processor
        if not run_batch_processor():
            sys.exit(1)
    
    # Step 4: Check if we have enough data
    try:
        import pandas as pd
        X = pd.read_csv('X_features.csv')
        
        if len(X) < 2:
            print("\n⚠️  WARNING: Only 1 match found!")
            print("You need at least 2 matches to train a model.")
            print("\nPlease download more matches:")
            print("  python download_matches.py")
            print("\nThen rerun this script.")
            sys.exit(1)
        
        if len(X) < 10:
            print(f"\n⚠️  WARNING: Only {len(X)} matches found")
            print("Model accuracy will be limited with small dataset.")
            print("Recommended: 20+ matches for better predictions\n")
            
            response = input("Continue training anyway? (y/N): ").strip().lower()
            if response != 'y':
                print("\nDownload more matches first:")
                print("  python download_matches.py")
                sys.exit(0)
    except Exception as e:
        print(f"\n❌ Error reading features: {e}")
        sys.exit(1)
    
    # Step 5: Train model
    if not run_training():
        sys.exit(1)
    
    # Step 6: Show next steps
    show_next_steps()


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nCancelled by user.")
        sys.exit(0)
