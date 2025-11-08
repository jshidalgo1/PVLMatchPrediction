#!/usr/bin/env python3
"""
Model Comparison Example

This script demonstrates how to compare multiple trained models
on the same holdout dataset.
"""

import sys
from pathlib import Path
import subprocess

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def compare_models():
    """Compare two model artifacts and display metrics."""
    
    print("=" * 70)
    print("MODEL COMPARISON EXAMPLE")
    print("=" * 70)
    print()
    
    models_dir = project_root / "models"
    
    # Define models to compare
    model_a = models_dir / "best_model_with_players_timeaware.pkl"
    model_b = models_dir / "calibrated_xgboost_with_players.pkl"
    
    # Check if models exist
    if not model_a.exists():
        print(f"❌ Model A not found: {model_a}")
        return
    
    if not model_b.exists():
        print(f"❌ Model B not found: {model_b}")
        return
    
    print(f"📊 Model A: {model_a.name}")
    print(f"📊 Model B: {model_b.name}")
    print()
    print("Running comparison...")
    print()
    
    # Run comparison script
    compare_script = project_root / "scripts" / "compare_metrics.py"
    
    result = subprocess.run(
        [sys.executable, str(compare_script), str(model_a), str(model_b)],
        cwd=project_root,
        capture_output=False,
        text=True
    )
    
    if result.returncode == 0:
        print()
        print("=" * 70)
        print("COMPARISON COMPLETE!")
        print("=" * 70)
        print()
        print("📁 Check outputs/ directory for:")
        print("  - metrics_comparison_*.md (formatted report)")
        print("  - metrics_comparison_*.json (structured data)")
    else:
        print()
        print("❌ Comparison failed. Check error messages above.")
    
    print()


if __name__ == "__main__":
    compare_models()
