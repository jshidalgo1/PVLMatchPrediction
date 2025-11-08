#!/usr/bin/env python3
"""
Basic Tournament Simulation Example

This script demonstrates how to run a simple tournament simulation
using a trained model.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from scripts.simulate_tournament import main


def basic_simulation():
    """Run a basic tournament simulation with the recommended model."""
    
    print("=" * 70)
    print("BASIC TOURNAMENT SIMULATION EXAMPLE")
    print("=" * 70)
    print()
    
    # Path to the recommended model
    model_path = project_root / "models" / "calibrated_xgboost_with_players.pkl"
    
    # Check if model exists
    if not model_path.exists():
        print(f"❌ Model not found: {model_path}")
        print()
        print("Please train a model first:")
        print("  python scripts/train_xgboost_with_players.py")
        return
    
    print(f"📊 Using model: {model_path.name}")
    print()
    
    # Run simulation without saving outputs
    main(
        model_path=str(model_path),
        save_outputs=False,
        keep_latest=0,
        champion_analysis=True
    )
    
    print()
    print("=" * 70)
    print("SIMULATION COMPLETE!")
    print("=" * 70)
    print()
    print("💡 Tips:")
    print("  - Add --save_outputs to persist results")
    print("  - Use --keep_latest to manage old outputs")
    print("  - Try different models from models/ directory")
    print()


if __name__ == "__main__":
    basic_simulation()
