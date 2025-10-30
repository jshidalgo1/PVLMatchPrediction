#!/usr/bin/env python3
"""
Volleyball Tournament Simulator - Main Entry Point
Adds optional --model PATH to choose which model file to load.
Defaults to config.BEST_MODEL if not provided.
"""

import sys
import argparse
from pathlib import Path

# Add scripts directory to path
scripts_dir = Path(__file__).parent / "scripts"
sys.path.insert(0, str(scripts_dir))

# Import simulation module
import simulate_remaining_matches as srm


def parse_args():
    parser = argparse.ArgumentParser(description="Run volleyball tournament simulation")
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Path to a specific model .pkl to use (overrides default)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    # Call simulate with optional override
    if hasattr(srm, "main"):
        try:
            srm.main(model_path=args.model)
        except TypeError:
            # Backward compatibility if main doesn't accept kwarg
            srm.main()
    else:
        raise SystemExit("simulate_remaining_matches.main not found")
