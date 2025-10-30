#!/usr/bin/env python3
"""
Volleyball Tournament Simulator - Main Entry Point
Adds optional --model PATH to choose which model file to load.
Defaults to config.BEST_MODEL if not provided.
"""

import argparse
import inspect

# Import simulation module from package
from scripts import simulate_remaining_matches as srm


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
        sig = None
        try:
            sig = inspect.signature(srm.main)
        except (ValueError, TypeError):
            # If signature can't be inspected, fall back to plain call
            srm.main()
        else:
            if any(p.kind in (p.KEYWORD_ONLY, p.POSITIONAL_OR_KEYWORD) and name == "model_path"
                   for name, p in sig.parameters.items()):
                srm.main(model_path=args.model)
            else:
                srm.main()
    else:
        raise SystemExit("simulate_remaining_matches.main not found")
