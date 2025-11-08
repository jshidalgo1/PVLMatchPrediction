#!/usr/bin/env python3
"""
Unified simulation entrypoint.

Delegates to `scripts.simulate_tournament.main(model_path=...)`.
Usage:
    python run_simulation.py --model models/calibrated_xgboost_with_players.pkl

If --model is omitted, simulate_tournament will choose a sensible default
(calibrated_xgboost_with_players.pkl if present, else BEST_MODEL).
"""

import argparse
import inspect

# Import tournament simulation module
from scripts import simulate_tournament as sim


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
    if not hasattr(sim, "main"):
        raise SystemExit("simulate_tournament.main not found")
    # Direct invocation; simulate_tournament handles default selection
    sim.main(model_path=args.model)
