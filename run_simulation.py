#!/usr/bin/env python3
"""
Volleyball Tournament Simulator - Main Entry Point
Run simulations with proper paths from project root
"""

import sys
from pathlib import Path

# Add scripts directory to path
scripts_dir = Path(__file__).parent / "scripts"
sys.path.insert(0, str(scripts_dir))

# Import and run the simulation
from simulate_remaining_matches import main

if __name__ == "__main__":
    main()
