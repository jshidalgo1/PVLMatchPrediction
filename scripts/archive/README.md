Archived scripts

This folder contains legacy, specialized, or one-off scripts moved out of the active pipeline to reduce clutter. They remain available for reference and can be restored if needed.

What moved here and why
- train_naive_bayes.py — legacy baseline trainer, superseded by multi-model pipelines
- train_xgboost.py — early trainer superseded by time-aware + calibrated trainers
- train_with_player_features.py — old combined-features trainer; replaced by multi-model scripts
- feature_engineering.py — early feature builder; replaced by feature_engineering_with_players.py with ELO
- complete_tournament_simulation.py — older end-to-end sim; replaced by multi_model_tournament_simulation*.py
- simulate_second_round.py — specialized second-round simulator; not used in main flows
- predict_tournament.py — stats-only champion estimator; kept for reference
- download_matches.py — sample downloader; replaced by curated dataset and batch processing
- fetch_all_matches.py — web scraper; not required for current pipeline
- fix_test_pvlr25_data.py — one-off DB fixer; kept for historical reference
- tournament_visualization.py — legacy visualization; modern summaries are in outputs/
- quickstart.py — interactive wizard; not part of current CLI
- validate_data.py — interactive validator; optional, not part of core flow

Restoring a script
- You can move any file back from scripts/archive/ to scripts/.
- If a script imports others from archive, move the dependencies together.

Current primary entry points
- scripts/multi_model_tournament_simulation.py (random baseline + stacking)
- scripts/multi_model_tournament_simulation_timeaware.py (chronological + calibrated + ELO + stacking)
- scripts/feature_engineering_with_players.py (features with ELO)
- run_simulation.py (runtime model selection and simulation)

Notes
- If you have automation that still invokes an archived script, update it to the current equivalents above.
- For large file hygiene, consider removing local virtual environments (.venv) and caches; see .gitignore at the repo root for recommended ignores.
