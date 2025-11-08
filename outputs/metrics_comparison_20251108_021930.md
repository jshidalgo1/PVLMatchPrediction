# Metrics Comparison (20251108_021930 UTC)


## Holdout (time-aware 20%)

| Model | Accuracy | LogLoss | Brier | AUC | ECE | MCE |

|---|---:|---:|---:|---:|---:|---:|

| best_model_with_players_timeaware.pkl | 0.6471 | 0.6176 | 0.2139 | 0.7305 | 0.1483 | 0.2699 |

| calibrated_xgboost_with_players.pkl | 0.7353 | 0.5620 | 0.1886 | 0.8017 | 0.1646 | 0.3462 |


## Delta (B - A)

| Metric | Delta |

|---|---:|

| Accuracy | +0.0882 |

| LogLoss | -0.0555 |

| Brier | -0.0254 |

| AUC | +0.0712 |

| ECE | +0.0163 |

| MCE | +0.0763 |
