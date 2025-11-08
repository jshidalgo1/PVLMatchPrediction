# Metrics Comparison (20251108_022111 UTC)


## Holdout (time-aware 20%)

| Model | Accuracy | LogLoss | Brier | AUC | ECE | MCE |

|---|---:|---:|---:|---:|---:|---:|

| best_model_with_players.pkl | 0.7647 | 0.5606 | 0.1879 | 0.8329 | 0.1326 | 0.2304 |

| calibrated_xgboost_with_players.pkl | 0.7353 | 0.5620 | 0.1886 | 0.8017 | 0.1646 | 0.3462 |


## Delta (B - A)

| Metric | Delta |

|---|---:|

| Accuracy | -0.0294 |

| LogLoss | +0.0014 |

| Brier | +0.0007 |

| AUC | -0.0312 |

| ECE | +0.0319 |

| MCE | +0.1158 |
