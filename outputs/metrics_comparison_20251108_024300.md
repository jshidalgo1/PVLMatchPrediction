# Metrics Comparison (20251108_024300 UTC)


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


## Calibration Buckets (quantile, 10 bins)

| Bin | A: mean_pred | A: frac_pos | B: mean_pred | B: frac_pos |

|---:|---:|---:|---:|---:|

| 1 | 0.294 | 0.364 | 0.176 | 0.182 |

| 2 | 0.352 | 0.100 | 0.201 | 0.100 |

| 3 | 0.375 | 0.500 | 0.264 | 0.400 |

| 4 | 0.432 | 0.300 | 0.348 | 0.100 |

| 5 | 0.482 | 0.400 | 0.458 | 0.400 |

| 6 | 0.525 | 0.300 | 0.581 | 0.800 |

| 7 | 0.573 | 0.400 | 0.646 | 0.300 |

| 8 | 0.627 | 0.600 | 0.700 | 0.600 |

| 9 | 0.679 | 0.800 | 0.727 | 0.900 |

| 10 | 0.730 | 1.000 | 0.736 | 1.000 |
