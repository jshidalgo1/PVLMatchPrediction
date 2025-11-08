"""
Compare two model artifacts on the same time-aware split and save a markdown report.

Usage:
  python scripts/compare_metrics.py --model_a models/best_model_with_players_timeaware.pkl \
                                    --model_b models/calibrated_xgboost_with_players.pkl

Outputs:
  outputs/metrics_comparison_<timestamp>.md
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Tuple
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    log_loss,
    brier_score_loss,
    roc_auc_score,
)
from sklearn.calibration import calibration_curve
import json

# Local config
try:
    from .config import VOLLEYBALL_FEATURES_STR, CSV_DIR_STR, OUTPUTS_DIR
except Exception:
    from config import VOLLEYBALL_FEATURES_STR, CSV_DIR_STR, OUTPUTS_DIR


def _safe_read_csv(primary: str, fallback: str | None = None) -> pd.DataFrame:
    p = Path(primary)
    if p.exists():
        return pd.read_csv(p)
    if fallback:
        fb = Path(fallback)
        if fb.exists():
            return pd.read_csv(fb)
    raise FileNotFoundError(f"Could not find CSV at {primary} or fallback {fallback}")


def _prepare_time_aware_split(df: pd.DataFrame):
    metadata_cols = ["match_id", "team_a_id", "team_b_id", "tournament_id"]
    target_col = "team_a_wins"
    feature_cols = [c for c in df.columns if c not in metadata_cols + [target_col]]
    df = df.sort_values("match_id").reset_index(drop=True)
    X = df[feature_cols]
    y = df[target_col].values.ravel()
    n = len(X)
    test_start = int(n * 0.8)
    X_train_full, y_train_full = X.iloc[:test_start], y[:test_start]
    cal_start = int(len(X_train_full) * 0.9)
    X_cal, y_cal = X_train_full.iloc[cal_start:], y_train_full[cal_start:]
    X_test, y_test = X.iloc[test_start:], y[test_start:]
    return feature_cols, (X_cal, y_cal, X_test, y_test)


def _align_features(X: pd.DataFrame, feature_names: list[str]) -> pd.DataFrame:
    X_aligned = X.copy()
    for col in feature_names:
        if col not in X_aligned.columns:
            X_aligned[col] = 0
    return X_aligned[feature_names]


@dataclass
class EvalResult:
    name: str
    acc: float
    logloss: float
    brier: float
    auc: float
    ece: float
    mce: float


def _ece_mce(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> Tuple[float, float]:
    bins = np.quantile(y_prob, np.linspace(0, 1, n_bins + 1))
    bins[0], bins[-1] = 0.0, 1.0
    ece = 0.0
    mce = 0.0
    total = len(y_true)
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (y_prob >= lo) & (y_prob <= hi) if i == n_bins - 1 else (y_prob >= lo) & (y_prob < hi)
        if not np.any(mask):
            continue
        conf = float(np.mean(y_prob[mask]))
        acc = float(np.mean(y_true[mask]))
        gap = abs(acc - conf)
        ece += (np.sum(mask) / total) * gap
        mce = max(mce, gap)
    return float(ece), float(mce)


def _evaluate_model(artifact_path: Path, X_test: pd.DataFrame, y_test: np.ndarray) -> EvalResult:
    art = joblib.load(artifact_path)
    if isinstance(art, dict) and 'model' in art:
        model = art['model']
        feat_names = art.get('feature_names', list(X_test.columns))
        X_eval = _align_features(X_test, feat_names)
        y_proba = model.predict_proba(X_eval)[:, 1]
        y_pred = (y_proba >= 0.5).astype(int)
    else:
        # assume a scikit estimator
        model = art
        try:
            feat_names = getattr(model, 'feature_names_in_', list(X_test.columns))
        except Exception:
            feat_names = list(X_test.columns)
        X_eval = _align_features(X_test, list(feat_names))
        y_proba = model.predict_proba(X_eval)[:, 1]
        y_pred = (y_proba >= 0.5).astype(int)

    acc = accuracy_score(y_test, y_pred)
    ll = log_loss(y_test, y_proba, labels=[0, 1])
    brier = brier_score_loss(y_test, y_proba)
    auc = roc_auc_score(y_test, y_proba)
    ece, mce = _ece_mce(y_test, y_proba, 10)
    return EvalResult(name=artifact_path.name, acc=acc, logloss=ll, brier=brier, auc=auc, ece=ece, mce=mce)


def main():
    parser = argparse.ArgumentParser(description="Compare two model artifacts on time-aware holdout")
    parser.add_argument('--model_a', type=str, required=False, default=str(Path('models') / 'best_model_with_players_timeaware.pkl'))
    parser.add_argument('--model_b', type=str, required=False, default=str(Path('models') / 'calibrated_xgboost_with_players.pkl'))
    args = parser.parse_args()

    # Load features
    features_path = VOLLEYBALL_FEATURES_STR
    fallback_features_path = str(Path(CSV_DIR_STR) / 'volleyball_features_with_players.csv')
    df = _safe_read_csv(features_path, fallback_features_path)
    feature_cols, (X_cal, y_cal, X_test, y_test) = _prepare_time_aware_split(df)

    # Evaluate both
    res_a = _evaluate_model(Path(args.model_a), X_test, y_test)
    res_b = _evaluate_model(Path(args.model_b), X_test, y_test)

    # Report
    outputs = Path('outputs')
    outputs.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
    out_md = outputs / f'metrics_comparison_{ts}.md'
    out_json = outputs / f'metrics_comparison_{ts}.json'

    # Build markdown
    lines = []
    lines.append(f"# Metrics Comparison ({ts} UTC)\n")
    lines.append("\n## Holdout (time-aware 20%)\n")
    lines.append("| Model | Accuracy | LogLoss | Brier | AUC | ECE | MCE |\n")
    lines.append("|---|---:|---:|---:|---:|---:|---:|\n")
    lines.append(f"| {res_a.name} | {res_a.acc:.4f} | {res_a.logloss:.4f} | {res_a.brier:.4f} | {res_a.auc:.4f} | {res_a.ece:.4f} | {res_a.mce:.4f} |\n")
    lines.append(f"| {res_b.name} | {res_b.acc:.4f} | {res_b.logloss:.4f} | {res_b.brier:.4f} | {res_b.auc:.4f} | {res_b.ece:.4f} | {res_b.mce:.4f} |\n")

    # deltas (B - A)
    def _sign(x: float) -> str:
        return f"+{x:.4f}" if x >= 0 else f"{x:.4f}"
    lines.append("\n## Delta (B - A)\n")
    lines.append("| Metric | Delta |\n")
    lines.append("|---|---:|\n")
    lines.append(f"| Accuracy | {_sign(res_b.acc - res_a.acc)} |\n")
    lines.append(f"| LogLoss | {_sign(res_b.logloss - res_a.logloss)} |\n")
    lines.append(f"| Brier | {_sign(res_b.brier - res_a.brier)} |\n")
    lines.append(f"| AUC | {_sign(res_b.auc - res_a.auc)} |\n")
    lines.append(f"| ECE | {_sign(res_b.ece - res_a.ece)} |\n")
    lines.append(f"| MCE | {_sign(res_b.mce - res_a.mce)} |\n")

    # Calibration buckets (quantile strategy) for both models
    def _predict_proba(artifact_path: Path, X_base: pd.DataFrame, fallback_features: list[str]) -> np.ndarray:
        art = joblib.load(artifact_path)
        if isinstance(art, dict) and 'model' in art:
            model = art['model']
            feat_names = art.get('feature_names', fallback_features)
            X_eval = _align_features(X_base, feat_names)
            return model.predict_proba(X_eval)[:, 1]
        model = art
        try:
            feat_names = getattr(model, 'feature_names_in_', fallback_features)
        except Exception:
            feat_names = fallback_features
        X_eval = _align_features(X_base, list(feat_names))
        return model.predict_proba(X_eval)[:, 1]

    y_proba_a = _predict_proba(Path(args.model_a), X_test, feature_cols)
    y_proba_b = _predict_proba(Path(args.model_b), X_test, feature_cols)
    prob_true_a, prob_pred_a = calibration_curve(y_test, y_proba_a, n_bins=10, strategy='quantile')
    prob_true_b, prob_pred_b = calibration_curve(y_test, y_proba_b, n_bins=10, strategy='quantile')

    lines.append("\n## Calibration Buckets (quantile, 10 bins)\n")
    lines.append("| Bin | A: mean_pred | A: frac_pos | B: mean_pred | B: frac_pos |\n")
    lines.append("|---:|---:|---:|---:|---:|\n")
    for i in range(min(len(prob_true_a), len(prob_true_b))):
        lines.append(f"| {i+1} | {prob_pred_a[i]:.3f} | {prob_true_a[i]:.3f} | {prob_pred_b[i]:.3f} | {prob_true_b[i]:.3f} |\n")

    out_md.write_text("\n".join(lines))

    # JSON export with metrics and calibration bins
    payload = {
        'timestamp_utc': ts,
        'model_a': {
            'name': res_a.name,
            'accuracy': res_a.acc,
            'logloss': res_a.logloss,
            'brier': res_a.brier,
            'auc': res_a.auc,
            'ece': res_a.ece,
            'mce': res_a.mce,
            'calibration_bins': [{'mean_pred': float(mp), 'frac_pos': float(fp)} for mp, fp in zip(prob_pred_a, prob_true_a)]
        },
        'model_b': {
            'name': res_b.name,
            'accuracy': res_b.acc,
            'logloss': res_b.logloss,
            'brier': res_b.brier,
            'auc': res_b.auc,
            'ece': res_b.ece,
            'mce': res_b.mce,
            'calibration_bins': [{'mean_pred': float(mp), 'frac_pos': float(fp)} for mp, fp in zip(prob_pred_b, prob_true_b)]
        },
        'delta': {
            'accuracy': float(res_b.acc - res_a.acc),
            'logloss': float(res_b.logloss - res_a.logloss),
            'brier': float(res_b.brier - res_a.brier),
            'auc': float(res_b.auc - res_a.auc),
            'ece': float(res_b.ece - res_a.ece),
            'mce': float(res_b.mce - res_a.mce)
        }
    }
    out_json.write_text(json.dumps(payload, indent=2))
    print(f"Saved metrics comparison: {out_md}\nSaved metrics JSON: {out_json}")


if __name__ == '__main__':
    main()
