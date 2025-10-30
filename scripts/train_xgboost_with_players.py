"""
Train XGBoost model with player-level features for volleyball match prediction

Upgrades in this version:
- Time-aware train/validation/test split (chronological by match_id)
- Blocked, time-ordered cross-validation on the training period
- Probability calibration (sigmoid/Platt) using a held-out calibration split
- Metrics: Accuracy, Log Loss, Brier Score, ROC-AUC
- Saves calibrated model to models/ for downstream simulation
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    log_loss,
    brier_score_loss,
    roc_auc_score,
)
from sklearn.calibration import CalibratedClassifierCV
from xgboost import XGBClassifier
import joblib
from config import (
    VOLLEYBALL_FEATURES_STR,
    BEST_MODEL_STR,
    CSV_DIR_STR,
    MODELS_DIR,
)

print("=" * 80)
print(" " * 15 + "XGBOOST TRAINING WITH PLAYER-LEVEL FEATURES (Time-aware + Calibrated)")
print("=" * 80)

def _safe_read_csv(primary: str, fallback: str | None = None) -> pd.DataFrame:
    """Read CSV from primary path, fallback to secondary if needed."""
    p = Path(primary)
    if p.exists():
        return pd.read_csv(p)
    if fallback:
        fb = Path(fallback)
        if fb.exists():
            return pd.read_csv(fb)
    raise FileNotFoundError(f"Could not find CSV at {primary} or fallback {fallback}")


# Load full features to access metadata and target deterministically
print("\nLoading features with metadata (for time-aware split)...")
features_path = VOLLEYBALL_FEATURES_STR
fallback_features_path = str(Path(CSV_DIR_STR) / "volleyball_features_with_players.csv")
features_df = _safe_read_csv(features_path, fallback_features_path)

# Identify columns
metadata_cols = ["match_id", "team_a_id", "team_b_id", "tournament_id"]
target_col = "team_a_wins"
feature_cols = [c for c in features_df.columns if c not in metadata_cols + [target_col]]

# Sort chronologically by match_id (IDs are generated in temporal order in this pipeline)
features_df = features_df.sort_values("match_id").reset_index(drop=True)

X = features_df[feature_cols]
y = features_df[target_col].values.ravel()
match_ids = features_df["match_id"].values
tournament_ids = features_df["tournament_id"].values

print(f"\n✓ Data loaded: {len(X)} samples, {len(feature_cols)} features")

print("\nFeature groups:")
print(f"  Team historical stats: {len([c for c in X.columns if 'win_rate' in c or 'avg' in c and 'starter' not in c and 'libero' not in c and 'scorer' not in c and 'roster' not in c and 'sets_per' not in c and '10plus' not in c])}")
print(f"  Player-based stats: {len([c for c in X.columns if 'starter' in c or 'libero' in c or 'scorer' in c or 'roster' in c or 'sets_per' in c or '10plus' in c])}")

# Split data
print("\n" + "=" * 80)
print("PREPARING DATA (Time-aware split)")
print("=" * 80)

# Chronological split: first 80% for training (including calibration), last 20% for final test
n = len(X)
test_start = int(n * 0.8)

X_train_full, y_train_full = X.iloc[:test_start], y[:test_start]
X_test, y_test = X.iloc[test_start:], y[test_start:]

# Within training, reserve last 10% as calibration set
cal_start = int(len(X_train_full) * 0.9)
X_subtrain, y_subtrain = X_train_full.iloc[:cal_start], y_train_full[:cal_start]
X_calib, y_calib = X_train_full.iloc[cal_start:], y_train_full[cal_start:]

print(f"Training window (sub-train): {len(X_subtrain)} samples")
print(f"Calibration window:          {len(X_calib)} samples")
print(f"Test window (holdout):       {len(X_test)} samples")
print(
    f"Class dist - Sub-train: {{1: {int(y_subtrain.sum())}, 0: {len(y_subtrain) - int(y_subtrain.sum())}}}"
)
print(
    f"Class dist - Calibration: {{1: {int(y_calib.sum())}, 0: {len(y_calib) - int(y_calib.sum())}}}"
)
print(
    f"Class dist - Test: {{1: {int(y_test.sum())}, 0: {len(y_test) - int(y_test.sum())}}}"
)

# Train model
print("\n" + "=" * 80)
print("TRAINING MODEL (with time-ordered CV)")
print("=" * 80)
print("\nTraining base XGBoost classifier...")

base_model = XGBClassifier(
    n_estimators=400,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.9,
    colsample_bytree=0.9,
    random_state=42,
    eval_metric="logloss",
    n_jobs=-1,
)

# Time-ordered cross-validation across the training window (blocked folds)
k = 5
fold_sizes = np.full(k, len(X_train_full) // k, dtype=int)
fold_sizes[: len(X_train_full) % k] += 1
current = 0
cv_metrics = []
for i, fold_size in enumerate(fold_sizes):
    start, stop = current, current + fold_size
    X_val = X_train_full.iloc[start:stop]
    y_val = y_train_full[start:stop]
    X_tr = X_train_full.iloc[:start]
    y_tr = y_train_full[:start]
    # ensure we have training data before first fold
    if len(X_tr) == 0:
        current = stop
        continue
    model_i = XGBClassifier(**base_model.get_params())
    model_i.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
    y_pred = model_i.predict(X_val)
    y_proba = model_i.predict_proba(X_val)[:, 1]
    metrics = {
        "fold": i + 1,
        "acc": accuracy_score(y_val, y_pred),
        "logloss": log_loss(y_val, y_proba, labels=[0, 1]),
        "brier": brier_score_loss(y_val, y_proba),
        "auc": roc_auc_score(y_val, y_proba) if len(np.unique(y_val)) > 1 else np.nan,
    }
    cv_metrics.append(metrics)
    print(
        f"Fold {metrics['fold']}: acc={metrics['acc']:.4f} | logloss={metrics['logloss']:.4f} | "
        f"brier={metrics['brier']:.4f} | auc={metrics['auc']:.4f}"
    )
    current = stop

# Fit on sub-train (leaving calibration slice untouched)
base_model.fit(X_subtrain, y_subtrain, eval_set=[(X_calib, y_calib)], verbose=False)
print("✓ Base model training complete")

# Evaluate
print("\n" + "=" * 80)
print("MODEL EVALUATION (Uncalibrated vs Calibrated)")
print("=" * 80)

# Uncalibrated on test
y_pred_unc = base_model.predict(X_test)
y_proba_unc = base_model.predict_proba(X_test)[:, 1]
acc_unc = accuracy_score(y_test, y_pred_unc)
ll_unc = log_loss(y_test, y_proba_unc, labels=[0, 1])
brier_unc = brier_score_loss(y_test, y_proba_unc)
auc_unc = roc_auc_score(y_test, y_proba_unc)

print(f"Uncalibrated → acc={acc_unc:.4f} | logloss={ll_unc:.4f} | brier={brier_unc:.4f} | auc={auc_unc:.4f}")

# Calibrate probabilities on calibration slice (cv='prefit')
calibrated_clf = CalibratedClassifierCV(estimator=base_model, method="sigmoid", cv="prefit")
calibrated_clf.fit(X_calib, y_calib)

y_pred_cal = calibrated_clf.predict(X_test)
y_proba_cal = calibrated_clf.predict_proba(X_test)[:, 1]
acc_cal = accuracy_score(y_test, y_pred_cal)
ll_cal = log_loss(y_test, y_proba_cal, labels=[0, 1])
brier_cal = brier_score_loss(y_test, y_proba_cal)
auc_cal = roc_auc_score(y_test, y_proba_cal)

print(f"Calibrated   → acc={acc_cal:.4f} | logloss={ll_cal:.4f} | brier={brier_cal:.4f} | auc={auc_cal:.4f}")

print("\nClassification Report (Calibrated):")
print(classification_report(y_test, y_pred_cal, target_names=["Team B Wins", "Team A Wins"]))

print("\nConfusion Matrix (Calibrated):")
cm = confusion_matrix(y_test, y_pred_cal)
print(f"                Predicted")
print(f"                B Wins  A Wins")
print(f"Actual B Wins     {cm[0][0]:3d}     {cm[0][1]:3d}")
print(f"Actual A Wins     {cm[1][0]:3d}     {cm[1][1]:3d}")

# Feature importance
print("\n" + "=" * 80)
print("FEATURE IMPORTANCE (Top 15)")
print("=" * 80)

feature_importance = (
    pd.DataFrame({"feature": feature_cols, "importance": base_model.feature_importances_})
    .sort_values("importance", ascending=False)
)

print("\nMost Important Features:")
for _, row in feature_importance.head(15).iterrows():
    print(f"  {row['feature']:40s} {row['importance']:.4f}")

# Save model
print("\n" + "=" * 80)
print("SAVING MODEL")
print("=" * 80)

MODELS_DIR.mkdir(parents=True, exist_ok=True)

uncalibrated_path = Path(BEST_MODEL_STR).with_name("volleyball_predictor_with_players_uncalibrated.pkl")
# Save calibrated model to a distinct filename to avoid overwriting other baselines
calibrated_path = Path(BEST_MODEL_STR).with_name("calibrated_xgboost_with_players.pkl")

joblib.dump({"model": base_model, "feature_names": feature_cols}, uncalibrated_path)
joblib.dump({"model": calibrated_clf, "feature_names": feature_cols}, calibrated_path)

print(f"✓ Uncalibrated model saved to: {uncalibrated_path}")
print(f"✓ Calibrated model saved to:   {calibrated_path}")

# Comparison with team-only model
print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
print(
    f"Time-aware CV: {len([m for m in cv_metrics])} folds | "
    f"Avg acc={np.nanmean([m['acc'] for m in cv_metrics]):.4f} | "
    f"Avg logloss={np.nanmean([m['logloss'] for m in cv_metrics]):.4f} | "
    f"Avg brier={np.nanmean([m['brier'] for m in cv_metrics]):.4f} | "
    f"Avg auc={np.nanmean([m['auc'] for m in cv_metrics if not np.isnan(m['auc'])]):.4f}"
)
print(
    f"Holdout (uncalibrated): acc={acc_unc:.4f}, logloss={ll_unc:.4f}, brier={brier_unc:.4f}, auc={auc_unc:.4f}"
)
print(
    f"Holdout (calibrated):   acc={acc_cal:.4f}, logloss={ll_cal:.4f}, brier={brier_cal:.4f}, auc={auc_cal:.4f}"
)

print("\n" + "=" * 80)
print("TRAINING COMPLETE!")
print("=" * 80)
print("\nYour calibrated XGBoost model with player features is ready!")
print("\nNext steps:")
print("1. Use models/best_model_with_players.pkl for predictions (calibrated)")
print("2. Run run_simulation.py or multi_model_tournament_simulation.py to compare")
print("3. Optionally inspect outputs/reliability later (to be added)")
