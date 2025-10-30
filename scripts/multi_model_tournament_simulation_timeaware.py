"""
Multi-Model Tournament Simulation (Time-aware + Calibrated)
- Chronological train/calibration/test split using match_id order
- Blocked time-ordered CV on training window
- Probability calibration (Platt/sigmoid) using held-out calibration slice
- Same tournament simulation flow as multi_model_tournament_simulation.py
"""

import warnings
warnings.filterwarnings('ignore')

from pathlib import Path
import argparse
from datetime import datetime
import numpy as np
import pandas as pd
import sqlite3
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    log_loss,
    brier_score_loss,
)
from sklearn.calibration import CalibratedClassifierCV
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import VotingClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
import joblib

from config import (
    DB_FILE,
    VOLLEYBALL_FEATURES_STR,
    CSV_DIR_STR,
    MODELS_DIR,
    OUTPUTS_DIR,
)


# ---------- Model wrapper (module-level for pickling) ----------

class StackingModel:
    def __init__(self, bases: dict, meta_clf, order: list[str]):
        self.bases = bases
        self.order = order
        self.meta = meta_clf

    def _stack(self, X):
        cols = []
        for name in self.order:
            p = self.bases[name].predict_proba(X)[:, 1]
            cols.append(p)
        return np.column_stack(cols)

    def predict_proba(self, X):
        Z = self._stack(X)
        p1 = self.meta.predict_proba(Z)[:, 1]
        return np.column_stack([1 - p1, p1])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


# ---------- Data Loading (time-aware) ----------

def _safe_read_csv(primary: str, fallback: str | None = None) -> pd.DataFrame:
    p = Path(primary)
    if p.exists():
        return pd.read_csv(p)
    if fallback:
        fb = Path(fallback)
        if fb.exists():
            return pd.read_csv(fb)
    raise FileNotFoundError(f"Could not find CSV at {primary} or fallback {fallback}")


def load_features_timeaware():
    print("Loading full features with metadata (time-aware)...")
    features_path = VOLLEYBALL_FEATURES_STR
    fallback_features_path = str(Path(CSV_DIR_STR) / "volleyball_features_with_players.csv")
    df = _safe_read_csv(features_path, fallback_features_path)

    metadata_cols = ["match_id", "team_a_id", "team_b_id", "tournament_id"]
    target_col = "team_a_wins"
    feature_cols = [c for c in df.columns if c not in metadata_cols + [target_col]]

    df = df.sort_values("match_id").reset_index(drop=True)

    X = df[feature_cols]
    y = df[target_col].values.ravel()

    print(f"✓ Loaded {len(X)} samples with {len(feature_cols)} features")
    return X, y, feature_cols


def time_aware_split(X: pd.DataFrame, y: np.ndarray):
    n = len(X)
    test_start = int(n * 0.8)

    X_train_full, y_train_full = X.iloc[:test_start], y[:test_start]
    X_test, y_test = X.iloc[test_start:], y[test_start:]

    cal_start = int(len(X_train_full) * 0.9)
    X_subtrain, y_subtrain = X_train_full.iloc[:cal_start], y_train_full[:cal_start]
    X_calib, y_calib = X_train_full.iloc[cal_start:], y_train_full[cal_start:]

    print("Split windows (chronological):")
    print(f"  Sub-train: {len(X_subtrain)} | Calib: {len(X_calib)} | Test: {len(X_test)}")
    return X_subtrain, y_subtrain, X_calib, y_calib, X_test, y_test, X_train_full, y_train_full


def time_ordered_cv_metrics(model_ctor, X_train_full, y_train_full, k: int = 5):
    k = max(2, min(k, len(X_train_full)))
    fold_sizes = np.full(k, len(X_train_full) // k, dtype=int)
    fold_sizes[: len(X_train_full) % k] += 1
    current = 0
    metrics = []
    for i, fold_size in enumerate(fold_sizes):
        start, stop = current, current + fold_size
        X_val = X_train_full.iloc[start:stop]
        y_val = y_train_full[start:stop]
        X_tr = X_train_full.iloc[:start]
        y_tr = y_train_full[:start]
        if len(X_tr) == 0 or len(np.unique(y_val)) < 2:
            current = stop
            continue
        m = model_ctor()
        m.fit(X_tr, y_tr)
        y_pred = m.predict(X_val)
        y_proba = m.predict_proba(X_val)[:, 1]
        metrics.append({
            "fold": i + 1,
            "acc": accuracy_score(y_val, y_pred),
            "logloss": log_loss(y_val, y_proba, labels=[0, 1]),
            "brier": brier_score_loss(y_val, y_proba),
            "auc": roc_auc_score(y_val, y_proba),
        })
        current = stop
    return metrics


# ---------- Tournament helpers (same as original) ----------

def get_team_features_for_prediction(team_code):
    conn = sqlite3.connect(str(DB_FILE))
    cursor = conn.cursor()

    cursor.execute('''
        SELECT 
            COUNT(*) as matches_played,
            SUM(CASE WHEN m.winner_id = t.id THEN 1 ELSE 0 END) as matches_won,
            SUM(CASE WHEN m.team_a_id = t.id THEN m.team_a_sets_won ELSE m.team_b_sets_won END) as sets_won,
            SUM(CASE WHEN m.team_a_id = t.id THEN m.team_b_sets_won ELSE m.team_a_sets_won END) as sets_lost,
            AVG(CASE WHEN tms.team_id = t.id THEN tms.total_points ELSE 0 END) as avg_points_scored,
            AVG(CASE WHEN tms.team_id != t.id AND tms.match_id IN (
                SELECT m2.id FROM matches m2 WHERE m2.team_a_id = t.id OR m2.team_b_id = t.id
            ) THEN tms.total_points ELSE 0 END) as avg_points_conceded
        FROM teams t
        LEFT JOIN matches m ON (t.id = m.team_a_id OR t.id = m.team_b_id)
        LEFT JOIN team_match_stats tms ON m.id = tms.match_id
        WHERE t.code = ?
        GROUP BY t.id
    ''', (team_code,))

    result = cursor.fetchone()

    if not result or result[0] == 0:
        conn.close()
        return {
            'prev_matches_played': 0, 'prev_matches_won': 0, 'prev_win_rate': 0,
            'prev_sets_won': 0, 'prev_sets_lost': 0, 'prev_set_win_rate': 0,
            'prev_avg_points_scored': 0, 'prev_avg_points_conceded': 0
        }

    matches_played, matches_won, sets_won, sets_lost, avg_pts_scored, avg_pts_conceded = result

    cursor.execute('''
        SELECT 
            AVG(pms.attack_points + pms.block_points + pms.serve_points) as avg_top_scorer_points,
            AVG(pms.attack_points) as avg_top_scorer_attacks,
            COUNT(DISTINCT pms.player_id) as num_regular_scorers
        FROM player_match_stats pms
        JOIN matches m ON pms.match_id = m.id
        JOIN teams t ON t.code = ?
        WHERE (m.team_a_id = t.id OR m.team_b_id = t.id)
        AND (pms.attack_points + pms.block_points + pms.serve_points) > 5
    ''', (team_code,))

    player_result = cursor.fetchone()
    conn.close()

    features = {
        'prev_matches_played': matches_played,
        'prev_matches_won': matches_won,
        'prev_win_rate': matches_won / matches_played if matches_played > 0 else 0,
        'prev_sets_won': sets_won or 0,
        'prev_sets_lost': sets_lost or 0,
        'prev_set_win_rate': sets_won / (sets_won + sets_lost) if (sets_won + sets_lost) > 0 else 0,
        'prev_avg_points_scored': avg_pts_scored or 0,
        'prev_avg_points_conceded': avg_pts_conceded or 0,
        'avg_top_scorer_points': player_result[0] or 0 if player_result else 0,
        'avg_top_scorer_attacks': player_result[1] or 0 if player_result else 0,
        'num_regular_scorers': player_result[2] or 0 if player_result else 0
    }

    return features


def predict_match(model, team_a_code, team_b_code, feature_names):
    team_a_features = get_team_features_for_prediction(team_a_code)
    team_b_features = get_team_features_for_prediction(team_b_code)

    # Compute current ELO ratings from historical matches (no leakage for sim context)
    def compute_current_elo():
        conn = sqlite3.connect(str(DB_FILE))
        cursor = conn.cursor()
        cursor.execute('''
            SELECT m.team_a_id, m.team_b_id, m.winner_id, ta.code as a_code, tb.code as b_code
            FROM matches m
            JOIN teams ta ON m.team_a_id = ta.id
            JOIN teams tb ON m.team_b_id = tb.id
            WHERE m.status IS NOT NULL
            ORDER BY m.id
        ''')
        elo = {}
        def get(code):
            return elo.get(code, 1500.0)
        def expected(ra, rb):
            return 1.0 / (1.0 + 10 ** (-(ra - rb) / 400.0))
        K = 20.0
        rows = cursor.fetchall()
        conn.close()
        for team_a_id, team_b_id, winner_id, a_code, b_code in rows:
            ra = get(a_code)
            rb = get(b_code)
            ea = expected(ra, rb)
            sa = 1.0 if winner_id == team_a_id else 0.0
            sb = 1.0 - sa
            ra_new = ra + K * (sa - ea)
            rb_new = rb + K * (sb - (1.0 - ea))
            elo[a_code] = ra_new
            elo[b_code] = rb_new
        return elo

    elo_map = compute_current_elo()
    team_a_elo = elo_map.get(team_a_code, 1500.0)
    team_b_elo = elo_map.get(team_b_code, 1500.0)
    elo_diff = team_a_elo - team_b_elo
    elo_prob_team_a = 1.0 / (1.0 + 10 ** (-(elo_diff) / 400.0))

    conn = sqlite3.connect(str(DB_FILE))
    cursor = conn.cursor()
    cursor.execute('''
        SELECT COUNT(*),
               SUM(CASE WHEN m.winner_id = ta.id THEN 1 ELSE 0 END) as team_a_wins
        FROM matches m
        JOIN teams ta ON (ta.code = ? AND (m.team_a_id = ta.id OR m.team_b_id = ta.id))
        JOIN teams tb ON (tb.code = ? AND (m.team_a_id = tb.id OR m.team_b_id = tb.id))
        WHERE (m.team_a_id = ta.id AND m.team_b_id = tb.id) 
           OR (m.team_a_id = tb.id AND m.team_b_id = ta.id)
    ''', (team_a_code, team_b_code))
    h2h_result = cursor.fetchone()
    conn.close()

    h2h_matches = h2h_result[0] if h2h_result else 0
    h2h_team_a_wins = h2h_result[1] if h2h_result and h2h_result[1] else 0

    features = {}
    for key, value in team_a_features.items():
        features[f'team_a_{key}'] = value
    for key, value in team_b_features.items():
        features[f'team_b_{key}'] = value
    features['h2h_matches'] = h2h_matches
    features['h2h_team_a_wins'] = h2h_team_a_wins
    # Add ELO features so model inputs match training columns
    features['team_a_elo'] = team_a_elo
    features['team_b_elo'] = team_b_elo
    features['elo_diff'] = elo_diff
    features['elo_prob_team_a'] = elo_prob_team_a

    X = pd.DataFrame([features])
    for col in feature_names:
        if col not in X.columns:
            X[col] = 0
    X = X[feature_names]

    prediction = model.predict(X)[0]
    probability = model.predict_proba(X)[0]

    winner = team_a_code if prediction == 1 else team_b_code
    confidence = max(probability)
    return winner, confidence


def get_current_standings_and_seeding():
    conn = sqlite3.connect(str(DB_FILE))
    cursor = conn.cursor()

    cursor.execute('''
        SELECT t.code, t.name,
               SUM(CASE WHEN m.winner_id = t.id THEN 1 ELSE 0 END) as wins,
               SUM(CASE WHEN m.winner_id != t.id THEN 1 ELSE 0 END) as losses,
               COUNT(*) as games_played
        FROM teams t
        JOIN matches m ON (t.id = m.team_a_id OR t.id = m.team_b_id)
        WHERE m.tournament_id = (SELECT id FROM tournaments WHERE code = 'TEST_PVLR25')
        GROUP BY t.code, t.name
        ORDER BY wins DESC, losses ASC
        LIMIT 8
    ''')

    top_8 = []
    for row in cursor.fetchall():
        top_8.append({
            'code': row[0],
            'name': row[1],
            'wins': row[2],
            'losses': row[3],
            'games_played': row[4],
            'win_pct': row[2] / row[4] if row[4] > 0 else 0
        })

    conn.close()
    return top_8


# ---------- Training and Simulation ----------

def train_models_timeaware(X_subtrain, y_subtrain, X_calib, y_calib, X_test, y_test, X_train_full, y_train_full):
    print("\n" + "=" * 80)
    print("TRAINING (Time-aware + Calibrated)")
    print("=" * 80)

    def make_xgb():
        return XGBClassifier(
            max_depth=7,
            learning_rate=0.05,
            n_estimators=200,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            eval_metric='logloss',
            verbosity=0,
        )

    def make_xgb_deep():
        return XGBClassifier(
            max_depth=10,
            learning_rate=0.03,
            n_estimators=300,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            eval_metric='logloss',
            verbosity=0,
        )

    def make_lgbm():
        return LGBMClassifier(
            num_leaves=70,
            max_depth=8,
            learning_rate=0.03,
            n_estimators=250,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbosity=-1,
        )

    def make_cat():
        return CatBoostClassifier(
            iterations=250,
            depth=8,
            learning_rate=0.03,
            l2_leaf_reg=3,
            random_state=42,
            verbose=False,
        )

    def make_rf():
        return RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_split=10,
            random_state=42,
            n_jobs=-1,
        )

    def make_gb():
        return GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=7,
            random_state=42,
        )

    constructors = {
        'XGBoost': make_xgb,
        'XGBoost_Deep': make_xgb_deep,
        'LightGBM': make_lgbm,
        'CatBoost': make_cat,
        'Random_Forest': make_rf,
        'Gradient_Boosting': make_gb,
    }

    results = []
    trained_models = {}

    for name, ctor in constructors.items():
        print(f"\n[{name}] (time-aware)")
        print("-" * 60)
        base_model = ctor()
        base_model.fit(X_subtrain, y_subtrain)

        # Calibrate probabilities on the calibration slice
        calibrated = CalibratedClassifierCV(estimator=base_model, method='sigmoid', cv='prefit')
        calibrated.fit(X_calib, y_calib)

        # Evaluate on holdout test (calibrated)
        y_pred = calibrated.predict(X_test)
        y_proba = calibrated.predict_proba(X_test)[:, 1]

        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_proba)
        ll = log_loss(y_test, y_proba, labels=[0, 1])
        brier = brier_score_loss(y_test, y_proba)

        # Time-ordered CV (uncalibrated base model)
        cv_metrics = time_ordered_cv_metrics(ctor, X_train_full, y_train_full, k=5)
        if len(cv_metrics) > 0:
            cv_acc_mean = np.nanmean([m['acc'] for m in cv_metrics])
        else:
            cv_acc_mean = np.nan

        print(f"  Accuracy:     {accuracy:.4f}")
        print(f"  F1 Score:     {f1:.4f}")
        print(f"  ROC-AUC:      {auc:.4f}")
        print(f"  Log Loss:     {ll:.4f}")
        print(f"  Brier Score:  {brier:.4f}")
        print(f"  Time-CV Acc:  {cv_acc_mean:.4f}")

        results.append({
            'name': name,
            'model': calibrated,
            'accuracy': accuracy,
            'f1_score': f1,
            'auc': auc,
            'logloss': ll,
            'brier': brier,
            'cv_acc': cv_acc_mean,
        })
        trained_models[name] = calibrated

    # Voting ensemble (calibrated)
    print(f"\n[Voting_Ensemble] (time-aware)")
    print("-" * 60)
    voting = VotingClassifier(
        estimators=[
            ('xgb', make_xgb_deep()),
            ('lgb', make_lgbm()),
            ('cat', make_cat()),
        ],
        voting='soft',
    )
    voting.fit(X_subtrain, y_subtrain)
    voting_cal = CalibratedClassifierCV(estimator=voting, method='sigmoid', cv='prefit')
    voting_cal.fit(X_calib, y_calib)

    y_pred = voting_cal.predict(X_test)
    y_proba = voting_cal.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    ll = log_loss(y_test, y_proba, labels=[0, 1])
    brier = brier_score_loss(y_test, y_proba)

    print(f"  Accuracy:     {accuracy:.4f}")
    print(f"  F1 Score:     {f1:.4f}")
    print(f"  ROC-AUC:      {auc:.4f}")
    print(f"  Log Loss:     {ll:.4f}")
    print(f"  Brier Score:  {brier:.4f}")

    results.append({
        'name': 'Voting_Ensemble',
        'model': voting_cal,
        'accuracy': accuracy,
        'f1_score': f1,
        'auc': auc,
        'logloss': ll,
        'brier': brier,
        'cv_acc': np.nan,
    })
    trained_models['Voting_Ensemble'] = voting_cal

    # Stacking (OOF-based meta-learner)
    print(f"\n[Stacking_Meta] (time-aware)")
    print("-" * 60)

    # Choose base learners for stacking
    stack_order = ['XGBoost_Deep', 'LightGBM', 'CatBoost', 'Random_Forest', 'Gradient_Boosting']

    # Generate OOF probabilities on X_subtrain using blocked time-ordered folds
    n = len(X_subtrain)
    k = 5
    fold_sizes = np.full(k, n // k, dtype=int)
    fold_sizes[: n % k] += 1
    starts = np.cumsum(np.concatenate(([0], fold_sizes[:-1])))
    stops = np.cumsum(fold_sizes)

    oof_matrix = {name: np.full(n, np.nan, dtype=float) for name in stack_order}
    for fold_idx, (start, stop) in enumerate(zip(starts, stops), start=1):
        X_val = X_subtrain.iloc[start:stop]
        y_val = y_subtrain[start:stop]
        X_tr = X_subtrain.iloc[:start]
        y_tr = y_subtrain[:start]
        if len(X_tr) == 0 or len(np.unique(y_val)) < 2:
            continue
        for name in stack_order:
            ctor = constructors[name]
            m = ctor()
            m.fit(X_tr, y_tr)
            try:
                proba = m.predict_proba(X_val)[:, 1]
            except Exception:
                proba = np.full(len(X_val), 0.5)
            oof_matrix[name][start:stop] = proba

    # If any NaNs remain (e.g., skipped early fold), backfill by training on all available subtrain
    for name in stack_order:
        mask = np.isnan(oof_matrix[name])
        if mask.any():
            m = constructors[name]()
            m.fit(X_subtrain, y_subtrain)
            try:
                oof_matrix[name][mask] = m.predict_proba(X_subtrain.iloc[mask])[:, 1]
            except Exception:
                oof_matrix[name][mask] = 0.5

    # Train meta-learner on OOF matrix
    X_oof = np.column_stack([oof_matrix[name] for name in stack_order])
    meta = LogisticRegression(max_iter=1000, solver='lbfgs')
    meta.fit(X_oof, y_subtrain)

    # Fit base models on subtrain and calibrate on calibration slice for downstream predictions
    calibrated_bases = {}
    for name in stack_order:
        base = constructors[name]()
        base.fit(X_subtrain, y_subtrain)
        cal = CalibratedClassifierCV(estimator=base, method='sigmoid', cv='prefit')
        cal.fit(X_calib, y_calib)
        calibrated_bases[name] = cal

    # Evaluate stacked model on test
    test_stack = []
    for name in stack_order:
        proba = calibrated_bases[name].predict_proba(X_test)[:, 1]
        test_stack.append(proba)
    X_test_meta = np.column_stack(test_stack)
    y_proba_meta = meta.predict_proba(X_test_meta)[:, 1]
    y_pred_meta = (y_proba_meta >= 0.5).astype(int)

    accuracy = accuracy_score(y_test, y_pred_meta)
    f1 = f1_score(y_test, y_pred_meta)
    auc = roc_auc_score(y_test, y_proba_meta)
    ll = log_loss(y_test, y_proba_meta, labels=[0, 1])
    brier = brier_score_loss(y_test, y_proba_meta)

    print(f"  Accuracy:     {accuracy:.4f}")
    print(f"  F1 Score:     {f1:.4f}")
    print(f"  ROC-AUC:      {auc:.4f}")
    print(f"  Log Loss:     {ll:.4f}")
    print(f"  Brier Score:  {brier:.4f}")

    stacked_model = StackingModel(calibrated_bases, meta, list(calibrated_bases.keys()))

    results.append({
        'name': 'Stacking_Meta',
        'model': stacked_model,
        'accuracy': accuracy,
        'f1_score': f1,
        'auc': auc,
        'logloss': ll,
        'brier': brier,
        'cv_acc': np.nan,
    })
    trained_models['Stacking_Meta'] = stacked_model

    return results, trained_models


def simulate_tournament_with_model(model_name, model, feature_names):
    print(f"\n{'='*80}")
    print(f"SIMULATING WITH {model_name} (time-aware)")
    print(f"{'='*80}")

    top_8 = get_current_standings_and_seeding()
    if len(top_8) < 8:
        print(f"⚠️  Warning: Only {len(top_8)} teams found in database")
        return None

    print("\nTOP 8 SEEDING:")
    for i, team in enumerate(top_8, 1):
        print(f"  #{i} {team['code']}: {team['wins']}-{team['losses']} ({team['win_pct']:.1%})")

    qf_matchups = [
        (top_8[0]['code'], top_8[7]['code'], 'QF1', '#1 vs #8'),
        (top_8[1]['code'], top_8[6]['code'], 'QF2', '#2 vs #7'),
        (top_8[2]['code'], top_8[5]['code'], 'QF3', '#3 vs #6'),
        (top_8[3]['code'], top_8[4]['code'], 'QF4', '#4 vs #5')
    ]

    print("\nQUARTERFINALS:")
    qf_winners = []
    for team_a, team_b, qf_name, seed_info in qf_matchups:
        winner, confidence = predict_match(model, team_a, team_b, feature_names)
        qf_winners.append(winner)
        loser = team_b if winner == team_a else team_a
        print(f"  {qf_name} ({seed_info}): {team_a} vs {team_b} → {winner} defeats {loser} ({confidence:.1%})")

    print("\nSEMIFINALS:")
    sf_matchups = [
        (qf_winners[0], qf_winners[3], 'SF1', 'QF1 vs QF4'),
        (qf_winners[1], qf_winners[2], 'SF2', 'QF2 vs QF3')
    ]

    sf_winners = []
    for team_a, team_b, sf_name, matchup_info in sf_matchups:
        winner, confidence = predict_match(model, team_a, team_b, feature_names)
        sf_winners.append(winner)
        loser = team_b if winner == team_a else team_a
        print(f"  {sf_name} ({matchup_info}): {team_a} vs {team_b} → {winner} defeats {loser} ({confidence:.1%})")

    print("\nFINALS:")
    champion, confidence = predict_match(model, sf_winners[0], sf_winners[1], feature_names)
    runner_up = sf_winners[1] if champion == sf_winners[0] else sf_winners[0]
    print(f"  {sf_winners[0]} vs {sf_winners[1]} → {champion} defeats {runner_up} ({confidence:.1%})")

    print(f"\n🏆 CHAMPION: {champion}")
    print(f"🥈 RUNNER-UP: {runner_up}")

    return {
        'model': model_name,
        'champion': champion,
        'runner_up': runner_up,
        'confidence': confidence,
        'qf_winners': qf_winners,
        'sf_winners': sf_winners,
        'seeding': [t['code'] for t in top_8]
    }


def main(save_summary: bool = False, summary_file: str | None = None):
    print("\n" + "="*80)
    print(" "*10 + "MULTI-MODEL TOURNAMENT SIMULATION (Time-aware + Calibrated)")
    print("="*80)

    # Load features and split
    X, y, feature_cols = load_features_timeaware()
    X_subtrain, y_subtrain, X_calib, y_calib, X_test, y_test, X_train_full, y_train_full = time_aware_split(X, y)

    print(f"  Sub-train: {len(X_subtrain)} | Calib: {len(X_calib)} | Test: {len(X_test)}")

    # Train/calibrate models
    results, trained_models = train_models_timeaware(
        X_subtrain, y_subtrain, X_calib, y_calib, X_test, y_test, X_train_full, y_train_full
    )

    # Show comparison
    print("\n" + "="*80)
    print("MODEL PERFORMANCE (Time-aware + Calibrated)")
    print("="*80)
    print(f"\n{'Model':<20} {'Acc':<8} {'F1':<8} {'AUC':<8} {'LogLoss':<10} {'Brier':<8} {'TimeCV Acc'}")
    print("-"*90)
    sorted_results = sorted(results, key=lambda x: x['accuracy'], reverse=True)
    for r in sorted_results:
        print(f"{r['name']:<20} {r['accuracy']:<8.4f} {r['f1_score']:<8.4f} {r['auc']:<8.4f} "
              f"{r['logloss']:<10.4f} {r['brier']:<8.4f} {r['cv_acc']}")

    # Simulate tournament with top models similar to original
    top_models_to_simulate = [
        'Stacking_Meta',
        'XGBoost_Deep',
        'LightGBM',
        'CatBoost',
        'Voting_Ensemble',
        'Random_Forest',
    ]

    print("\n" + "="*80)
    print("TOURNAMENT SIMULATIONS (Time-aware + Calibrated)")
    print("="*80)
    tournament_results = []
    for model_name in top_models_to_simulate:
        if model_name in trained_models:
            result = simulate_tournament_with_model(model_name, trained_models[model_name], feature_cols)
            tournament_results.append(result)

    # Save best calibrated time-aware model (do not overwrite default BEST_MODEL)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    best = sorted_results[0]
    out_path = MODELS_DIR / "best_model_with_players_timeaware.pkl"
    joblib.dump({
        'model': best['model'],
        'feature_names': feature_cols,
        'accuracy': best['accuracy'],
        'model_name': best['name'],
        'calibrated': True,
        'time_aware': True,
    }, str(out_path))
    print(f"\nSaved best time-aware calibrated model: {best['name']} ({best['accuracy']:.4f}) → {out_path}")

    # Save stacked model artifact for convenient use in run_simulation
    if 'Stacking_Meta' in trained_models:
        stack_path = MODELS_DIR / "best_model_with_players_timeaware_stacking.pkl"
        joblib.dump({
            'model': trained_models['Stacking_Meta'],
            'feature_names': feature_cols,
            'model_name': 'Stacking_Meta',
            'calibrated': True,
            'time_aware': True,
            'note': 'Stacked meta-learner over calibrated base models',
        }, str(stack_path))
        print(f"Saved stacking model artifact → {stack_path}")

    # Optionally write a concise summary to a file for diffing
    if save_summary:
        OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
        default_summary = OUTPUTS_DIR / "simulation_output_timeaware.txt"
        target_path = Path(summary_file) if summary_file else default_summary

        lines = []
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        lines.append(f"Time-aware + Calibrated Simulation Summary\nGenerated: {ts}")
        lines.append("")
        lines.append(f"Samples: {len(X)} | Features: {len(feature_cols)}")
        lines.append(f"Split: sub-train={len(X_subtrain)}, calib={len(X_calib)}, test={len(X_test)}")
        lines.append("")
        lines.append("Model performance (sorted by accuracy):")
        for r in sorted_results:
            lines.append(
                f"- {r['name']}: acc={r['accuracy']:.4f}, f1={r['f1_score']:.4f}, auc={r['auc']:.4f}, "
                f"logloss={r['logloss']:.4f}, brier={r['brier']:.4f}, timeCV={r['cv_acc']}"
            )
        lines.append("")
        lines.append("Tournament results:")
        for tr in tournament_results:
            if tr:
                lines.append(
                    f"- {tr['model']}: champion={tr['champion']}, runner_up={tr['runner_up']}, conf={tr['confidence']:.3f}"
                )
        lines.append("")
        lines.append(f"Saved best model: {best['name']} ({best['accuracy']:.4f}) → {out_path}")
        if 'Stacking_Meta' in trained_models:
            lines.append(f"Saved stacked model → {MODELS_DIR / 'best_model_with_players_timeaware_stacking.pkl'}")

        with open(target_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        print(f"\nSummary written to: {target_path}")

    print("\n" + "="*80)
    print("SIMULATION COMPLETE (Time-aware + Calibrated)")
    print("="*80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Time-aware + Calibrated multi-model simulation")
    parser.add_argument("--save-summary", action="store_true", help="Write a text summary to outputs/")
    parser.add_argument("--summary-file", type=str, default=None, help="Custom path for the summary file")
    args = parser.parse_args()
    main(save_summary=args.save_summary, summary_file=args.summary_file)
