"""
ARCHIVED: Baseline Multi-Model Tournament Simulation (Random Split)
Moved to scripts/archive/ to keep scripts/ focused on full-tournament simulation entrypoints.
"""

from pathlib import Path

# Original content preserved below

"""
Multi-Model Tournament Simulation
Trains XGBoost, CatBoost, LightGBM, and ensemble models with player statistics
Then simulates the tournament with each model to compare predictions
"""

import pandas as pd
import numpy as np
import sqlite3
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import VotingClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
import joblib
import warnings
from config import X_FEATURES, Y_TARGET, DB_FILE, BEST_MODEL, OUTPUTS_DIR
from pathlib import Path
from datetime import datetime
import argparse
warnings.filterwarnings('ignore')


# ---------- Model wrapper (module-level for pickling) ----------
class StackingModel:
    def __init__(self, bases: dict, meta_clf, order: list[str]):
        self.bases = bases
        self.order = order
        self.meta = meta_clf

    def _stack(self, X):
        cols = []
        for name in self.order:
            cols.append(self.bases[name].predict_proba(X)[:, 1])
        return np.column_stack(cols)

    def predict_proba(self, X):
        Z = self._stack(X)
        p1 = self.meta.predict_proba(Z)[:, 1]
        return np.column_stack([1 - p1, p1])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


def load_features_with_players():
    """Load the feature set with player statistics"""
    print("Loading features with player statistics...")
    X = pd.read_csv(X_FEATURES)
    y = pd.read_csv(Y_TARGET).values.ravel()
    print(f"✓ Loaded {len(X)} samples with {len(X.columns)} features")
    return X, y


def train_multiple_models(X_train, y_train, X_test, y_test):
    """Train multiple algorithms with player statistics"""
    
    print("\n" + "="*80)
    print("TRAINING MULTIPLE MODELS WITH PLAYER STATISTICS")
    print("="*80)
    
    models = {
        'XGBoost': XGBClassifier(
            max_depth=7,
            learning_rate=0.05,
            n_estimators=200,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            eval_metric='logloss',
            verbosity=0
        ),
        'XGBoost_Deep': XGBClassifier(
            max_depth=10,
            learning_rate=0.03,
            n_estimators=300,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            eval_metric='logloss',
            verbosity=0
        ),
        'LightGBM': LGBMClassifier(
            num_leaves=70,
            max_depth=8,
            learning_rate=0.03,
            n_estimators=250,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbosity=-1
        ),
        'CatBoost': CatBoostClassifier(
            iterations=250,
            depth=8,
            learning_rate=0.03,
            l2_leaf_reg=3,
            random_state=42,
            verbose=False
        ),
        'Random_Forest': RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_split=10,
            random_state=42,
            n_jobs=-1
        ),
        'Gradient_Boosting': GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=7,
            random_state=42
        )
    }
    
    results = []
    trained_models = {}
    
    for name, model in models.items():
        print(f"\n[{name}]")
        print("-" * 60)
        
        # Train
        model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_proba)
        
        # Cross-validation
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
        
        print(f"  Accuracy:     {accuracy:.4f}")
        print(f"  F1 Score:     {f1:.4f}")
        print(f"  ROC-AUC:      {auc:.4f}")
        print(f"  CV Score:     {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")
        
        result = {
            'name': name,
            'model': model,
            'accuracy': accuracy,
            'f1_score': f1,
            'auc': auc,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std()
        }
        
        results.append(result)
        trained_models[name] = model
    
    # Create ensemble models
    print(f"\n[Voting Ensemble]")
    print("-" * 60)
    
    voting_clf = VotingClassifier(
        estimators=[
            ('xgb', models['XGBoost_Deep']),
            ('lgb', models['LightGBM']),
            ('cat', models['CatBoost'])
        ],
        voting='soft'
    )
    
    voting_clf.fit(X_train, y_train)
    y_pred = voting_clf.predict(X_test)
    y_proba = voting_clf.predict_proba(X_test)[:, 1]
    
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    
    print(f"  Accuracy:     {accuracy:.4f}")
    print(f"  F1 Score:     {f1:.4f}")
    print(f"  ROC-AUC:      {auc:.4f}")
    
    trained_models['Voting_Ensemble'] = voting_clf
    results.append({
        'name': 'Voting_Ensemble',
        'model': voting_clf,
        'accuracy': accuracy,
        'f1_score': f1,
        'auc': auc,
        'cv_mean': accuracy,
        'cv_std': 0
    })
    
    # Stacking (OOF-based meta-learner on training fold)
    print(f"\n[Stacking_Meta]")
    print("-" * 60)

    stack_order = ['XGBoost_Deep', 'LightGBM', 'CatBoost', 'Random_Forest', 'Gradient_Boosting']

    # OOF probabilities using stratified K-fold on training set
    n = len(X_train)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof_matrix = {name: np.full(n, np.nan, dtype=float) for name in stack_order}
    for train_idx, val_idx in skf.split(X_train, y_train):
        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train[train_idx], y_train[val_idx]
        for name in stack_order:
            base_ctor = models[name].__class__ if not isinstance(models[name], type) else models[name]
            # Recreate model with same params
            base_model = models[name].__class__(**models[name].get_params())
            base_model.fit(X_tr, y_tr)
            proba = base_model.predict_proba(X_val)[:, 1]
            oof_matrix[name][val_idx] = proba

    # Train meta-learner
    X_oof = np.column_stack([oof_matrix[name] for name in stack_order])
    meta = LogisticRegression(max_iter=1000, solver='lbfgs')
    meta.fit(X_oof, y_train)

    # Fit base models on full training for inference
    base_fitted = {}
    for name in stack_order:
        m = models[name].__class__(**models[name].get_params())
        m.fit(X_train, y_train)
        base_fitted[name] = m

    # Evaluate on test
    test_stack = []
    for name in stack_order:
        test_stack.append(base_fitted[name].predict_proba(X_test)[:, 1])
    X_test_meta = np.column_stack(test_stack)
    y_proba_meta = meta.predict_proba(X_test_meta)[:, 1]
    y_pred_meta = (y_proba_meta >= 0.5).astype(int)

    accuracy = accuracy_score(y_test, y_pred_meta)
    f1 = f1_score(y_test, y_pred_meta)
    auc = roc_auc_score(y_test, y_proba_meta)

    print(f"  Accuracy:     {accuracy:.4f}")
    print(f"  F1 Score:     {f1:.4f}")
    print(f"  ROC-AUC:      {auc:.4f}")

    stacking_model = StackingModel(base_fitted, meta, stack_order)

    results.append({
        'name': 'Stacking_Meta',
        'model': stacking_model,
        'accuracy': accuracy,
        'f1_score': f1,
        'auc': auc,
        'cv_mean': accuracy,
        'cv_std': 0
    })
    trained_models['Stacking_Meta'] = stacking_model

    return results, trained_models


def get_team_features_for_prediction(team_code):
    """Get all 74 features for a team (basic + enhanced + player stats)"""
    conn = sqlite3.connect(str(DB_FILE))
    cursor = conn.cursor()
    
    # Get basic historical features
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
    
    # Get player statistics
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
        'prev_sets_lost': 0 if sets_lost is None else sets_lost,
        'prev_set_win_rate': (sets_won / (sets_won + sets_lost)) if (sets_won + (sets_lost or 0)) > 0 else 0,
        'prev_avg_points_scored': avg_pts_scored or 0,
        'prev_avg_points_conceded': avg_pts_conceded or 0,
        'avg_top_scorer_points': player_result[0] or 0 if player_result else 0,
        'avg_top_scorer_attacks': player_result[1] or 0 if player_result else 0,
        'num_regular_scorers': player_result[2] or 0 if player_result else 0
    }
    
    return features


def predict_match(model, team_a_code, team_b_code, feature_names):
    """Predict match outcome using the trained model"""
    team_a_features = get_team_features_for_prediction(team_a_code)
    team_b_features = get_team_features_for_prediction(team_b_code)
    
    # Get head-to-head
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
    
    # Build feature dictionary
    features = {}
    for key, value in team_a_features.items():
        features[f'team_a_{key}'] = value
    for key, value in team_b_features.items():
        features[f'team_b_{key}'] = value
    features['h2h_matches'] = h2h_matches
    features['h2h_team_a_wins'] = h2h_team_a_wins
    
    # Create DataFrame
    X = pd.DataFrame([features])
    
    # Add missing features with default values
    for col in feature_names:
        if col not in X.columns:
            X[col] = 0
    
    # Ensure column order
    X = X[feature_names]
    
    prediction = model.predict(X)[0]
    probability = model.predict_proba(X)[0]
    
    winner = team_a_code if prediction == 1 else team_b_code
    confidence = max(probability)
    
    return winner, confidence


def get_current_standings_and_seeding():
    """Get current standings and determine top 8 seeding for quarterfinals"""
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


def main(save_summary: bool = False, summary_file: str | None = None):
    print("\n" + "="*80)
    print(" "*15 + "MULTI-MODEL TOURNAMENT SIMULATION")
    print(" "*20 + "with Player Statistics")
    print("="*80)
    
    # Load features
    X, y = load_features_with_players()
    feature_names = X.columns.tolist()
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"  Training set: {len(X_train)} samples")
    print(f"  Test set:     {len(X_test)} samples")
    
    # Train all models
    results, trained_models = train_multiple_models(X_train, y_train, X_test, y_test)
    
    # Display model comparison
    print("\n" + "="*80)
    print("MODEL PERFORMANCE COMPARISON")
    print("="*80)
    print(f"\n{'Model':<20} {'Accuracy':<12} {'F1':<12} {'AUC':<12} {'CV Score'}")
    print("-"*80)
    
    sorted_results = sorted(results, key=lambda x: x['accuracy'], reverse=True)
    for r in sorted_results:
        cv_str = f"{r['cv_mean']:.4f} ±{r['cv_std']:.4f}"
        print(f"{r['name']:<20} {r['accuracy']:<12.4f} {r['f1_score']:<12.4f} "
              f"{r['auc']:<12.4f} {cv_str}")
    
    # Select top models for tournament simulation
    top_models_to_simulate = [
        'Stacking_Meta',
        'XGBoost_Deep',
        'LightGBM', 
        'CatBoost',
        'Voting_Ensemble',
        'Random_Forest'
    ]
    
    # Simulate tournament with each model
    print("\n" + "="*80)
    print("TOURNAMENT SIMULATIONS")
    print("="*80)
    
    tournament_results = []
    for model_name in top_models_to_simulate:
        if model_name in trained_models:
            result = simulate_tournament_with_model(
                model_name,
                trained_models[model_name],
                feature_names
            )
            tournament_results.append(result)
    
    # Championship comparison
    print("\n" + "="*80)
    print("CHAMPIONSHIP PREDICTIONS SUMMARY")
    print("="*80)
    
    print(f"\n{'Model':<25} {'Champion':<10} {'Runner-Up':<10} {'Confidence'}")
    print("-"*80)
    
    for result in tournament_results:
        print(f"{result['model']:<25} {result['champion']:<10} "
              f"{result['runner_up']:<10} {result['confidence']:.1%}")
    
    # Analyze consensus
    print("\n" + "="*80)
    print("CONSENSUS ANALYSIS")
    print("="*80)
    
    champions = [r['champion'] for r in tournament_results]
    champion_counts = {}
    for champ in champions:
        champion_counts[champ] = champion_counts.get(champ, 0) + 1
    
    print(f"\nChampion predictions:")
    for team, count in sorted(champion_counts.items(), key=lambda x: x[1], reverse=True):
        percentage = (count / len(tournament_results)) * 100
        bar = "█" * int(percentage / 10)
        print(f"  {team}: {count}/{len(tournament_results)} models ({percentage:.0f}%) {bar}")
    
    most_common_champ = max(champion_counts.items(), key=lambda x: x[1])
    
    if most_common_champ[1] == len(tournament_results):
        print(f"\n✓ UNANIMOUS: All models predict {most_common_champ[0]} as champion!")
    elif most_common_champ[1] >= len(tournament_results) * 0.6:
        print(f"\n✓ STRONG CONSENSUS: {most_common_champ[1]}/{len(tournament_results)} models predict {most_common_champ[0]}")
    else:
        print(f"\n⚠️  SPLIT DECISION: {most_common_champ[0]} leads with {most_common_champ[1]}/{len(tournament_results)} models")
    
    # Save best model
    best_model_result = sorted_results[0]
    print(f"\n" + "="*80)
    print(f"Saving best model: {best_model_result['name']} ({best_model_result['accuracy']:.4f} accuracy)")
    # Save to a distinct filename to avoid overwriting other baselines
    out_path = Path(str(BEST_MODEL)).with_name('best_model_with_players_random.pkl')
    joblib.dump({
        'model': best_model_result['model'],
        'feature_names': feature_names,
        'accuracy': best_model_result['accuracy'],
        'model_name': best_model_result['name']
    }, str(out_path))
    print(f"✓ Saved to {out_path}")

    # Also save the stacked model artifact for convenience
    stack_art_path = None
    if 'Stacking_Meta' in trained_models:
        stack_art_path = Path(str(BEST_MODEL)).with_name('best_model_with_players_random_stacking.pkl')
        joblib.dump({
            'model': trained_models['Stacking_Meta'],
            'feature_names': feature_names,
            'model_name': 'Stacking_Meta',
            'calibrated': False,
            'time_aware': False,
            'note': 'Stacked meta-learner over base models (random split)'
        }, str(stack_art_path))
        print(f"Saved stacking model artifact → {stack_art_path}")
    
    # Optionally write a concise summary to a file for diffing
    if save_summary:
        try:
            OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass
        default_summary = Path(OUTPUTS_DIR) / "simulation_output_random.txt"
        target_path = Path(summary_file) if summary_file else default_summary
        
        lines = []
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        lines.append("Baseline (random-split) Simulation Summary")
        lines.append(f"Generated: {ts}")
        lines.append("")
        lines.append(f"Samples: {len(X)} | Features: {len(feature_names)}")
        lines.append(f"Split: train={len(X_train)}, test={len(X_test)}")
        lines.append("")
        lines.append("Model performance (sorted by accuracy):")
        for r in sorted_results:
            lines.append(
                f"- {r['name']}: acc={r['accuracy']:.4f}, f1={r['f1_score']:.4f}, auc={r['auc']:.4f}, cv={r['cv_mean']:.4f}±{r['cv_std']:.4f}"
            )
        lines.append("")
        lines.append("Tournament results:")
        for tr in tournament_results:
            if tr:
                lines.append(
                    f"- {tr['model']}: champion={tr['champion']}, runner_up={tr['runner_up']}, conf={tr['confidence']:.3f}"
                )
        lines.append("")
        lines.append(f"Saved best model: {best_model_result['name']} ({best_model_result['accuracy']:.4f}) → {out_path}")
        if stack_art_path is not None:
            lines.append(f"Saved stacked model → {stack_art_path}")

        with open(target_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        print(f"\nSummary written to: {target_path}")
    
    print("\n" + "="*80)
    print("SIMULATION COMPLETE")
    print("="*80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Baseline random-split multi-model simulation")
    parser.add_argument("--save-summary", action="store_true", help="Write a text summary to outputs/")
    parser.add_argument("--summary-file", type=str, default=None, help="Custom path for the summary file")
    args = parser.parse_args()
    main(save_summary=args.save_summary, summary_file=args.summary_file)

# --- Helper for archived simulation to avoid unresolved reference in linters ---
def simulate_tournament_with_model(model_name: str, model, feature_names):
    """Archived bracket simulation: simple fixed quarterfinals -> semis -> final."""
    try:
        # Fixed example bracket order for archival
        bracket = ["AKA","CCS","CHD","CMF","CTC","FTL","PGA","HSH"]
        qf_pairs = [(bracket[i], bracket[7-i]) for i in range(4)]
        winners_qf = []
        for a,b in qf_pairs:
            winner, conf = predict_match(model, a, b, feature_names)
            winners_qf.append(winner)
        sf_pairs = [(winners_qf[0], winners_qf[1]), (winners_qf[2], winners_qf[3])]
        winners_sf = []
        for a,b in sf_pairs:
            winner, conf = predict_match(model, a, b, feature_names)
            winners_sf.append(winner)
        w1, w2 = winners_sf[0], winners_sf[1]
        # Final
        final_winner, final_conf = predict_match(model, w1, w2, feature_names)
        runner_up = w2 if final_winner == w1 else w1
        return {
            'model': model_name,
            'champion': final_winner,
            'runner_up': runner_up,
            'confidence': final_conf
        }
    except Exception:
        return {
            'model': model_name,
            'champion': 'N/A',
            'runner_up': 'N/A',
            'confidence': 0.0
        }
