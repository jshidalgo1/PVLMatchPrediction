"""
ARCHIVED: Time-Aware Multi-Model Tournament Simulation (Chronological Split + Calibration)
Moved to scripts/archive/ to declutter active simulation entrypoints.
"""

import pandas as pd
import numpy as np
import sqlite3
from pathlib import Path
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, log_loss, brier_score_loss
from sklearn.calibration import CalibratedClassifierCV
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
import joblib
import argparse
from datetime import datetime
from config import X_FEATURES, Y_TARGET, DB_FILE, BEST_MODEL, OUTPUTS_DIR
import warnings
warnings.filterwarnings('ignore')


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


def load_player_features():
    X = pd.read_csv(X_FEATURES)
    y = pd.read_csv(Y_TARGET).values.ravel()
    return X, y


def chronological_split(X, y, train_ratio=0.6, calib_ratio=0.2):
    n = len(X)
    train_end = int(n * train_ratio)
    calib_end = int(n * (train_ratio + calib_ratio))
    X_train, y_train = X.iloc[:train_end], y[:train_end]
    X_calib, y_calib = X.iloc[train_end:calib_end], y[train_end:calib_end]
    X_test, y_test = X.iloc[calib_end:], y[calib_end:]
    return X_train, y_train, X_calib, y_calib, X_test, y_test


def train_models_timeaware(X_train, y_train, X_test, y_test):
    models = {
        'XGBoost': XGBClassifier(
            max_depth=7, learning_rate=0.05, n_estimators=250,
            subsample=0.8, colsample_bytree=0.8, random_state=42,
            eval_metric='logloss', verbosity=0
        ),
        'LightGBM': LGBMClassifier(
            num_leaves=70, max_depth=8, learning_rate=0.03, n_estimators=300,
            subsample=0.8, colsample_bytree=0.8, random_state=42, verbosity=-1
        ),
        'CatBoost': CatBoostClassifier(
            iterations=300, depth=8, learning_rate=0.03, l2_leaf_reg=3,
            random_state=42, verbose=False
        ),
        'Random_Forest': RandomForestClassifier(
            n_estimators=250, max_depth=12, min_samples_split=8, random_state=42, n_jobs=-1
        ),
        'Gradient_Boosting': GradientBoostingClassifier(
            n_estimators=250, learning_rate=0.05, max_depth=7, random_state=42
        )
    }
    
    trained = {}
    perf = []
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        perf.append({
            'name': name,
            'accuracy': accuracy_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred),
            'auc': roc_auc_score(y_test, y_proba),
            'logloss': log_loss(y_test, y_proba),
            'brier': brier_score_loss(y_test, y_proba)
        })
        trained[name] = model
    return perf, trained


def train_stacking_timeaware(trained_bases, X_train, y_train, X_test, y_test):
    order = list(trained_bases.keys())
    # OOF generation
    n = len(X_train)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof = {name: np.full(n, np.nan) for name in order}
    for tr_idx, val_idx in skf.split(X_train, y_train):
        X_tr, X_val = X_train.iloc[tr_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train[tr_idx], y_train[val_idx]
        for name in order:
            base = trained_bases[name].__class__(**trained_bases[name].get_params())
            base.fit(X_tr, y_tr)
            oof[name][val_idx] = base.predict_proba(X_val)[:, 1]
    X_oof = np.column_stack([oof[name] for name in order])
    meta = LogisticRegression(max_iter=1000)
    meta.fit(X_oof, y_train)
    # Fit full bases
    fitted = {}
    for name in order:
        base = trained_bases[name].__class__(**trained_bases[name].get_params())
        base.fit(X_train, y_train)
        fitted[name] = base
    stack = StackingModel(fitted, meta, order)
    y_proba = stack.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= 0.5).astype(int)
    return stack, {
        'name': 'Stacking_Meta',
        'accuracy': accuracy_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'auc': roc_auc_score(y_test, y_proba),
        'logloss': log_loss(y_test, y_proba),
        'brier': brier_score_loss(y_test, y_proba)
    }


def get_team_snapshot(team_code):
    conn = sqlite3.connect(str(DB_FILE))
    cursor = conn.cursor()
    cursor.execute('''
        SELECT 
            COUNT(*) as matches_played,
            SUM(CASE WHEN m.winner_id = t.id THEN 1 ELSE 0 END) as matches_won,
            SUM(CASE WHEN m.team_a_id = t.id THEN m.team_a_sets_won ELSE m.team_b_sets_won END) as sets_won,
            SUM(CASE WHEN m.team_a_id = t.id THEN m.team_b_sets_won ELSE m.team_a_sets_won END) as sets_lost
        FROM teams t
        LEFT JOIN matches m ON (t.id = m.team_a_id OR t.id = m.team_b_id)
        WHERE t.code = ?
        GROUP BY t.id
    ''', (team_code,))
    r = cursor.fetchone()
    # Player stats snapshot
    cursor.execute('''
        SELECT 
            AVG(pms.attack_points + pms.block_points + pms.serve_points),
            AVG(pms.attack_points),
            COUNT(DISTINCT pms.player_id)
        FROM player_match_stats pms
        JOIN matches m ON pms.match_id = m.id
        JOIN teams t ON t.code = ?
        WHERE (m.team_a_id = t.id OR m.team_b_id = t.id)
        AND (pms.attack_points + pms.block_points + pms.serve_points) > 5
    ''', (team_code,))
    pr = cursor.fetchone()
    conn.close()
    if not r:
        return {
            'prev_matches_played': 0,'prev_matches_won':0,'prev_win_rate':0,
            'prev_sets_won':0,'prev_sets_lost':0,'prev_set_win_rate':0,
            'avg_top_scorer_points':0,'avg_top_scorer_attacks':0,'num_regular_scorers':0
        }
    mp, mw, sw, sl = r
    pts, att, reg = pr if pr else (0,0,0)
    return {
        'prev_matches_played': mp,
        'prev_matches_won': mw,
        'prev_win_rate': mw/mp if mp else 0,
        'prev_sets_won': sw or 0,
        'prev_sets_lost': sl or 0,
        'prev_set_win_rate': (sw/(sw+sl)) if (sw+sl)>0 else 0,
        'avg_top_scorer_points': pts or 0,
        'avg_top_scorer_attacks': att or 0,
        'num_regular_scorers': reg or 0
    }


def build_feature_vector(team_a_code, team_b_code, feature_names):
    ta = get_team_snapshot(team_a_code)
    tb = get_team_snapshot(team_b_code)
    conn = sqlite3.connect(str(DB_FILE))
    cursor = conn.cursor()
    cursor.execute('''
        SELECT COUNT(*),
               SUM(CASE WHEN m.winner_id = ta.id THEN 1 ELSE 0 END)
        FROM matches m
        JOIN teams ta ON (ta.code = ? AND (m.team_a_id = ta.id OR m.team_b_id = ta.id))
        JOIN teams tb ON (tb.code = ? AND (m.team_a_id = tb.id OR m.team_b_id = tb.id))
        WHERE (m.team_a_id = ta.id AND m.team_b_id = tb.id) OR (m.team_a_id = tb.id AND m.team_b_id = ta.id)
    ''', (team_a_code, team_b_code))
    h2h = cursor.fetchone()
    conn.close()
    h2h_matches = h2h[0] if h2h else 0
    h2h_team_a_wins = h2h[1] if h2h and h2h[1] else 0
    feat = {}
    for k,v in ta.items(): feat[f'team_a_{k}']=v
    for k,v in tb.items(): feat[f'team_b_{k}']=v
    feat['h2h_matches']=h2h_matches
    feat['h2h_team_a_wins']=h2h_team_a_wins
    X = pd.DataFrame([feat])
    for col in feature_names:
        if col not in X.columns:
            X[col]=0
    return X[feature_names]


def simulate_bracket(model, feature_names):
    # Placeholder bracket ordering for archival; logic mirrors active script conceptually
    bracket = ["AKA","CCS","CHD","CMF","CTC","FTL","PGA","HSH"]
    qf_pairs = [(bracket[i], bracket[7-i]) for i in range(4)]
    winners_qf = []
    for a,b in qf_pairs:
        X = build_feature_vector(a,b,feature_names)
        proba = model.predict_proba(X)[0][1]
        winner = a if proba >= 0.5 else b
        winners_qf.append(winner)
    sf_pairs = [(winners_qf[0], winners_qf[1]), (winners_qf[2], winners_qf[3])]
    winners_sf = []
    for a,b in sf_pairs:
        X = build_feature_vector(a,b,feature_names)
        proba = model.predict_proba(X)[0][1]
        winner = a if proba >= 0.5 else b
        winners_sf.append(winner)
    X_final = build_feature_vector(winners_sf[0], winners_sf[1], feature_names)
    proba_final = model.predict_proba(X_final)[0][1]
    champ = winners_sf[0] if proba_final >= 0.5 else winners_sf[1]
    runner_up = winners_sf[1] if champ == winners_sf[0] else winners_sf[0]
    confidence = max(proba_final, 1-proba_final)
    return champ, runner_up, confidence


def main(save_summary: bool=False, summary_file: str|None=None):
    X, y = load_player_features()
    feature_names = X.columns.tolist()
    X_train, y_train, X_calib, y_calib, X_test, y_test = chronological_split(X, y)
    perf, trained = train_models_timeaware(X_train, y_train, X_test, y_test)
    stack_model, stack_perf = train_stacking_timeaware(trained, X_train, y_train, X_test, y_test)
    perf.append(stack_perf)
    # Calibration on mid slice for best model (example using XGBoost)
    best = sorted(perf, key=lambda d: d['accuracy'], reverse=True)[0]
    base_best = trained[best['name']] if best['name'] != 'Stacking_Meta' else stack_model
    cal = CalibratedClassifierCV(base_best, cv='prefit', method='sigmoid')
    cal.fit(X_calib, y_calib)
    y_proba_cal = cal.predict_proba(X_test)[:,1]
    cal_metrics = {
        'accuracy': accuracy_score(y_test, (y_proba_cal>=0.5).astype(int)),
        'logloss': log_loss(y_test, y_proba_cal),
        'brier': brier_score_loss(y_test, y_proba_cal)
    }
    champ, runner_up, conf = simulate_bracket(cal, feature_names)
    out_path = Path(str(BEST_MODEL)).with_name('best_model_with_players_timeaware.pkl')
    joblib.dump({
        'model': cal,
        'feature_names': feature_names,
        'calibrated': True,
        'time_aware': True,
        'champion_prediction': champ,
        'runner_up_prediction': runner_up,
        'confidence': conf,
        'metrics': cal_metrics
    }, out_path)
    if save_summary:
        OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
        lines = []
        ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        lines.append('Time-Aware Simulation Summary (ARCHIVED)')
        lines.append(f'Generated: {ts}')
        lines.append(f'Samples: {len(X)} | Features: {len(feature_names)}')
        lines.append(f'Splits: train={len(X_train)}, calib={len(X_calib)}, test={len(X_test)}')
        lines.append('')
        lines.append('Model performance:')
        for p in sorted(perf, key=lambda d: d['accuracy'], reverse=True):
            lines.append(f"- {p['name']}: acc={p['accuracy']:.4f}, f1={p.get('f1',0):.4f}, auc={p.get('auc',0):.4f}")
        lines.append('')
        lines.append(f"Calibrated best model ({best['name']}) → acc={cal_metrics['accuracy']:.4f}, logloss={cal_metrics['logloss']:.4f}, brier={cal_metrics['brier']:.4f}")
        lines.append(f"Champion: {champ} over {runner_up} (confidence={conf:.2%})")
        summ_path = Path(summary_file) if summary_file else Path(OUTPUTS_DIR)/'simulation_output_timeaware.txt'
        with open(summ_path,'w',encoding='utf-8') as f: f.write('\n'.join(lines))
    print('ARCHIVED time-aware simulation complete.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Archived time-aware simulation with player stats.')
    parser.add_argument('--save-summary', action='store_true')
    parser.add_argument('--summary-file', type=str, default=None)
    args = parser.parse_args()
    main(save_summary=args.save_summary, summary_file=args.summary_file)
