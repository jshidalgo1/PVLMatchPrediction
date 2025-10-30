"""
Multi-Model Tournament Simulation
Trains XGBoost, CatBoost, LightGBM, and ensemble models with player statistics
Then simulates the tournament with each model to compare predictions
"""

import pandas as pd
import numpy as np
import sqlite3
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import VotingClassifier, RandomForestClassifier, GradientBoostingClassifier
import joblib
import warnings
from config import X_FEATURES, Y_TARGET, DB_FILE, BEST_MODEL
warnings.filterwarnings('ignore')


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


def simulate_tournament_with_model(model_name, model, feature_names):
    """Simulate tournament bracket with a specific model using proper seeding"""
    
    print(f"\n{'='*80}")
    print(f"SIMULATING WITH {model_name}")
    print(f"{'='*80}")
    
    # Get seeding from database
    top_8 = get_current_standings_and_seeding()
    
    if len(top_8) < 8:
        print(f"⚠️  Warning: Only {len(top_8)} teams found in database")
        return None
    
    print("\nTOP 8 SEEDING:")
    for i, team in enumerate(top_8, 1):
        print(f"  #{i} {team['code']}: {team['wins']}-{team['losses']} ({team['win_pct']:.1%})")
    
    # Quarterfinals - proper seeding (#1 vs #8, #2 vs #7, #3 vs #6, #4 vs #5)
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
    
    # Semifinals - QF1 winner vs QF4 winner, QF2 winner vs QF3 winner
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
    
    # Finals
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


def main():
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
    joblib.dump({
        'model': best_model_result['model'],
        'feature_names': feature_names,
        'accuracy': best_model_result['accuracy'],
        'model_name': best_model_result['name']
    }, str(BEST_MODEL))
    print(f"✓ Saved to {BEST_MODEL}")
    
    print("\n" + "="*80)
    print("SIMULATION COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()
