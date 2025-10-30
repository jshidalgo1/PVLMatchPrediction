"""
COMPLETE PVL REINFORCED CONFERENCE 2025 TOURNAMENT SIMULATION
Simulates entire tournament: First Round → Second Round → Quarterfinals → Semifinals → Finals
Uses multiple algorithms (XGBoost, LightGBM, CatBoost, Random Forest, Voting Ensemble)
Based on tournament_format.md
"""

import sqlite3
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
import joblib
from itertools import combinations
import warnings
warnings.filterwarnings('ignore')

# Import visualization module
from tournament_visualization import save_tournament_visualizations


def load_features_with_players():
    """Load the feature set with player statistics"""
    X = pd.read_csv('X_features_with_players.csv')
    y = pd.read_csv('y_target_with_players.csv').values.ravel()
    return X, y


def train_models():
    """Train all models with player statistics"""
    print("\n" + "="*80)
    print(" "*25 + "TRAINING MODELS")
    print("="*80)
    
    X, y = load_features_with_players()
    feature_names = X.columns.tolist()
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"  Training set: {len(X_train)} samples")
    print(f"  Test set:     {len(X_test)} samples")
    print(f"  Features:     {len(feature_names)}")
    
    models = {
        'XGBoost_Deep': XGBClassifier(
            max_depth=10, learning_rate=0.03, n_estimators=300,
            subsample=0.8, colsample_bytree=0.8, random_state=42,
            eval_metric='logloss', verbosity=0
        ),
        'LightGBM': LGBMClassifier(
            num_leaves=70, max_depth=8, learning_rate=0.03, n_estimators=250,
            subsample=0.8, colsample_bytree=0.8, random_state=42, verbosity=-1
        ),
        'CatBoost': CatBoostClassifier(
            iterations=250, depth=8, learning_rate=0.03,
            l2_leaf_reg=3, random_state=42, verbose=False
        ),
        'Random_Forest': RandomForestClassifier(
            n_estimators=200, max_depth=10, min_samples_split=10,
            random_state=42, n_jobs=-1
        )
    }
    
    trained_models = {}
    
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"  ✓ {name}: {accuracy:.4f} accuracy")
        trained_models[name] = model
    
    # Voting Ensemble
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
    accuracy = accuracy_score(y_test, y_pred)
    print(f"  ✓ Voting_Ensemble: {accuracy:.4f} accuracy")
    trained_models['Voting_Ensemble'] = voting_clf
    
    return trained_models, feature_names


def get_team_features(team_code):
    """Get features for a team"""
    conn = sqlite3.connect('volleyball_data.db')
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
    
    return {
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


def get_h2h(team_a_code, team_b_code):
    """Get head-to-head record"""
    conn = sqlite3.connect('volleyball_data.db')
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
    
    result = cursor.fetchone()
    conn.close()
    
    h2h_matches = result[0] if result else 0
    h2h_team_a_wins = result[1] if result and result[1] else 0
    
    return h2h_matches, h2h_team_a_wins


def predict_match(model, team_a, team_b, feature_names):
    """Predict match outcome"""
    team_a_features = get_team_features(team_a)
    team_b_features = get_team_features(team_b)
    h2h_matches, h2h_team_a_wins = get_h2h(team_a, team_b)
    
    features = {}
    for key, value in team_a_features.items():
        features[f'team_a_{key}'] = value
    for key, value in team_b_features.items():
        features[f'team_b_{key}'] = value
    features['h2h_matches'] = h2h_matches
    features['h2h_team_a_wins'] = h2h_team_a_wins
    
    X = pd.DataFrame([features])
    
    for col in feature_names:
        if col not in X.columns:
            X[col] = 0
    
    X = X[feature_names]
    
    prediction = model.predict(X)[0]
    probability = model.predict_proba(X)[0]
    
    winner = team_a if prediction == 1 else team_b
    confidence = max(probability)
    
    return winner, confidence


def get_played_matchups():
    """Get all matchups that have been played"""
    conn = sqlite3.connect('volleyball_data.db')
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT ta.code, tb.code, m.team_a_sets_won, m.team_b_sets_won, t.code as winner
        FROM matches m
        JOIN teams ta ON m.team_a_id = ta.id
        JOIN teams tb ON m.team_b_id = tb.id
        JOIN teams t ON m.winner_id = t.id
        JOIN tournaments tour ON m.tournament_id = tour.id
        WHERE tour.code = 'TEST_PVLR25'
    ''')
    
    played = set()
    results = []
    for row in cursor.fetchall():
        played.add(tuple(sorted([row[0], row[1]])))
        results.append({
            'team_a': row[0], 'team_b': row[1],
            'sets_a': row[2], 'sets_b': row[3], 'winner': row[4]
        })
    
    conn.close()
    return played, results


def simulate_first_round(model, feature_names, standings):
    """Simulate remaining first round matches"""
    print("\n" + "="*80)
    print(" "*27 + "FIRST ROUND")
    print("="*80)
    
    pool_a = ['HSH', 'FFF', 'CMF', 'CHD', 'CAP', 'NXL']
    pool_b = ['ZUS', 'CCS', 'AKA', 'PGA', 'CTC', 'GTH']
    
    played_matchups, played_results = get_played_matchups()
    
    print(f"\n✓ {len(played_results)} matches already played")
    
    # Generate remaining matchups
    all_matchups = [(a, b, 'Pool A') for a, b in combinations(pool_a, 2)] + \
                   [(a, b, 'Pool B') for a, b in combinations(pool_b, 2)]
    
    remaining = [(a, b, p) for a, b, p in all_matchups 
                 if tuple(sorted([a, b])) not in played_matchups]
    
    print(f"✓ {len(remaining)} matches to simulate\n")
    
    predictions = []
    for team_a, team_b, pool in remaining:
        winner, confidence = predict_match(model, team_a, team_b, feature_names)
        loser = team_b if winner == team_a else team_a
        predictions.append({'winner': winner, 'loser': loser})
        standings[winner]['wins'] += 1
        standings[winner]['games_played'] += 1
        standings[loser]['losses'] += 1
        standings[loser]['games_played'] += 1
    
    # Update win percentages
    for team in standings:
        played = standings[team]['games_played']
        standings[team]['win_pct'] = standings[team]['wins'] / played if played > 0 else 0
    
    # Display Pool A standings
    print("POOL A FINAL STANDINGS:")
    pool_a_sorted = sorted([(t, standings[t]) for t in pool_a], 
                          key=lambda x: x[1]['win_pct'], reverse=True)
    for i, (team, stats) in enumerate(pool_a_sorted, 1):
        print(f"  {i}. {team}: {stats['wins']}-{stats['losses']} ({stats['win_pct']:.1%})")
    
    print("\nPOOL B FINAL STANDINGS:")
    pool_b_sorted = sorted([(t, standings[t]) for t in pool_b], 
                          key=lambda x: x[1]['win_pct'], reverse=True)
    for i, (team, stats) in enumerate(pool_b_sorted, 1):
        print(f"  {i}. {team}: {stats['wins']}-{stats['losses']} ({stats['win_pct']:.1%})")
    
    return pool_a_sorted, pool_b_sorted


def simulate_second_round(model, feature_names, pool_a_sorted, pool_b_sorted, standings):
    """Simulate second round matches"""
    print("\n" + "="*80)
    print(" "*26 + "SECOND ROUND")
    print("="*80)
    
    # Form new pools
    top_3_pool_a = [team[0] for team in pool_a_sorted[:3]]
    bottom_3_pool_b = [team[0] for team in pool_b_sorted[3:]]
    top_3_pool_b = [team[0] for team in pool_b_sorted[:3]]
    bottom_3_pool_a = [team[0] for team in pool_a_sorted[3:]]
    
    print(f"\nPool C: {', '.join(top_3_pool_a + bottom_3_pool_b)}")
    print(f"Pool D: {', '.join(top_3_pool_b + bottom_3_pool_a)}")
    
    played_matchups, _ = get_played_matchups()
    
    # Generate second round matches
    second_round_matches = []
    for team_a in top_3_pool_a:
        for team_b in bottom_3_pool_b:
            if tuple(sorted([team_a, team_b])) not in played_matchups:
                second_round_matches.append((team_a, team_b))
    
    for team_a in top_3_pool_b:
        for team_b in bottom_3_pool_a:
            if tuple(sorted([team_a, team_b])) not in played_matchups:
                second_round_matches.append((team_a, team_b))
    
    print(f"\n✓ {len(second_round_matches)} second round matches to simulate\n")
    
    # Simulate matches
    print("SECOND ROUND MATCH RESULTS:")
    print("-"*60)
    
    pool_c_matches = []
    pool_d_matches = []
    
    for team_a, team_b in second_round_matches:
        winner, confidence = predict_match(model, team_a, team_b, feature_names)
        loser = team_b if winner == team_a else team_a
        
        # Determine which pool
        if team_a in top_3_pool_a or team_b in top_3_pool_a:
            pool = 'Pool C'
            pool_c_matches.append((team_a, team_b, winner, confidence))
        else:
            pool = 'Pool D'
            pool_d_matches.append((team_a, team_b, winner, confidence))
        
        standings[winner]['wins'] += 1
        standings[winner]['games_played'] += 1
        standings[loser]['losses'] += 1
        standings[loser]['games_played'] += 1
    
    # Display Pool C matches
    print("\nPool C Matches:")
    for team_a, team_b, winner, confidence in sorted(pool_c_matches):
        loser = team_b if winner == team_a else team_a
        print(f"  {team_a} vs {team_b} → {winner} defeats {loser} ({confidence:.1%})")
    
    # Display Pool D matches
    print("\nPool D Matches:")
    for team_a, team_b, winner, confidence in sorted(pool_d_matches):
        loser = team_b if winner == team_a else team_a
        print(f"  {team_a} vs {team_b} → {winner} defeats {loser} ({confidence:.1%})")
    
    # Update win percentages
    for team in standings:
        played = standings[team]['games_played']
        standings[team]['win_pct'] = standings[team]['wins'] / played if played > 0 else 0
    
    # Display Pool C standings
    print("\n" + "="*80)
    print("POOL C STANDINGS (After Second Round):")
    print("-"*50)
    
    pool_c_teams = top_3_pool_a + bottom_3_pool_b
    pool_c_sorted = sorted([(t, standings[t]) for t in pool_c_teams],
                          key=lambda x: (x[1]['wins'], x[1]['win_pct']),
                          reverse=True)
    
    print(f"{'Rank':<6} {'Team':<6} {'Record':<10} {'Win %':<10} {'Games'}")
    print("-"*50)
    for i, (team, stats) in enumerate(pool_c_sorted, 1):
        record = f"{stats['wins']}-{stats['losses']}"
        emoji = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else "  "
        print(f"{emoji} {i:<3} {team:<6} {record:<10} {stats['win_pct']:.1%}      {stats['games_played']}")
    
    # Display Pool D standings
    print("\n" + "="*80)
    print("POOL D STANDINGS (After Second Round):")
    print("-"*50)
    
    pool_d_teams = top_3_pool_b + bottom_3_pool_a
    pool_d_sorted = sorted([(t, standings[t]) for t in pool_d_teams],
                          key=lambda x: (x[1]['wins'], x[1]['win_pct']),
                          reverse=True)
    
    print(f"{'Rank':<6} {'Team':<6} {'Record':<10} {'Win %':<10} {'Games'}")
    print("-"*50)
    for i, (team, stats) in enumerate(pool_d_sorted, 1):
        record = f"{stats['wins']}-{stats['losses']}"
        emoji = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else "  "
        print(f"{emoji} {i:<3} {team:<6} {record:<10} {stats['win_pct']:.1%}      {stats['games_played']}")
    
    # Display final standings
    all_teams_sorted = sorted(standings.items(), 
                             key=lambda x: (x[1]['wins'], x[1]['win_pct']), 
                             reverse=True)
    
    print("COMBINED STANDINGS (After First + Second Round):")
    print(f"{'Rank':<6} {'Team':<6} {'Record':<10} {'Win %':<10} {'Status'}")
    print("-"*50)
    
    for i, (team, stats) in enumerate(all_teams_sorted, 1):
        record = f"{stats['wins']}-{stats['losses']}"
        status = "✓ QF" if i <= 8 else "❌ OUT"
        emoji = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else "  "
        print(f"{emoji} {i:<3} {team:<6} {record:<10} {stats['win_pct']:.1%}      {status}")
    
    return all_teams_sorted[:8], all_teams_sorted


def simulate_playoffs(model_name, model, feature_names, top_8):
    """Simulate quarterfinals, semifinals, and finals"""
    print("\n" + "="*80)
    print(f" "*20 + f"PLAYOFFS - {model_name.upper()}")
    print("="*80)
    
    # Quarterfinals
    print("\n" + "─"*80)
    print(" "*30 + "QUARTERFINALS")
    print("─"*80)
    
    qf_matchups = [
        (top_8[0][0], top_8[7][0], 'QF1', '#1 vs #8'),
        (top_8[1][0], top_8[6][0], 'QF2', '#2 vs #7'),
        (top_8[2][0], top_8[5][0], 'QF3', '#3 vs #6'),
        (top_8[3][0], top_8[4][0], 'QF4', '#4 vs #5')
    ]
    
    qf_winners = []
    for team_a, team_b, qf_name, seed_info in qf_matchups:
        winner, confidence = predict_match(model, team_a, team_b, feature_names)
        qf_winners.append(winner)
        loser = team_b if winner == team_a else team_a
        print(f"{qf_name} ({seed_info}): {team_a} vs {team_b} → {winner} wins ({confidence:.1%})")
    
    # Semifinals
    print("\n" + "─"*80)
    print(" "*31 + "SEMIFINALS")
    print("─"*80)
    
    sf_matchups = [
        (qf_winners[0], qf_winners[3], 'SF1', 'QF1 vs QF4'),
        (qf_winners[1], qf_winners[2], 'SF2', 'QF2 vs QF3')
    ]
    
    sf_winners = []
    sf_losers = []
    for team_a, team_b, sf_name, matchup_info in sf_matchups:
        winner, confidence = predict_match(model, team_a, team_b, feature_names)
        sf_winners.append(winner)
        loser = team_b if winner == team_a else team_a
        sf_losers.append(loser)
        print(f"{sf_name} ({matchup_info}): {team_a} vs {team_b} → {winner} wins ({confidence:.1%})")
    
    # Third Place Match
    print("\n" + "─"*80)
    print(" "*28 + "THIRD PLACE MATCH")
    print("─"*80)
    
    bronze_winner, confidence = predict_match(model, sf_losers[0], sf_losers[1], feature_names)
    bronze_loser = sf_losers[1] if bronze_winner == sf_losers[0] else sf_losers[0]
    print(f"{sf_losers[0]} vs {sf_losers[1]} → {bronze_winner} wins bronze ({confidence:.1%})")
    
    # Finals
    print("\n" + "─"*80)
    print(" "*28 + "CHAMPIONSHIP MATCH")
    print("─"*80)
    
    champion, confidence = predict_match(model, sf_winners[0], sf_winners[1], feature_names)
    runner_up = sf_winners[1] if champion == sf_winners[0] else sf_winners[0]
    print(f"{sf_winners[0]} vs {sf_winners[1]} → {champion} wins gold ({confidence:.1%})")
    
    print("\n" + "="*80)
    print(" "*28 + "FINAL STANDINGS")
    print("="*80)
    print(f"  🥇 CHAMPION:     {champion}")
    print(f"  🥈 RUNNER-UP:    {runner_up}")
    print(f"  🥉 THIRD PLACE:  {bronze_winner}")
    print(f"     FOURTH PLACE: {bronze_loser}")
    
    return {
        'model': model_name,
        'champion': champion,
        'runner_up': runner_up,
        'third': bronze_winner,
        'fourth': bronze_loser,
        'confidence': confidence
    }


def main():
    print("\n" + "="*80)
    print(" "*10 + "PVL REINFORCED CONFERENCE 2025 - COMPLETE SIMULATION")
    print(" "*15 + "Multi-Model Tournament Prediction System")
    print("="*80)
    
    # Train models
    trained_models, feature_names = train_models()
    
    # Initialize standings from database
    conn = sqlite3.connect('volleyball_data.db')
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT t.code,
               SUM(CASE WHEN m.winner_id = t.id THEN 1 ELSE 0 END) as wins,
               SUM(CASE WHEN m.winner_id != t.id THEN 1 ELSE 0 END) as losses,
               COUNT(*) as games_played
        FROM teams t
        JOIN matches m ON (t.id = m.team_a_id OR t.id = m.team_b_id)
        JOIN tournaments tour ON m.tournament_id = tour.id
        WHERE tour.code = 'TEST_PVLR25'
        GROUP BY t.code
    ''')
    
    standings = {}
    for row in cursor.fetchall():
        standings[row[0]] = {
            'wins': row[1], 'losses': row[2], 
            'games_played': row[3],
            'win_pct': row[1] / row[3] if row[3] > 0 else 0
        }
    
    # Add teams that haven't played yet
    all_teams = ['HSH', 'FFF', 'CMF', 'CHD', 'CAP', 'NXL', 
                 'ZUS', 'CCS', 'AKA', 'PGA', 'CTC', 'GTH']
    for team in all_teams:
        if team not in standings:
            standings[team] = {'wins': 0, 'losses': 0, 'games_played': 0, 'win_pct': 0}
    
    conn.close()
    
    # Use Voting Ensemble (best model) for preliminary rounds
    model = trained_models['Voting_Ensemble']
    
    # Simulate First Round
    pool_a_sorted, pool_b_sorted = simulate_first_round(model, feature_names, standings)
    
    # Simulate Second Round
    top_8, all_teams_sorted = simulate_second_round(model, feature_names, pool_a_sorted, pool_b_sorted, standings)
    
    # Simulate playoffs with all models
    print("\n" + "="*80)
    print(" "*22 + "MULTI-MODEL PLAYOFF PREDICTIONS")
    print("="*80)
    
    all_results = []
    for model_name, model in trained_models.items():
        result = simulate_playoffs(model_name, model, feature_names, top_8)
        all_results.append(result)
    
    # Consensus Analysis
    print("\n" + "="*80)
    print(" "*28 + "CONSENSUS ANALYSIS")
    print("="*80)
    
    champions = [r['champion'] for r in all_results]
    champion_counts = {}
    for champ in champions:
        champion_counts[champ] = champion_counts.get(champ, 0) + 1
    
    print(f"\n{'Model':<25} {'Champion':<10} {'Runner-Up':<10} {'Third':<10}")
    print("-"*60)
    for result in all_results:
        print(f"{result['model']:<25} {result['champion']:<10} "
              f"{result['runner_up']:<10} {result['third']:<10}")
    
    print("\nChampion predictions:")
    for team, count in sorted(champion_counts.items(), key=lambda x: x[1], reverse=True):
        percentage = (count / len(all_results)) * 100
        bar = "█" * int(percentage / 5)
        print(f"  {team}: {count}/{len(all_results)} models ({percentage:.0f}%) {bar}")
    
    most_common_champ = max(champion_counts.items(), key=lambda x: x[1])
    
    print("\n" + "="*80)
    if most_common_champ[1] == len(all_results):
        print(f"  ✓ UNANIMOUS PREDICTION: {most_common_champ[0]} wins the championship!")
    elif most_common_champ[1] >= len(all_results) * 0.6:
        print(f"  ✓ STRONG CONSENSUS: {most_common_champ[0]} ({most_common_champ[1]}/{len(all_results)} models)")
    else:
        print(f"  ⚠️  SPLIT DECISION: {most_common_champ[0]} leads ({most_common_champ[1]}/{len(all_results)} models)")
    print("="*80)
    
    print("\n" + "="*80)
    print(" "*25 + "SIMULATION COMPLETE!")
    print("="*80)
    
    # Generate visualizations
    save_tournament_visualizations(top_8, all_results, pool_a_sorted, 
                                   pool_b_sorted, all_teams_sorted)


if __name__ == "__main__":
    main()
