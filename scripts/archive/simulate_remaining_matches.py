"""
ARCHIVED: Simulate Remaining First Round Matches
This file was moved from scripts/ to scripts/archive/ to keep only full-tournament
simulation in the active scripts folder. Functionality preserved for reference.
"""

# Original content preserved below

"""
Simulate Remaining First Round Matches
Uses the best Voting Ensemble model to predict remaining pool matches

Update:
- Optional model_path override via CLI or function argument
"""

import sqlite3
import argparse
import pandas as pd
import joblib
from itertools import combinations

from config import DB_FILE_STR, BEST_MODEL_STR  # Simplified import (archive script)


def get_played_matches():
    """Get all matches already played in the tournament"""
    conn = sqlite3.connect(DB_FILE_STR)
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT ta.code, tb.code, 
               m.team_a_sets_won, m.team_b_sets_won,
               t.code as winner
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
        team_a, team_b = row[0], row[1]
        played.add(tuple(sorted([team_a, team_b])))
        results.append({
            'team_a': team_a,
            'team_b': team_b,
            'sets_a': row[2],
            'sets_b': row[3],
            'winner': row[4]
        })
    
    conn.close()
    return played, results


def get_team_features(team_code):
    """Get features for a team"""
    conn = sqlite3.connect(DB_FILE_STR)
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
    conn = sqlite3.connect(DB_FILE_STR)
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
    
    winner = team_a if prediction == 1 else team_b
    confidence = max(probability)
    
    return winner, confidence


def main(model_path: str | None = None):
    print("="*80)
    print(" "*20 + "SIMULATE REMAINING FIRST ROUND MATCHES (ARCHIVED)")
    print("="*80)
    
    # Define pools
    pool_a = ['HSH', 'FFF', 'CMF', 'CHD', 'CAP', 'NXL']
    pool_b = ['ZUS', 'CCS', 'AKA', 'PGA', 'CTC', 'GTH']
    
    # Load model (allow override)
    chosen_model_path = model_path or BEST_MODEL_STR
    print(f"\nLoading model from: {chosen_model_path}")
    model_data = joblib.load(chosen_model_path)
    model = model_data['model']
    feature_names = model_data['feature_names']
    print(f"✓ Model loaded (accuracy: {model_data.get('accuracy', 'N/A')})")
    
    # Get already played matches
    played_matchups, played_results = get_played_matches()
    
    print(f"\n✓ {len(played_results)} matches already played")
    
    # Generate all possible matchups for each pool
    pool_a_matchups = list(combinations(pool_a, 2))
    pool_b_matchups = list(combinations(pool_b, 2))
    
    all_matchups = [
        (team_a, team_b, 'Pool A') for team_a, team_b in pool_a_matchups
    ] + [
        (team_a, team_b, 'Pool B') for team_a, team_b in pool_b_matchups
    ]
    
    # Find unplayed matches
    remaining = []
    for team_a, team_b, pool in all_matchups:
        matchup = tuple(sorted([team_a, team_b]))
        if matchup not in played_matchups:
            remaining.append((team_a, team_b, pool))
    
    print(f"✓ {len(remaining)} matches remaining to simulate")
    
    # Show played matches
    print("\n" + "="*80)
    print("MATCHES ALREADY PLAYED")
    print("="*80)
    for result in played_results:
        print(f"  {result['team_a']} vs {result['team_b']}: "
              f"{result['winner']} wins {result['sets_a']}-{result['sets_b']}")
    
    # Predict remaining matches
    print("\n" + "="*80)
    print("PREDICTING REMAINING MATCHES")
    print("="*80)
    
    predictions_pool_a = []
    predictions_pool_b = []
    
    for team_a, team_b, pool in sorted(remaining, key=lambda x: (x[2], x[0])):
        winner, confidence = predict_match(model, team_a, team_b, feature_names)
        loser = team_b if winner == team_a else team_a
        
        prediction = {
            'team_a': team_a,
            'team_b': team_b,
            'winner': winner,
            'loser': loser,
            'confidence': confidence,
            'pool': pool
        }
        
        if pool == 'Pool A':
            predictions_pool_a.append(prediction)
        else:
            predictions_pool_b.append(prediction)
        
        conf_str = "█" * int(confidence * 20)
        print(f"\n[{pool}] {team_a} vs {team_b}")
        print(f"  → {winner} defeats {loser}")
        print(f"  Confidence: {confidence:.1%} {conf_str}")
    
    print("\n" + "="*80)
    print("SIMULATION COMPLETE (ARCHIVED)")
    print("="*80)
