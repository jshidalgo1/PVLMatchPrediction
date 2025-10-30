"""
Simulate Second Round Matches
Based on tournament_format.md:
- Top 3 from Pool A + Bottom 3 from Pool B = New Pool
- Top 3 from Pool B + Bottom 3 from Pool A = New Pool
- Teams only play opponents they haven't faced (3 matches each)
- Results carry over from first round
"""

import sqlite3
import pandas as pd
import joblib
from itertools import combinations


def get_first_round_standings():
    """Get first round results and standings"""
    conn = sqlite3.connect('volleyball_data.db')
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT t.code, t.name,
               SUM(CASE WHEN m.winner_id = t.id THEN 1 ELSE 0 END) as wins,
               SUM(CASE WHEN m.winner_id != t.id THEN 1 ELSE 0 END) as losses,
               COUNT(*) as games_played
        FROM teams t
        JOIN matches m ON (t.id = m.team_a_id OR t.id = m.team_b_id)
        JOIN tournaments tour ON m.tournament_id = tour.id
        WHERE tour.code = 'TEST_PVLR25'
        GROUP BY t.code, t.name
        ORDER BY wins DESC, losses ASC
    ''')
    
    standings = {}
    for row in cursor.fetchall():
        standings[row[0]] = {
            'name': row[1],
            'wins': row[2],
            'losses': row[3],
            'games_played': row[4],
            'win_pct': row[2] / row[4] if row[4] > 0 else 0
        }
    
    conn.close()
    return standings


def get_played_matchups():
    """Get all matchups that have been played"""
    conn = sqlite3.connect('volleyball_data.db')
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT ta.code, tb.code
        FROM matches m
        JOIN teams ta ON m.team_a_id = ta.id
        JOIN teams tb ON m.team_b_id = tb.id
        JOIN tournaments tour ON m.tournament_id = tour.id
        WHERE tour.code = 'TEST_PVLR25'
    ''')
    
    played = set()
    for row in cursor.fetchall():
        played.add(tuple(sorted([row[0], row[1]])))
    
    conn.close()
    return played


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


def main():
    print("="*80)
    print(" "*25 + "SECOND ROUND SIMULATION")
    print(" "*20 + "Based on tournament_format.md")
    print("="*80)
    
    # First round pools
    pool_a_original = ['HSH', 'FFF', 'CMF', 'CHD', 'CAP', 'NXL']
    pool_b_original = ['ZUS', 'CCS', 'AKA', 'PGA', 'CTC', 'GTH']
    
    # Get first round standings
    standings = get_first_round_standings()
    
    # Sort each pool by wins
    pool_a_sorted = sorted(
        [(team, standings.get(team, {'wins': 0, 'win_pct': 0})) for team in pool_a_original],
        key=lambda x: x[1]['win_pct'],
        reverse=True
    )
    
    pool_b_sorted = sorted(
        [(team, standings.get(team, {'wins': 0, 'win_pct': 0})) for team in pool_b_original],
        key=lambda x: x[1]['win_pct'],
        reverse=True
    )
    
    # Form second round pools
    # Pool C: Top 3 from Pool A + Bottom 3 from Pool B
    top_3_pool_a = [team[0] for team in pool_a_sorted[:3]]
    bottom_3_pool_b = [team[0] for team in pool_b_sorted[3:]]
    pool_c = top_3_pool_a + bottom_3_pool_b
    
    # Pool D: Top 3 from Pool B + Bottom 3 from Pool A
    top_3_pool_b = [team[0] for team in pool_b_sorted[:3]]
    bottom_3_pool_a = [team[0] for team in pool_a_sorted[3:]]
    pool_d = top_3_pool_b + bottom_3_pool_a
    
    print("\nSECOND ROUND POOL ASSIGNMENTS")
    print("="*80)
    
    print("\nPool C (Top 3 Pool A + Bottom 3 Pool B):")
    print(f"  From Pool A (Top 3): {', '.join(top_3_pool_a)}")
    print(f"  From Pool B (Bottom 3): {', '.join(bottom_3_pool_b)}")
    print(f"  → Pool C: {', '.join(pool_c)}")
    
    print("\nPool D (Top 3 Pool B + Bottom 3 Pool A):")
    print(f"  From Pool B (Top 3): {', '.join(top_3_pool_b)}")
    print(f"  From Pool A (Bottom 3): {', '.join(bottom_3_pool_a)}")
    print(f"  → Pool D: {', '.join(pool_d)}")
    
    # Load model
    print("\n" + "="*80)
    print("Loading best model...")
    model_data = joblib.load('best_model_with_players.pkl')
    model = model_data['model']
    feature_names = model_data['feature_names']
    print(f"✓ Model loaded")
    
    # Get already played matchups
    played_matchups = get_played_matchups()
    
    # Determine second round matches
    # Each team plays 3 matches against opponents from the other pool (that they haven't faced)
    print("\n" + "="*80)
    print("SECOND ROUND MATCHES (Teams play only NEW opponents)")
    print("="*80)
    
    second_round_matches = []
    
    # Pool C matches: Top 3 Pool A vs Bottom 3 Pool B
    for team_a in top_3_pool_a:
        for team_b in bottom_3_pool_b:
            matchup = tuple(sorted([team_a, team_b]))
            if matchup not in played_matchups:
                second_round_matches.append((team_a, team_b, 'Pool C'))
    
    # Pool D matches: Top 3 Pool B vs Bottom 3 Pool A
    for team_a in top_3_pool_b:
        for team_b in bottom_3_pool_a:
            matchup = tuple(sorted([team_a, team_b]))
            if matchup not in played_matchups:
                second_round_matches.append((team_a, team_b, 'Pool D'))
    
    print(f"\nTotal second round matches to simulate: {len(second_round_matches)}")
    
    # Simulate matches
    predictions = []
    
    print("\n" + "="*80)
    print("PREDICTING SECOND ROUND MATCHES")
    print("="*80)
    
    for team_a, team_b, pool in sorted(second_round_matches, key=lambda x: (x[2], x[0])):
        winner, confidence = predict_match(model, team_a, team_b, feature_names)
        loser = team_b if winner == team_a else team_a
        
        predictions.append({
            'team_a': team_a,
            'team_b': team_b,
            'winner': winner,
            'loser': loser,
            'confidence': confidence,
            'pool': pool
        })
        
        conf_bar = "█" * int(confidence * 20)
        print(f"\n[{pool}] {team_a} vs {team_b}")
        print(f"  → {winner} defeats {loser}")
        print(f"  Confidence: {confidence:.1%} {conf_bar}")
    
    # Calculate combined standings (First Round + Second Round)
    print("\n" + "="*80)
    print("COMBINED STANDINGS (First Round + Second Round)")
    print("="*80)
    
    # Start with first round results
    combined_standings = {team: dict(stats) for team, stats in standings.items()}
    
    # Add second round predictions
    for pred in predictions:
        if pred['winner'] in combined_standings:
            combined_standings[pred['winner']]['wins'] += 1
            combined_standings[pred['winner']]['games_played'] += 1
        if pred['loser'] in combined_standings:
            combined_standings[pred['loser']]['losses'] += 1
            combined_standings[pred['loser']]['games_played'] += 1
    
    # Recalculate win percentages
    for team in combined_standings:
        played = combined_standings[team]['games_played']
        combined_standings[team]['win_pct'] = combined_standings[team]['wins'] / played if played > 0 else 0
    
    # Sort by wins
    all_teams_sorted = sorted(
        combined_standings.items(),
        key=lambda x: (x[1]['wins'], x[1]['win_pct']),
        reverse=True
    )
    
    print(f"\n{'Rank':<6} {'Team':<6} {'Record':<10} {'Win %':<10} {'Games'}")
    print("-"*50)
    
    for i, (team, stats) in enumerate(all_teams_sorted, 1):
        record = f"{stats['wins']}-{stats['losses']}"
        emoji = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else "  "
        qualifier = "✓ QF" if i <= 8 else "❌ OUT"
        print(f"{emoji} {i:<3} {team:<6} {record:<10} {stats['win_pct']:.1%}      "
              f"{stats['games_played']:<6} {qualifier}")
    
    # Top 8 for quarterfinals
    print("\n" + "="*80)
    print("TOP 8 ADVANCING TO QUARTERFINALS")
    print("="*80)
    
    top_8 = all_teams_sorted[:8]
    
    print(f"\n{'Seed':<6} {'Team':<6} {'Record':<12} {'Win %'}")
    print("-"*40)
    
    for i, (team, stats) in enumerate(top_8, 1):
        record = f"{stats['wins']}-{stats['losses']}"
        emoji = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else "  "
        print(f"{emoji} #{i:<3} {team:<6} {record:<12} {stats['win_pct']:.1%}")
    
    # Show quarterfinal matchups
    print("\n" + "="*80)
    print("QUARTERFINAL MATCHUPS (Based on Seeding)")
    print("="*80)
    
    print(f"\nQF1: #{1} {top_8[0][0]} vs #{8} {top_8[7][0]}")
    print(f"QF2: #{2} {top_8[1][0]} vs #{7} {top_8[6][0]}")
    print(f"QF3: #{3} {top_8[2][0]} vs #{6} {top_8[5][0]}")
    print(f"QF4: #{4} {top_8[3][0]} vs #{5} {top_8[4][0]}")
    
    print("\n" + "="*80)
    print("SECOND ROUND SIMULATION COMPLETE")
    print("="*80)
    print(f"\n✓ {len(predictions)} second round matches predicted")
    print("✓ Combined standings calculated")
    print("✓ Top 8 qualified for quarterfinals")
    print("\nNext: Run multi_model_tournament_simulation.py for playoff predictions")


if __name__ == "__main__":
    main()
