"""
Simulate remaining first-round matches and predict final tournament outcomes.
Based on tournament_format.md and current standings.
"""

import sqlite3
import joblib
import pandas as pd
import numpy as np

def get_current_standings():
    """Get current match records from database."""
    conn = sqlite3.connect('volleyball_data.db')
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
        ORDER BY t.code
    ''')
    
    standings = {}
    for row in cursor.fetchall():
        standings[row[0]] = {
            'name': row[1],
            'wins': row[2],
            'losses': row[3],
            'games_played': row[4]
        }
    
    conn.close()
    return standings

def identify_missing_matches():
    """Identify which first-round matches haven't been played yet."""
    # Based on the image:
    # Pool A: HSH, FFF, CMF, CHD, CAP, NXL (each should play 5 matches)
    # Pool B: ZUS, CCS, AKA, PGA, CTC, GTH (each should play 5 matches)
    
    pool_a = ['HSH', 'FFF', 'CMF', 'CHD', 'CAP', 'NXL']
    pool_b = ['ZUS', 'CCS', 'AKA', 'PGA', 'CTC', 'GTH']
    
    conn = sqlite3.connect('volleyball_data.db')
    cursor = conn.cursor()
    
    # Get all matchups that have been played
    cursor.execute('''
        SELECT ta.code, tb.code
        FROM matches m
        JOIN teams ta ON m.team_a_id = ta.id
        JOIN teams tb ON m.team_b_id = tb.id
        WHERE m.tournament_id = (SELECT id FROM tournaments WHERE code = 'TEST_PVLR25')
    ''')
    
    played_matchups = set()
    for row in cursor.fetchall():
        team_a, team_b = row[0], row[1]
        played_matchups.add(tuple(sorted([team_a, team_b])))
    
    conn.close()
    
    # Find missing Pool A matches
    missing_matches = []
    for pool, pool_name in [(pool_a, 'Pool A'), (pool_b, 'Pool B')]:
        for i, team_a in enumerate(pool):
            for team_b in pool[i+1:]:
                matchup = tuple(sorted([team_a, team_b]))
                if matchup not in played_matchups:
                    missing_matches.append({
                        'team_a': team_a,
                        'team_b': team_b,
                        'pool': pool_name
                    })
    
    return missing_matches

def get_team_historical_features(team_code):
    """Get historical performance features for a team up to current point."""
    conn = sqlite3.connect('volleyball_data.db')
    cursor = conn.cursor()
    
    # Get team's historical performance
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
    conn.close()
    
    if not result or result[0] == 0:  # No history
        return {
            'prev_matches_played': 0,
            'prev_matches_won': 0,
            'prev_win_rate': 0,
            'prev_sets_won': 0,
            'prev_sets_lost': 0,
            'prev_set_win_rate': 0,
            'prev_avg_points_scored': 0,
            'prev_avg_points_conceded': 0
        }
    
    matches_played, matches_won, sets_won, sets_lost, avg_pts_scored, avg_pts_conceded = result
    
    return {
        'prev_matches_played': matches_played,
        'prev_matches_won': matches_won,
        'prev_win_rate': matches_won / matches_played if matches_played > 0 else 0,
        'prev_sets_won': sets_won or 0,
        'prev_sets_lost': sets_lost or 0,
        'prev_set_win_rate': sets_won / (sets_won + sets_lost) if (sets_won + sets_lost) > 0 else 0,
        'prev_avg_points_scored': avg_pts_scored or 0,
        'prev_avg_points_conceded': avg_pts_conceded or 0
    }

def get_head_to_head_features(team_a_code, team_b_code):
    """Get head-to-head record between two teams."""
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
    
    if not result or result[0] == 0:
        return {'h2h_matches': 0, 'h2h_team_a_wins': 0}
    
    return {
        'h2h_matches': result[0],
        'h2h_team_a_wins': result[1] or 0
    }

def predict_match(model, team_a_code, team_b_code):
    """Predict the winner of a match using the trained model."""
    # Get historical features for both teams
    team_a_features = get_team_historical_features(team_a_code)
    team_b_features = get_team_historical_features(team_b_code)
    h2h_features = get_head_to_head_features(team_a_code, team_b_code)
    
    # Combine features with correct naming
    features = {}
    for key, value in team_a_features.items():
        features[f'team_a_{key}'] = value
    for key, value in team_b_features.items():
        features[f'team_b_{key}'] = value
    features.update(h2h_features)
    
    # Create DataFrame with features in correct order
    feature_cols = [
        'team_a_prev_matches_played', 'team_a_prev_matches_won', 'team_a_prev_win_rate',
        'team_a_prev_sets_won', 'team_a_prev_sets_lost', 'team_a_prev_set_win_rate',
        'team_a_prev_avg_points_scored', 'team_a_prev_avg_points_conceded',
        'team_b_prev_matches_played', 'team_b_prev_matches_won', 'team_b_prev_win_rate',
        'team_b_prev_sets_won', 'team_b_prev_sets_lost', 'team_b_prev_set_win_rate',
        'team_b_prev_avg_points_scored', 'team_b_prev_avg_points_conceded',
        'h2h_matches', 'h2h_team_a_wins'
    ]
    
    X = pd.DataFrame([features])[feature_cols]
    
    # Predict
    prediction = model.predict(X)[0]
    probability = model.predict_proba(X)[0]
    
    winner = team_a_code if prediction == 1 else team_b_code
    confidence = max(probability)
    
    return winner, confidence

def main():
    print("=" * 80)
    print(" " * 20 + "PVL REINFORCED CONFERENCE 2025 - TOURNAMENT SIMULATION")
    print("=" * 80)
    
    # Load model
    model_data = joblib.load('volleyball_predictor.pkl')
    model = model_data['model']
    
    # Get current standings
    print("\n[1] CURRENT STANDINGS (from database)")
    print("-" * 80)
    standings = get_current_standings()
    for code in sorted(standings.keys()):
        team = standings[code]
        print(f"{code}: {team['wins']}-{team['losses']} ({team['games_played']} games)")
    
    # Identify missing matches
    print("\n[2] MISSING FIRST ROUND MATCHES")
    print("-" * 80)
    missing_matches = identify_missing_matches()
    
    if not missing_matches:
        print("✓ All first round matches completed!")
    else:
        print(f"Found {len(missing_matches)} matches remaining:\n")
        for match in missing_matches:
            print(f"  {match['pool']}: {match['team_a']} vs {match['team_b']}")
    
    # Predict missing matches
    print("\n[3] PREDICTING REMAINING MATCHES")
    print("-" * 80)
    
    predictions = []
    for match in missing_matches:
        winner, confidence = predict_match(model, match['team_a'], match['team_b'])
        loser = match['team_b'] if winner == match['team_a'] else match['team_a']
        
        predictions.append({
            'team_a': match['team_a'],
            'team_b': match['team_b'],
            'winner': winner,
            'confidence': confidence
        })
        
        print(f"  {match['team_a']} vs {match['team_b']}: "
              f"Predicted Winner = {winner} ({confidence:.1%} confidence)")
        
        # Update standings
        if winner in standings:
            standings[winner]['wins'] += 1
            standings[winner]['games_played'] += 1
        if loser in standings:
            standings[loser]['losses'] += 1
            standings[loser]['games_played'] += 1
    
    # Show projected standings after first round
    print("\n[4] PROJECTED STANDINGS AFTER FIRST ROUND")
    print("-" * 80)
    
    # Calculate win percentage and sort
    for code, team in standings.items():
        team['win_pct'] = team['wins'] / team['games_played'] if team['games_played'] > 0 else 0
    
    sorted_standings = sorted(standings.items(), 
                            key=lambda x: x[1]['win_pct'], 
                            reverse=True)
    
    print(f"{'Rank':<6} {'Team':<6} {'Record':<10} {'Win %':<10} {'Games'}")
    print("-" * 80)
    for i, (code, team) in enumerate(sorted_standings, 1):
        medal = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else "  "
        print(f"{medal} {i:<3} {code:<6} {team['wins']}-{team['losses']:<8} "
              f"{team['win_pct']:.1%}      {team['games_played']}")
    
    # Top 8 advance to quarterfinals
    print("\n[5] QUARTERFINAL MATCHUPS (Top 8 advance)")
    print("-" * 80)
    top_8 = sorted_standings[:8]
    
    matchups = [
        (top_8[0][0], top_8[7][0], "QF1"),
        (top_8[1][0], top_8[6][0], "QF2"),
        (top_8[2][0], top_8[5][0], "QF3"),
        (top_8[3][0], top_8[4][0], "QF4"),
    ]
    
    qf_winners = []
    for team_a, team_b, qf_name in matchups:
        winner, confidence = predict_match(model, team_a, team_b)
        qf_winners.append(winner)
        print(f"{qf_name}: #{matchups.index((team_a, team_b, qf_name))+1} {team_a} vs "
              f"#{8-matchups.index((team_a, team_b, qf_name))} {team_b} → "
              f"Winner: {winner} ({confidence:.1%})")
    
    # Semifinals
    print("\n[6] SEMIFINAL MATCHUPS")
    print("-" * 80)
    sf_matchups = [
        (qf_winners[0], qf_winners[3], "SF1"),  # QF1 winner vs QF4 winner
        (qf_winners[1], qf_winners[2], "SF2"),  # QF2 winner vs QF3 winner
    ]
    
    sf_winners = []
    for team_a, team_b, sf_name in sf_matchups:
        winner, confidence = predict_match(model, team_a, team_b)
        sf_winners.append(winner)
        print(f"{sf_name}: {team_a} vs {team_b} → Winner: {winner} ({confidence:.1%})")
    
    # Finals
    print("\n[7] CHAMPIONSHIP MATCH")
    print("-" * 80)
    champion, confidence = predict_match(model, sf_winners[0], sf_winners[1])
    runner_up = sf_winners[1] if champion == sf_winners[0] else sf_winners[0]
    
    print(f"🏆 {sf_winners[0]} vs {sf_winners[1]}")
    print(f"\n🥇 PREDICTED CHAMPION: {champion} ({confidence:.1%} confidence)")
    print(f"🥈 PREDICTED RUNNER-UP: {runner_up}")
    
    print("\n" + "=" * 80)
    print("SIMULATION COMPLETE!")
    print("=" * 80)

if __name__ == "__main__":
    main()
