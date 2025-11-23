import json
import sqlite3
import pandas as pd
import joblib
from pathlib import Path
from datetime import datetime

# Import existing modules
try:
    from scripts.config import DB_FILE_STR, BEST_MODEL_STR, MODELS_DIR, canonicalize_team_code
    from scripts.simulate_tournament import (
        _compute_current_elo, _build_features_for_pair, _approx_simulated_sets,
        _rank_fivb, _apply_match, _finalize_ratios, _fetch_tournament_match_rows,
        _played_matchups_in_tournament, _normalize_code, ELO_DEFAULT
    )
except ImportError:
    # Fallback if running from scripts/ directory directly
    import sys
    sys.path.append(str(Path(__file__).parent.parent))
    from scripts.config import DB_FILE_STR, BEST_MODEL_STR, MODELS_DIR, canonicalize_team_code
    from scripts.simulate_tournament import (
        _compute_current_elo, _build_features_for_pair, _approx_simulated_sets,
        _rank_fivb, _apply_match, _finalize_ratios, _fetch_tournament_match_rows,
        _played_matchups_in_tournament, _normalize_code, ELO_DEFAULT
    )

OUTPUT_JSON_PATH = Path(__file__).parent.parent / "dashboard" / "public" / "data.json"

def get_teams_data(conn):
    """Fetch all teams and their basic info."""
    cursor = conn.cursor()
    cursor.execute("SELECT id, code, name, coach, assistant_coach FROM teams")
    teams = []
    for row in cursor.fetchall():
        teams.append({
            "id": row[0],
            "code": _normalize_code(row[1]),
            "name": row[2],
            "coach": row[3],
            "assistant_coach": row[4]
        })
    return teams

def get_players_data(conn):
    """Fetch all players and their stats."""
    cursor = conn.cursor()
    # Get basic player info
    cursor.execute("SELECT id, first_name, last_name, full_name FROM players")
    players_map = {}
    for row in cursor.fetchall():
        players_map[row[0]] = {
            "id": row[0],
            "first_name": row[1],
            "last_name": row[2],
            "full_name": row[3],
            "stats": {
                "sets_played": 0,
                "attack_points": 0,
                "block_points": 0,
                "serve_points": 0,
                "dig_excellent": 0,
                "reception_excellent": 0,
                "set_excellent": 0
            },
            "teams": set()
        }

    # Aggregate stats
    cursor.execute("""
        SELECT player_id, team_id, 
               SUM(sets_played), SUM(attack_points), SUM(block_points), SUM(serve_points),
               SUM(dig_excellent), SUM(reception_excellent), SUM(set_excellent)
        FROM player_match_stats
        GROUP BY player_id, team_id
    """)
    
    # Helper to get team code from ID
    cursor2 = conn.cursor()
    cursor2.execute("SELECT id, code FROM teams")
    team_id_to_code = {r[0]: _normalize_code(r[1]) for r in cursor2.fetchall()}

    for row in cursor.fetchall():
        pid = row[0]
        tid = row[1]
        if pid in players_map:
            p = players_map[pid]
            p["stats"]["sets_played"] += row[2]
            p["stats"]["attack_points"] += row[3]
            p["stats"]["block_points"] += row[4]
            p["stats"]["serve_points"] += row[5]
            p["stats"]["dig_excellent"] += row[6]
            p["stats"]["reception_excellent"] += row[7]
            p["stats"]["set_excellent"] += row[8]
            if tid in team_id_to_code:
                p["teams"].add(team_id_to_code[tid])

    # Convert sets to lists for JSON serialization
    players_list = []
    for p in players_map.values():
        p["teams"] = list(p["teams"])
        players_list.append(p)
        
    return players_list

def predict_match_wrapper(model, feature_names, conn, elo_map, team_a, team_b):
    """Wrapper for prediction that returns dict."""
    X = _build_features_for_pair(conn, elo_map, team_a, team_b, feature_names)
    proba = model.predict_proba(X)[0]
    pred = int(proba[1] >= 0.5)
    winner = team_a if pred == 1 else team_b
    confidence = max(proba[0], proba[1])
    return winner, confidence

def run_simulation_for_tournament(conn, tournament_id, tournament_code):\n    \"\"\"Run simulation for a specific tournament.\"\"\"\n    \n    # Load Model\n    cal = Path(MODELS_DIR) / 'calibrated_xgboost_with_players.pkl'\n    model_path = cal if cal.exists() else Path(BEST_MODEL_STR)\n    model_art = joblib.load(model_path)\n    model = model_art['model']\n    feature_names = model_art.get('feature_names')\n\n    elo_map = _compute_current_elo(conn)\n    \n    # 1. Current Standings (from actual database matches)\n    cursor = conn.cursor()\n    cursor.execute('''\n        SELECT t.code, t.name,\n               SUM(CASE WHEN m.winner_id = t.id THEN 1 ELSE 0 END) as wins,\n               SUM(CASE WHEN m.winner_id != t.id THEN 1 ELSE 0 END) as losses,\n               COUNT(*) as games_played\n        FROM teams t\n        JOIN matches m ON (t.id = m.team_a_id OR t.id = m.team_b_id)\n        WHERE m.tournament_id = ?\n        GROUP BY t.code, t.name\n        ORDER BY t.code\n    ''', (tournament_id,))\n    \n    standings = {}\n    for row in cursor.fetchall():\n        code = _normalize_code(row[0])\n        entry = standings.setdefault(code, {'name': row[1], 'wins': 0, 'losses': 0, 'games_played': 0})\n        entry['wins'] += row[2]\n        entry['losses'] += row[3]\n        entry['games_played'] += row[4]\n\n    current_standings = [\n        {'team': code, 'name': rec['name'], 'wins': rec['wins'], 'losses': rec['losses'], 'games_played': rec['games_played']}\n        for code, rec in standings.items()\n    ]\n\n    # Only run full simulation for TEST_PVLR25 (the one with defined pools)\n    if tournament_code != 'TEST_PVLR25':\n        return {\n            \"current_standings\": current_standings,\n            \"future_matches\": [],\n            \"final_standings\": current_standings,\n            \"pools\": {}\n        }\n\n    # 2. Missing Matches & Simulation for TEST_PVLR25\n    pool_a = ['HSH', 'FFF', 'CMF', 'CSS', 'CAP', 'NXL']\n    pool_b = ['ZUS', 'CCS', 'AKA', 'PGA', 'CTC', 'GTH']\n    played_matchups = _played_matchups_in_tournament(conn)\n    \n    # Find all missing matches (both round 1 and round 2)\n    all_future_matches = []\n    \n    # First round missing matches\n    for pool, pool_name in [(pool_a, 'Pool A'), (pool_b, 'Pool B')]:\n        for i, team_a in enumerate(pool):\n            for team_b in pool[i+1:]:\n                matchup = tuple(sorted([_normalize_code(team_a), _normalize_code(team_b)]))\n                if matchup not in played_matchups:\n                    all_future_matches.append({\n                        'team_a': _normalize_code(team_a),\n                        'team_b': _normalize_code(team_b),\n                        'pool': pool_name,\n                        'round': 'Round 1'\n                    })\n\n    # Second round setup: Determine Pool C and D\n    rows = _fetch_tournament_match_rows(conn)\n    a_rec, b_rec = {}, {}\n    pool_a_set, pool_b_set = set(pool_a), set(pool_b)\n    for r in rows:\n        aC, bC = r['a_code'], r['b_code']\n        if aC in pool_a_set and bC in pool_a_set:\n            _apply_match(a_rec, aC, bC, r['a_sets'], r['b_sets'], r['a_pts'], r['b_pts'])\n        if aC in pool_b_set and bC in pool_b_set:\n            _apply_match(b_rec, aC, bC, r['a_sets'], r['b_sets'], r['a_pts'], r['b_pts'])\n           \n    a_ranked = _rank_fivb(a_rec, pool_a, rows)\n    b_ranked = _rank_fivb(b_rec, pool_b, rows)\n    \n    a_top3, a_bot3 = a_ranked[:3], a_ranked[3:]\n    b_top3, b_bot3 = b_ranked[:3], b_ranked[3:]\n    pool_c = a_top3 + b_bot3\n    pool_d = b_top3 + a_bot3\n\n    # Second round missing matches\n    def find_cross_pool_matches(pool, origin_top3, origin_other_bot3, pool_name):\n        for a in origin_top3:\n            for b in origin_other_bot3:\n                pair = tuple(sorted([a, b]))\n                if pair not in played_matchups:\n                    all_future_matches.append({\n                        'team_a': a,\n                        'team_b': b,\n                        'pool': pool_name,\n                        'round': 'Round 2'\n                    })\n\n    find_cross_pool_matches(pool_c, a_top3, b_bot3, \"Pool C\")\n    find_cross_pool_matches(pool_c, b_bot3, a_top3, \"Pool C\")\n    find_cross_pool_matches(pool_d, b_top3, a_bot3, \"Pool D\")\n    find_cross_pool_matches(pool_d, a_bot3, b_top3, \"Pool D\")\n\n    # Predict all future matches\n    future_predictions = []\n    temp_standings = {k: v.copy() for k, v in standings.items()}\n    \n    combined = {}\n    for r in rows:\n        _apply_match(combined, r['a_code'], r['b_code'], r['a_sets'], r['b_sets'], r['a_pts'], r['b_pts'])\n\n    for match in all_future_matches:\n        winner, confidence = predict_match_wrapper(model, feature_names, conn, elo_map, match['team_a'], match['team_b'])\n        loser = match['team_b'] if winner == match['team_a'] else match['team_a']\n        \n        future_predictions.append({\n            'team_a': match['team_a'],\n            'team_b': match['team_b'],\n            'winner': winner,\n            'confidence': confidence,\n            'pool': match['pool'],\n            'round': match['round']\n        })\n        \n        # Update temp standings for final projection\n        if winner in temp_standings:\n            temp_standings[winner]['wins'] += 1\n            temp_standings[winner]['games_played'] += 1\n        if loser in temp_standings:\n            temp_standings[loser]['losses'] += 1\n            temp_standings[loser]['games_played'] += 1\n            \n        # Apply to combined for FIVB ranking\n        w_sets, l_sets, w_pts, l_pts = _approx_simulated_sets(confidence)\n        if winner == match['team_a']:\n            _apply_match(combined, match['team_a'], match['team_b'], w_sets, l_sets, w_pts, l_pts)\n        else:\n            _apply_match(combined, match['team_a'], match['team_b'], l_sets, w_sets, l_pts, w_pts)\n\n    # Final projected standings using FIVB ranking\n    _finalize_ratios(combined)\n    ordered_codes = _rank_fivb(combined, match_rows=rows)\n    \n    final_standings = []\n    for i, code in enumerate(ordered_codes, 1):\n        rec = combined[code]\n        final_standings.append({\n            'rank': i,\n            'team': code,\n            'wins': rec['wins'],\n            'losses': rec['losses'],\n            'match_points': rec['match_points'],\n            'set_ratio': rec['set_ratio'],\n            'point_ratio': rec['point_ratio'],\n            'games_played': rec['games_played']\n        })\n\n    return {\n        \"current_standings\": current_standings,\n        \"future_matches\": future_predictions,\n        \"final_standings\": final_standings,\n        \"pools\": {\n            \"pool_a\": pool_a,\n            \"pool_b\": pool_b,\n            \"pool_c\": pool_c,\n            \"pool_d\": pool_d\n        }\n    }

def main():
    print("Exporting dashboard data...")
    conn = sqlite3.connect(DB_FILE_STR)
    
    # Get all tournaments
    cursor = conn.cursor()
    cursor.execute("SELECT id, code, name FROM tournaments ORDER BY id DESC")
    tournaments_data = []
    
    for tournament_row in cursor.fetchall():
        tid, tcode, tname = tournament_row
        print(f"Processing tournament: {tname} ({tcode})")
        
        simulation = run_simulation_for_tournament(conn, tid, tcode)
        tournaments_data.append({
            "id": tid,
            "code": tcode,
            "name": tname,
            "simulation": simulation
        })
    
    data = {
        "teams": get_teams_data(conn),
        "players": get_players_data(conn),
        "tournaments": tournaments_data,
        "last_updated": datetime.now().isoformat()
    }
    
    conn.close()
    
    # Ensure output directory exists
    OUTPUT_JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
    
    with open(OUTPUT_JSON_PATH, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"Data exported to {OUTPUT_JSON_PATH}")

if __name__ == "__main__":
    main()

