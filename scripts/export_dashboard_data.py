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

def get_players_data(conn, tournament_id=None):
    """Fetch all players and their stats, optionally filtered by tournament."""
    cursor = conn.cursor()
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
                "total_points": 0,
                "attack_points": 0,
                "block_points": 0,
                "serve_points": 0,
                "dig_excellent": 0,
                "reception_excellent": 0,
                "reception_total_attempts": 0,
                "set_excellent": 0,
                "team_total_sets": 0
            },
            "teams": set(),
            "team_ids": set()
        }

    query = """
        SELECT pms.player_id, pms.team_id, 
               SUM(pms.sets_played), SUM(pms.attack_points), SUM(pms.block_points), SUM(pms.serve_points),
               SUM(pms.dig_excellent), SUM(pms.reception_excellent), SUM(pms.set_excellent),
               SUM(pms.reception_excellent + pms.reception_faults + pms.reception_continues) as reception_total
        FROM player_match_stats pms
        JOIN matches m ON pms.match_id = m.id
        WHERE pms.player_id IS NOT NULL
    """
    params = []
    
    if tournament_id:
        query += " AND m.tournament_id = ?"
        params.append(tournament_id)
        
    query += " GROUP BY pms.player_id, pms.team_id"

    cursor.execute(query, params)
    
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
            p["stats"]["total_points"] += (row[3] + row[4] + row[5])
            p["stats"]["dig_excellent"] += row[6]
            p["stats"]["reception_excellent"] += row[7]
            p["stats"]["set_excellent"] += row[8]
            p["stats"]["reception_total_attempts"] += (row[9] or 0)
            if tid in team_id_to_code:
                p["teams"].add(team_id_to_code[tid])
                p["team_ids"].add(tid)

    # Calculate team total sets played in the tournament
    team_total_sets_map = {}
    team_sets_query = """
        SELECT t.id, 
               SUM(m.team_a_sets_won + m.team_b_sets_won) as total_sets
        FROM teams t
        JOIN matches m ON (t.id = m.team_a_id OR t.id = m.team_b_id)
        WHERE m.winner_id IS NOT NULL
    """
    team_sets_params = []
    
    if tournament_id:
        team_sets_query += " AND m.tournament_id = ?"
        team_sets_params.append(tournament_id)
    
    team_sets_query += " GROUP BY t.id"
    
    cursor.execute(team_sets_query, team_sets_params)
    for row in cursor.fetchall():
        team_id = row[0]
        total_sets = row[1] or 0
        team_total_sets_map[team_id] = total_sets

    # Assign team total sets to each player
    for p in players_map.values():
        if p["team_ids"]:
            # Calculate average team total sets if player played for multiple teams
            team_sets_values = [team_total_sets_map.get(tid, 0) for tid in p["team_ids"]]
            p["stats"]["team_total_sets"] = int(sum(team_sets_values) / len(team_sets_values)) if team_sets_values else 0

    players_list = []
    for p in players_map.values():
        # Only include players who have played in this tournament (have stats > 0 or team association)
        if p["stats"]["sets_played"] > 0:
            p["teams"] = list(p["teams"])
            # Remove team_ids as it's only needed for internal calculation
            p.pop("team_ids", None)
            players_list.append(p)
        
    return players_list

def predict_match_wrapper(model, feature_names, conn, elo_map, team_a, team_b):
    """Wrapper for prediction."""
    X = _build_features_for_pair(conn, elo_map, team_a, team_b, feature_names)
    proba = model.predict_proba(X)[0]
    pred = int(proba[1] >= 0.5)
    winner = team_a if pred == 1 else team_b
    confidence = max(proba[0], proba[1])
    return winner, confidence

def get_tournament_history(conn, tournament_id):
    """Get tournament match history organized by phase."""
    cursor = conn.cursor()
    cursor.execute("""
        SELECT m.match_no, m.date, m.phase_no, m.phase_description,
               ta.code as team_a, tb.code as team_b,
               tw.code as winner, m.team_a_sets_won, m.team_b_sets_won
        FROM matches m
        JOIN teams ta ON m.team_a_id = ta.id
        JOIN teams tb ON m.team_b_id = tb.id
        LEFT JOIN teams tw ON m.winner_id = tw.id
        WHERE m.tournament_id = ?
        ORDER BY m.date, m.match_no
    """, (tournament_id,))
    
    matches = []
    champion = None
    phases_data = {}
    
    for row in cursor.fetchall():
        match_data = {
            "match_no": row[0],
            "date": row[1],
            "phase_no": row[2],
            "phase_description": row[3] or "Unknown",
            "team_a": _normalize_code(row[4]),
            "team_b": _normalize_code(row[5]),
            "winner": _normalize_code(row[6]) if row[6] else None,
            "team_a_sets": row[7],
            "team_b_sets": row[8]
        }
        matches.append(match_data)
        
        # Identify champion from championship phase
        if row[3] and "championship" in row[3].lower() and row[6]:
            champion = _normalize_code(row[6])
        
        # Group by phase
        phase_key = row[3] or "Unknown"
        if phase_key not in phases_data:
            phases_data[phase_key] = {
                "phase_description": phase_key,
                "phase_no": row[2],
                "matches": []
            }
        phases_data[phase_key]["matches"].append(match_data)
    
    # Convert to list and sort by phase_no
    phases_list = sorted(phases_data.values(), key=lambda x: x["phase_no"] if x["phase_no"] else 999)
    
    return {
        "champion": champion,
        "all_matches": matches,
        "phases": phases_list
    }
def _compute_final_preliminary_standings(conn, tournament_id):
    """Calculate final standings after all preliminary rounds (Group C & D complete)."""
    # Use FIVB ranking system - simplified version using just wins for now
    cursor = conn.cursor()
    cursor.execute("""
        SELECT t.code,
               COUNT(*) as games_played,
               SUM(CASE WHEN m.winner_id = t.id THEN 1 ELSE 0 END) as wins,
               SUM(CASE WHEN m.winner_id != t.id THEN 1 ELSE 0 END) as losses
        FROM teams t
        JOIN matches m ON (t.id = m.team_a_id OR t.id = m.team_b_id)
        WHERE m.tournament_id = ?
        AND (m.phase_description LIKE 'Preliminaries%' 
             OR m.phase_description LIKE 'Preliminary Round%')
        GROUP BY t.code
        ORDER BY wins DESC, losses ASC
    """, (tournament_id,))
    
    standings = []
    for idx, row in enumerate(cursor.fetchall(), 1):
        standings.append({
            "seed": idx,
            "team": _normalize_code(row[0]),
            "games": row[1],
            "wins": row[2],
            "losses": row[3]
        })
    
    return standings[:8]  # Top 8 advance to playoffs

def _get_playoff_matches(conn, tournament_id, phase_name):
    """Retrieve playoff matches from database for a specific phase."""
    cursor = conn.cursor()
    cursor.execute("""
        SELECT ta.code, tb.code, tw.code
        FROM matches m
        JOIN teams ta ON m.team_a_id = ta.id
        JOIN teams tb ON m.team_b_id = tb.id
        LEFT JOIN teams tw ON m.winner_id = tw.id
        WHERE m.tournament_id = ?
        AND m.phase_description LIKE ?
        ORDER BY m.match_no
    """, (tournament_id, f"%{phase_name}%"))
    
    matches = []
    for row in cursor.fetchall():
        matches.append({
            "team_a": _normalize_code(row[0]),
            "team_b": _normalize_code(row[1]),
            "winner": _normalize_code(row[2]) if row[2] else None,
            "is_actual": True
        })
    
    return matches if matches else None

def _simulate_playoff_bracket(conn, model, feature_names, elo_map, tournament_id):
    """Simulate playoff bracket dynamically using DB data when available."""
    
    # Get top 8 teams from preliminary rounds
    top_8 = _compute_final_preliminary_standings(conn, tournament_id)
    
    if len(top_8) < 8:
        return None  # Not enough teams yet
    
    # Quarterfinals
    qf_actual = _get_playoff_matches(conn, tournament_id, "Quarterfinal")
    
    if qf_actual:
        qf_matches = qf_actual
    else:
        # Predict quarterfinals based on seeding
        qf_matches = []
        matchups = [
            ("QF1", top_8[0]["team"], top_8[7]["team"]),  # #1 vs #8
            ("QF2", top_8[1]["team"], top_8[6]["team"]),  # #2 vs #7
            ("QF3", top_8[2]["team"], top_8[5]["team"]),  # #3 vs #6
            ("QF4", top_8[3]["team"], top_8[4]["team"]),  # #4 vs #5
        ]
        
        for label, team_a, team_b in matchups:
            winner, conf = predict_match_wrapper(model, feature_names, conn, elo_map, team_a, team_b)
            qf_matches.append({
                "matchup": label,
                "team_a": team_a,
                "team_b": team_b,
                "winner": winner,
                "confidence": conf,
                "is_actual": False
            })
    
    # Extract QF winners
    qf_winners = [m["winner"] for m in qf_matches if m["winner"]]
    
    if len(qf_winners) < 4:
        # Can't proceed to semifinals yet
        return {"quarterfinals": qf_matches, "semifinals": [], "championship": None, "third_place": None}
    
    # Semifinals
    sf_actual = _get_playoff_matches(conn, tournament_id, "Semifinal")
    
    if sf_actual:
        sf_matches = sf_actual
    else:
        # QF1 winner vs QF4 winner, QF2 winner vs QF3 winner
        sf_matches = []
        matchups = [
            ("SF1", qf_winners[0], qf_winners[3]),  # QF1 vs QF4
            ("SF2", qf_winners[1], qf_winners[2]),  # QF2 vs QF3
        ]
        
        for label, team_a, team_b in matchups:
            winner, conf = predict_match_wrapper(model, feature_names, conn, elo_map, team_a, team_b)
            sf_matches.append({
                "matchup": label,
                "team_a": team_a,
                "team_b": team_b,
                "winner": winner,
                "confidence": conf,
                "is_actual": False
            })
    
    # Extract SF winners and losers
    sf_winners = [m["winner"] for m in sf_matches if m["winner"]]
    sf_losers = [m["team_a"] if m["winner"] == m["team_b"] else m["team_b"] 
                 for m in sf_matches if m["winner"]]
    
    # Championship
    championship = None
    third_place = None
    
    if len(sf_winners) >= 2:
        champ_actual = _get_playoff_matches(conn, tournament_id, "Championship")
        
        if champ_actual and len(champ_actual) > 0:
            championship = champ_actual[0]
        else:
            winner, conf = predict_match_wrapper(model, feature_names, conn, elo_map, sf_winners[0], sf_winners[1])
            championship = {
                "matchup": "Championship",
                "team_a": sf_winners[0],
                "team_b": sf_winners[1],
                "winner": winner,
                "confidence": conf,
                "is_actual": False
            }
    
    if len(sf_losers) >= 2:
        third_actual = _get_playoff_matches(conn, tournament_id, "3rd")
        
        if third_actual and len(third_actual) > 0:
            third_place = third_actual[0]
        else:
            winner, conf = predict_match_wrapper(model, feature_names, conn, elo_map, sf_losers[0], sf_losers[1])
            third_place = {
                "matchup": "3rd Place",
                "team_a": sf_losers[0],
                "team_b": sf_losers[1],
                "winner": winner,
                "confidence": conf,
                "is_actual": False
            }
    
    return {
        "quarterfinals": qf_matches,
        "semifinals": sf_matches,
        "championship": championship,
        "third_place": third_place,
        "top_8_seeds": top_8
    }


def run_simulation_for_tournament(conn, tournament_id, tournament_code):
    """Run simulation for a specific tournament using actual phase data from database."""
    
    # For TEST_PVLR25, try to load the latest simulation file first
    if tournament_code == "TEST_PVLR25":
        outputs_dir = Path(__file__).parent.parent / "outputs"
        if outputs_dir.exists():
            sim_files = sorted(outputs_dir.glob("tournament_simulation_*.json"), reverse=True)
            if sim_files:
                latest_sim_file = sim_files[0]
                try:
                    with open(latest_sim_file) as f:
                        sim_data = json.load(f)
                    
                    # Extract playoff bracket from simulation file
                    playoffs_from_sim = None
                    if "quarterfinals" in sim_data and "semifinals" in sim_data and "championship" in sim_data:
                        # Build playoffs structure
                        playoffs_from_sim = {
                            "quarterfinals": [
                                {
                                    "matchup": qf["match"],
                                    "team_a": qf["team_a"],
                                    "team_b": qf["team_b"],
                                    "winner": qf["winner"],
                                    "confidence": qf["confidence"],
                                    "is_actual": False
                                }
                                for qf in sim_data["quarterfinals"]
                            ],
                            "semifinals": [
                                {
                                    "matchup": sf["match"],
                                    "team_a": sf["team_a"],
                                    "team_b": sf["team_b"],
                                    "winner": sf["winner"],
                                    "confidence": sf["confidence"],
                                    "is_actual": False
                                }
                                for sf in sim_data["semifinals"]
                            ],
                            "championship": {
                                "matchup": "Championship",
                                "team_a": sim_data["championship"]["team_a"],
                                "team_b": sim_data["championship"]["team_b"],
                                "winner": sim_data["championship"]["champion"],
                                "confidence": sim_data["championship"]["confidence"],
                                "is_actual": False
                            },
                            "third_place": None,
                            "top_8_seeds": sim_data.get("second_round_combined_rankings", [])[:8]
                        }
                        
                        # Add third place if available
                        if "third_place" in sim_data and sim_data["third_place"]:
                            playoffs_from_sim["third_place"] = {
                                "matchup": "3rd Place",
                                "team_a": sim_data["third_place"]["team_a"],
                                "team_b": sim_data["third_place"]["team_b"],
                                "winner": sim_data["third_place"]["winner"],
                                "confidence": sim_data["third_place"]["confidence"],
                                "is_actual": False
                            }
                except Exception as e:
                    print(f"Warning: Could not load simulation file {latest_sim_file}: {e}")
                    playoffs_from_sim = None
    
    cal = Path(MODELS_DIR) / "calibrated_xgboost_with_players.pkl"
    model_path = cal if cal.exists() else Path(BEST_MODEL_STR)
    model_art = joblib.load(model_path)
    model = model_art["model"]
    feature_names = model_art.get("feature_names")
    elo_map = _compute_current_elo(conn)
    
    cursor = conn.cursor()
    
    # Get current standings
    query = """
        SELECT t.code, t.name,
               SUM(CASE WHEN m.winner_id = t.id THEN 1 ELSE 0 END) as wins,
               SUM(CASE WHEN m.winner_id != t.id THEN 1 ELSE 0 END) as losses,
               COUNT(*) as games_played
        FROM teams t
        JOIN matches m ON (t.id = m.team_a_id OR t.id = m.team_b_id)
        WHERE m.tournament_id = ?
        GROUP BY t.code, t.name
        ORDER BY t.code
    """
    cursor.execute(query, (tournament_id,))
    
    standings = {}
    for row in cursor.fetchall():
        code = _normalize_code(row[0])
        entry = standings.setdefault(code, {"name": row[1], "wins": 0, "losses": 0, "games_played": 0})
        entry["wins"] += row[2]
        entry["losses"] += row[3]
        entry["games_played"] += row[4]

    current_standings = [
        {"team": code, "name": rec["name"], "wins": rec["wins"], "losses": rec["losses"], "games_played": rec["games_played"]}
        for code, rec in standings.items()
    ]

    # Only do full simulation for TEST_PVLR25
    if tournament_code != "TEST_PVLR25":
        return {
            "current_standings": current_standings,
            "predictions_by_phase": [],
            "final_standings": current_standings
        }

    # Get actual Group C and D team compositions from database (from completed matches)
    # This is better than trying to recalculate because the database has the truth
    cursor.execute("""
        SELECT DISTINCT t.code, m.phase_description
        FROM matches m
        JOIN teams ta ON m.team_a_id = ta.id
        JOIN teams tb ON m.team_b_id = tb.id
        JOIN teams t ON (t.id = ta.id OR t.id = tb.id)
        WHERE m.tournament_id = ?
        AND (m.phase_description = 'Preliminary Round 2 (Group C)' 
             OR m.phase_description = 'Preliminary Round 2 (Group D)')
    """, (tournament_id,))
    
    group_c_teams = set()
    group_d_teams = set()
    for row in cursor.fetchall():
        team = _normalize_code(row[0])
        phase = row[1]
        if 'Group C' in phase:
            group_c_teams.add(team)
        elif 'Group D' in phase:
            group_d_teams.add(team)

    
    # Get all matches with their phases
    cursor.execute("""
        SELECT m.id, m.phase_description, 
               ta.code as team_a, tb.code as team_b,
               m.winner_id, m.team_a_sets_won, m.team_b_sets_won
        FROM matches m
        JOIN teams ta ON m.team_a_id = ta.id
        JOIN teams tb ON m.team_b_id = tb.id
        WHERE m.tournament_id = ?
        ORDER BY m.phase_no, m.date, m.match_no
    """, (tournament_id,))
    
    all_matches = cursor.fetchall()
    
    # Organize matches by phase
    matches_by_phase = {}
    completed_matches = set()
    
    for match in all_matches:
        match_id, phase_desc, team_a, team_b, winner_id, sets_a, sets_b = match
        phase_desc = phase_desc or "Unknown"
        
        if phase_desc not in matches_by_phase:
            matches_by_phase[phase_desc] = {
                "completed": [],
                "pending": []
            }
        
        team_a_norm = _normalize_code(team_a)
        team_b_norm = _normalize_code(team_b)
        
        match_data = {
            "match_id": match_id,
            "team_a": team_a_norm,
            "team_b": team_b_norm,
            "phase": phase_desc
        }
        
        if winner_id is not None:
            # Completed match
            matches_by_phase[phase_desc]["completed"].append({
                **match_data,
                "winner": team_a_norm if sets_a > sets_b else team_b_norm,
                "team_a_sets": sets_a,
                "team_b_sets": sets_b
            })
            completed_matches.add(tuple(sorted([team_a_norm, team_b_norm])))
        else:
            # Pending match
            matches_by_phase[phase_desc]["pending"].append(match_data)
    
    
    # Only check for missing matches in groups that have some matches already
    # Group C
    if group_c_teams:
        group_c_phase = "Preliminary Round 2 (Group C)"
        if group_c_phase not in matches_by_phase:
            matches_by_phase[group_c_phase] = {"completed": [], "pending": []}
        
        # All possible matches within Group C (should be 9 for 6 teams playing cross-pool)
        # Since we don't know which 3 are from Pool A and which from Pool B, we won't add missing matches
        # We'll only predict if there are already some matches in the database
        pass  # Group C complete based on database
    
    # Group D  
    if group_d_teams:
        group_d_phase = "Preliminary Round 2 (Group D)"
        if group_d_phase not in matches_by_phase:
            matches_by_phase[group_d_phase] = {"completed": [], "pending": []}
        
        # For Group D, we know the composition: GTH, PGA, CTC from Pool B and HSH, FFF, CAP from Pool A
        # Check if all 9 cross-pool matches exist
        pool_b_in_d = {"GTH", "PGA", "CTC"}
        pool_a_in_d = {"HSH", "FFF", "CAP"}
        
        required_group_d = []
        for pool_b_team in pool_b_in_d:
            if pool_b_team in group_d_teams:  # Verify team is actually in Group D
                for pool_a_team in pool_a_in_d:
                    if pool_a_team in group_d_teams:
                        pair = tuple(sorted([pool_b_team, pool_a_team]))
                        required_group_d.append(pair)
        
        for pair in required_group_d:
            if pair not in completed_matches:
                matches_by_phase[group_d_phase]["pending"].append({
                    "match_id": None,
                    "team_a": pair[0],
                    "team_b": pair[1],
                    "phase": group_d_phase
                })
    
    # Generate predictions for pending matches by phase
    predictions_by_phase = []
    
    for phase_desc in sorted(matches_by_phase.keys()):
        phase_data = matches_by_phase[phase_desc]
        phase_predictions = {
            "phase_description": phase_desc,
            "predictions": []
        }
        
        for match in phase_data["pending"]:
            winner, confidence = predict_match_wrapper(
                model, feature_names, conn, elo_map,
                match["team_a"], match["team_b"]
            )
            
            phase_predictions["predictions"].append({
                "team_a": match["team_a"],
                "team_b": match["team_b"],
                "winner": winner,
                "confidence": confidence
            })
        
        predictions_by_phase.append(phase_predictions)
    
    # Calculate final standings with FIVB ranking (Match Points, Set Ratio, Point Ratio)
    # We need to rebuild the 'combined' dictionary from all matches (completed + pending/predicted if we wanted, but here just completed)
    # Actually, for final standings of preliminaries, we just want the top 8 based on completed matches
    
    # Re-fetch all match rows for ranking
    cursor.execute("""
        SELECT m.match_no, m.date, 
               ta.code, tb.code, 
               m.team_a_sets_won, m.team_b_sets_won
        FROM matches m
        JOIN teams ta ON m.team_a_id = ta.id
        JOIN teams tb ON m.team_b_id = tb.id
        WHERE m.tournament_id = ?
        AND (m.phase_description LIKE 'Preliminaries%' OR m.phase_description LIKE 'Preliminary Round%')
        AND m.winner_id IS NOT NULL
    """, (tournament_id,))
    
    rows = cursor.fetchall()
    combined = {}
    
    for row in rows:
        # row: match_no, date, team_a, team_b, sets_a, sets_b
        t_a, t_b = row[2], row[3]
        s_a, s_b = row[4], row[5]
        
        # Initialize if needed
        for t in [t_a, t_b]:
            if t not in combined:
                combined[t] = {
                    "wins": 0, "losses": 0, "games_played": 0,
                    "sets_won": 0, "sets_lost": 0,
                    "points_won": 0, "points_lost": 0,
                    "match_points": 0
                }
        
        # Update stats
        combined[t_a]["games_played"] += 1
        combined[t_b]["games_played"] += 1
        combined[t_a]["sets_won"] += s_a
        combined[t_a]["sets_lost"] += s_b
        combined[t_b]["sets_won"] += s_b
        combined[t_b]["sets_lost"] += s_a
        
        # Match points
        if s_a == 3:
            if s_b == 0 or s_b == 1:
                combined[t_a]["match_points"] += 3
            elif s_b == 2:
                combined[t_a]["match_points"] += 2
                combined[t_b]["match_points"] += 1
        elif s_b == 3:
            if s_a == 0 or s_a == 1:
                combined[t_b]["match_points"] += 3
            elif s_a == 2:
                combined[t_b]["match_points"] += 2
                combined[t_a]["match_points"] += 1
        
        if s_a > s_b:
            combined[t_a]["wins"] += 1
            combined[t_b]["losses"] += 1
        else:
            combined[t_b]["wins"] += 1
            combined[t_a]["losses"] += 1

    _finalize_ratios(combined)
    ordered_codes = _rank_fivb(combined, match_rows=rows)
    
    final_standings = []
    for i, code in enumerate(ordered_codes, 1):
        rec = combined[code]
        final_standings.append({
            "rank": i,
            "team": code,
            "wins": rec["wins"],
            "losses": rec["losses"],
            "match_points": rec["match_points"],
            "set_ratio": rec["set_ratio"],
            "point_ratio": rec["point_ratio"],
            "games_played": rec["games_played"]
        })

    # Simulate Playoffs
    # For TEST_PVLR25, use playoffs from simulation file if available
    if tournament_code == "TEST_PVLR25" and 'playoffs_from_sim' in locals() and playoffs_from_sim is not None:
        playoffs = playoffs_from_sim
    else:
        playoffs = _simulate_playoff_bracket(conn, model, feature_names, elo_map, tournament_id)

    return {
        "current_standings": current_standings,
        "predictions_by_phase": predictions_by_phase,
        "final_standings": final_standings,
        "matches_by_phase": {
            phase: {
                "completed_count": len(data["completed"]),
                "pending_count": len(data["pending"])
            }
            for phase, data in matches_by_phase.items()
        },
        "playoffs": playoffs
    }


def main():
    print("Exporting dashboard data with multi-tournament support...")
    conn = sqlite3.connect(DB_FILE_STR)
    
    cursor = conn.cursor()
    cursor.execute("SELECT id, code, name FROM tournaments ORDER BY id DESC")
    tournaments_data = []
    
    for tournament_row in cursor.fetchall():
        tid, tcode, tname = tournament_row
        print(f"Processing tournament: {tname} ({tcode})")
        
        simulation = run_simulation_for_tournament(conn, tid, tcode)
        history = get_tournament_history(conn, tid)
        
        tournaments_data.append({
            "id": tid,
            "code": tcode,
            "name": tname,
            "simulation": simulation,
            "history": history
        })
    
    # Find ID for TEST_PVLR25 to filter player stats
    cursor.execute("SELECT id FROM tournaments WHERE code = 'TEST_PVLR25'")
    row = cursor.fetchone()
    current_tournament_id = row[0] if row else None

    data = {
        "teams": get_teams_data(conn),
        "players": get_players_data(conn, tournament_id=current_tournament_id),
        "tournaments": tournaments_data,
        "last_updated": datetime.now().isoformat()
    }
    
    conn.close()
    
    OUTPUT_JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
    
    with open(OUTPUT_JSON_PATH, "w") as f:
        json.dump(data, f, indent=2)
    
    print(f"\\nData exported successfully to {OUTPUT_JSON_PATH}")
    print(f"Total tournaments: {len(tournaments_data)}")

if __name__ == "__main__":
    main()
