"""
Full tournament simulation with player-aware features and ELO.

This module loads a calibrated model (default) and predicts outcomes for the
remaining schedule and bracket rounds based on current standings in the DB.

Updates:
- Uses configured database path from scripts.config (no hardcoded filenames)
- Accepts optional model_path argument (passed via run_simulation.py)
- Builds inference features compatible with player-aware training columns
  (ELO + team historical + aggregated player metrics)
"""

import sqlite3
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timezone
import json

# Support import from project root (scripts as a package) and direct execution
try:  # when imported as scripts.simulate_tournament
    from .config import DB_FILE_STR, BEST_MODEL_STR, MODELS_DIR
except Exception:  # when run directly from scripts/
    from config import DB_FILE_STR, BEST_MODEL_STR, MODELS_DIR

# Normalize legacy / variant team codes (e.g., treat CHD as CSS going forward)
try:
    from .config import canonicalize_team_code as _canonical
except Exception:
    from config import canonicalize_team_code as _canonical

def _normalize_code(code: str) -> str:
    return _canonical(code)


# ----------------------------- ELO Utilities -----------------------------
ELO_DEFAULT = 1500.0
ELO_K = 20.0

# ------------------------- Ranking/Scoring Utilities -------------------------

def _init_record() -> dict:
    return {
        'name': '',
        'wins': 0,
        'losses': 0,
        'games_played': 0,
        'match_points': 0,
        'sets_won': 0,
        'sets_lost': 0,
        'points_scored': 0,
        'points_allowed': 0,
        # derived
        'win_pct': 0.0,
        'set_ratio': 0.0,
        'point_ratio': 0.0,
    }

def _match_points(w_sets: int, l_sets: int) -> tuple[int, int]:
    # FIVB: 3-0/3-1 → 3/0; 3-2 → 2/1
    if w_sets == 3 and l_sets in (0, 1):
        return 3, 0
    return 2, 1

def _ensure_team(standings: dict, code: str):
    if code not in standings:
        standings[code] = _init_record()

def _apply_match(standings: dict, a_code: str, b_code: str, a_sets: int, b_sets: int, a_pts: int | float, b_pts: int | float):
    """Apply a single finished match to standings using FIVB scoring."""
    aN, bN = _normalize_code(a_code), _normalize_code(b_code)
    _ensure_team(standings, aN)
    _ensure_team(standings, bN)
    # winner/loser by sets
    if a_sets == b_sets:
        return  # malformed
    w_code, l_code = (aN, bN) if a_sets > b_sets else (bN, aN)
    w_sets, l_sets = (a_sets, b_sets) if a_sets > b_sets else (b_sets, a_sets)
    w_pts, l_pts = (a_pts, b_pts) if a_sets > b_sets else (b_pts, a_pts)

    # matches won / losses
    standings[w_code]['wins'] += 1
    standings[l_code]['losses'] += 1
    standings[w_code]['games_played'] += 1
    standings[l_code]['games_played'] += 1

    # match points
    mp_w, mp_l = _match_points(w_sets, l_sets)
    standings[w_code]['match_points'] += mp_w
    standings[l_code]['match_points'] += mp_l

    # sets and points
    standings[w_code]['sets_won'] += w_sets
    standings[w_code]['sets_lost'] += l_sets
    standings[l_code]['sets_won'] += l_sets
    standings[l_code]['sets_lost'] += w_sets
    standings[w_code]['points_scored'] += w_pts or 0
    standings[w_code]['points_allowed'] += l_pts or 0
    standings[l_code]['points_scored'] += l_pts or 0
    standings[l_code]['points_allowed'] += w_pts or 0

def _finalize_ratios(standings: dict):
    for rec in standings.values():
        gp = rec.get('games_played', 0) or 0
        rec['win_pct'] = (rec.get('wins', 0) / gp) if gp else 0.0
        sw, sl = rec.get('sets_won', 0), rec.get('sets_lost', 0)
        rec['set_ratio'] = (sw / sl) if sl else (sw if sw else 0.0)
        ps, pa = rec.get('points_scored', 0), rec.get('points_allowed', 0)
        rec['point_ratio'] = (ps / pa) if pa else (ps if ps else 0.0)

def _fetch_tournament_match_rows(conn: sqlite3.Connection):
    cur = conn.cursor()
    cur.execute(
        '''
        SELECT m.id,
               ta.code AS team_a_code,
               tb.code AS team_b_code,
               m.team_a_sets_won,
               m.team_b_sets_won,
               SUM(CASE WHEN tms.team_id = ta.id THEN tms.total_points ELSE 0 END) AS a_points,
               SUM(CASE WHEN tms.team_id = tb.id THEN tms.total_points ELSE 0 END) AS b_points
        FROM matches m
        JOIN teams ta ON m.team_a_id = ta.id
        JOIN teams tb ON m.team_b_id = tb.id
        LEFT JOIN team_match_stats tms ON tms.match_id = m.id
        WHERE m.tournament_id = (SELECT id FROM tournaments WHERE code = 'TEST_PVLR25')
          AND m.status IS NOT NULL
        GROUP BY m.id
        ORDER BY m.id
        '''
    )
    rows = []
    for r in cur.fetchall():
        _, a_code, b_code, a_sets, b_sets, a_pts, b_pts = r
        if a_sets is None or b_sets is None:
            continue
        rows.append({
            'a_code': _normalize_code(a_code),
            'b_code': _normalize_code(b_code),
            'a_sets': int(a_sets),
            'b_sets': int(b_sets),
            'a_pts': int(a_pts or 0),
            'b_pts': int(b_pts or 0),
        })
    return rows

def _rank_fivb(standings: dict, codes_subset: list[str] | None = None, match_rows: list[dict] | None = None) -> list[str]:
    """Rank teams using FIVB rules: wins > match points > set ratio > point ratio > head-to-head among tied.

    If match_rows is provided, resolve tied blocks using head-to-head wins among the tied teams.
    Each row in match_rows should have keys: a_code, b_code, a_sets, b_sets.
    """
    _finalize_ratios(standings)
    codes = list(codes_subset) if codes_subset else list(standings.keys())

    def sort_key(code):
        rec = standings.get(code, {})
        return (
            rec.get('wins', 0),
            rec.get('match_points', 0),
            rec.get('set_ratio', 0.0),
            rec.get('point_ratio', 0.0),
        )

    # initial sort
    ordered = sorted(codes, key=sort_key, reverse=True)

    # head-to-head adjustment within tied blocks
    if match_rows:
        i = 0
        while i < len(ordered):
            j = i + 1
            key_i = sort_key(ordered[i])
            while j < len(ordered) and sort_key(ordered[j]) == key_i:
                j += 1
            if j - i > 1:
                tied = ordered[i:j]
                tied_set = set(tied)
                # mini-table based on H2H wins within tied block
                h2h_wins = {c: 0 for c in tied}
                h2h_sets_won = {c: 0 for c in tied}
                h2h_sets_lost = {c: 0 for c in tied}
                for r in match_rows:
                    a, b = r['a_code'], r['b_code']
                    if a in tied_set and b in tied_set:
                        if r['a_sets'] == r['b_sets']:
                            continue
                        winner = a if r['a_sets'] > r['b_sets'] else b
                        h2h_wins[winner] += 1
                        # accumulate sets for set ratio tie-break
                        h2h_sets_won[a] += r['a_sets']
                        h2h_sets_lost[a] += r['b_sets']
                        h2h_sets_won[b] += r['b_sets']
                        h2h_sets_lost[b] += r['a_sets']
                # stable sort tied block by h2h wins (desc)
                def h2h_set_ratio(c: str) -> float:
                    sw = h2h_sets_won.get(c, 0)
                    sl = h2h_sets_lost.get(c, 0)
                    return (sw / sl) if sl else (float(sw) if sw else 0.0)
                tied_sorted = sorted(
                    tied,
                    key=lambda c: (h2h_wins.get(c, 0), h2h_set_ratio(c)),
                    reverse=True,
                )
                ordered[i:j] = tied_sorted
            i = j
    return ordered

def _approx_simulated_sets(confidence: float) -> tuple[int, int, int, int]:
    """Return (w_sets, l_sets, w_points, l_points) for a simulated match based on confidence."""
    if confidence < 0.55:
        # tight 3-2
        return 3, 2, 108, 106
    if confidence < 0.75:
        # moderate 3-1
        return 3, 1, 100, 85
    # decisive 3-0
    return 3, 0, 75, 60

def _compute_current_elo(conn: sqlite3.Connection) -> dict:
    """Compute current ELO ratings with alias consolidation (e.g., CHD counted as CSS)."""
    cur = conn.cursor()
    # Preload code->id to resolve canonical IDs quickly
    cur.execute("SELECT id, code FROM teams")
    code_to_id = {code: tid for tid, code in cur.fetchall()}

    def canonical_id_for_code(code: str) -> int | None:
        return code_to_id.get(_normalize_code(code))

    cur.execute('''
        SELECT m.id, ta.code, tb.code, w.code as w_code
        FROM matches m
        JOIN teams ta ON m.team_a_id = ta.id
        JOIN teams tb ON m.team_b_id = tb.id
        LEFT JOIN teams w ON m.winner_id = w.id
        WHERE m.status IS NOT NULL
        ORDER BY m.id
    ''')
    elo: dict[int, float] = {}

    def g(team_id: int) -> float:
        return elo.get(team_id, ELO_DEFAULT)
    def expected(ra: float, rb: float) -> float:
        return 1.0 / (1.0 + 10 ** (-(ra - rb) / 400.0))

    for mid, a_code, b_code, w_code in cur.fetchall():
        a_cid = canonical_id_for_code(a_code)
        b_cid = canonical_id_for_code(b_code)
        if a_cid is None or b_cid is None or w_code is None:
            continue
        w_cid = canonical_id_for_code(w_code)
        if w_cid is None:
            continue
        ra, rb = g(a_cid), g(b_cid)
        ea = expected(ra, rb)
        sa = 1.0 if w_cid == a_cid else 0.0
        sb = 1.0 - sa
        elo[a_cid] = ra + ELO_K * (sa - ea)
        elo[b_cid] = rb + ELO_K * (sb - (1.0 - ea))
    return elo


def _get_team_id(conn: sqlite3.Connection, code: str) -> int | None:
    """Return the canonical team's ID for a given code, preferring the normalized code's row."""
    norm = _normalize_code(code)
    cur = conn.cursor()
    # Prefer canonical
    cur.execute("SELECT id FROM teams WHERE code = ?", (norm,))
    row = cur.fetchone()
    if row:
        return row[0]
    # Fallback to raw code if canonical not present
    cur.execute("SELECT id FROM teams WHERE code = ?", (code,))
    row = cur.fetchone()
    return row[0] if row else None

def _canonical_team_ids(conn: sqlite3.Connection, code: str) -> list[int]:
    """Return all team IDs whose codes canonicalize to the same normalized code (e.g., CSS includes CHD)."""
    norm = _normalize_code(code)
    cur = conn.cursor()
    cur.execute("SELECT id, code FROM teams")
    ids: list[int] = []
    for tid, tcode in cur.fetchall():
        if _normalize_code(tcode) == norm:
            ids.append(tid)
    return ids


def _team_hist_stats(conn: sqlite3.Connection, team_ids: list[int]) -> dict:
    """Aggregate historical stats across all IDs that map to the same canonical team code."""
    if not team_ids:
        return {
            'matches_played': 0,
            'win_rate': 0.5,
            'avg_points_scored': 0,
            'avg_points_conceded': 0,
            'avg_attack_points': 0,
            'avg_block_points': 0,
            'avg_serve_points': 0,
        }
    cur = conn.cursor()
    placeholders = ','.join(['?'] * len(team_ids))
    query = f'''
        SELECT 
            COUNT(*) as matches_played,
            SUM(CASE WHEN m.winner_id IN ({placeholders}) THEN 1 ELSE 0 END) as wins,
            AVG(CASE WHEN tms.team_id IN ({placeholders}) THEN tms.total_points END) as avg_points_scored,
            AVG(CASE WHEN tms.team_id NOT IN ({placeholders}) AND tms.match_id IN (
                SELECT m2.id FROM matches m2 WHERE m2.team_a_id IN ({placeholders}) OR m2.team_b_id IN ({placeholders})
            ) THEN tms.total_points END) as avg_points_conceded,
            AVG(CASE WHEN tms.team_id IN ({placeholders}) THEN tms.attack_points END) as avg_attack_points,
            AVG(CASE WHEN tms.team_id IN ({placeholders}) THEN tms.block_points END) as avg_block_points,
            AVG(CASE WHEN tms.team_id IN ({placeholders}) THEN tms.serve_points END) as avg_serve_points
        FROM matches m
        JOIN team_match_stats tms ON m.id = tms.match_id
        WHERE (m.team_a_id IN ({placeholders}) OR m.team_b_id IN ({placeholders})) AND m.status IS NOT NULL
    '''
    # We use the same list of team_ids for each placeholder group
    params = team_ids * 10
    cur.execute(query, params)
    r = cur.fetchone()
    if not r or r[0] == 0:
        return {
            'matches_played': 0,
            'win_rate': 0.5,
            'avg_points_scored': 0,
            'avg_points_conceded': 0,
            'avg_attack_points': 0,
            'avg_block_points': 0,
            'avg_serve_points': 0,
        }
    mp, wins, aps, apc, aatk, ablk, asrv = r
    return {
        'matches_played': mp or 0,
        'win_rate': (wins or 0) / mp if mp else 0.5,
        'avg_points_scored': aps or 0,
        'avg_points_conceded': apc or 0,
        'avg_attack_points': aatk or 0,
        'avg_block_points': ablk or 0,
        'avg_serve_points': asrv or 0,
    }


def _player_agg_stats(conn: sqlite3.Connection, team_ids: list[int]) -> dict:
    cur = conn.cursor()
    if not team_ids:
        return {
            'starter_avg_attack': 0,
            'starter_avg_block': 0,
            'starter_avg_serve': 0,
            'top_scorer_attack': 0,
            'libero_avg_digs': 0,
            'libero_avg_reception': 0,
            'roster_depth': 0,
            'avg_sets_per_player': 0,
            'count_10plus_scorers': 0,
        }
    placeholders = ','.join(['?'] * len(team_ids))
    query = f'''
        SELECT 
            AVG(CASE WHEN is_starter = 1 THEN attack_points ELSE 0 END),
            AVG(CASE WHEN is_starter = 1 THEN block_points ELSE 0 END),
            AVG(CASE WHEN is_starter = 1 THEN serve_points ELSE 0 END),
            MAX(attack_points),
            AVG(CASE WHEN is_libero = 1 THEN dig_excellent ELSE 0 END),
            AVG(CASE WHEN is_libero = 1 THEN reception_excellent ELSE 0 END),
            COUNT(DISTINCT jersey_number),
            AVG(sets_played),
            SUM(CASE WHEN attack_points >= 10 THEN 1 ELSE 0 END)
        FROM player_match_stats pms
        JOIN matches m ON pms.match_id = m.id
        WHERE pms.team_id IN ({placeholders}) AND m.status IS NOT NULL
    '''
    cur.execute(query, team_ids)
    r = cur.fetchone()
    if not r:
        return {
            'starter_avg_attack': 0,
            'starter_avg_block': 0,
            'starter_avg_serve': 0,
            'top_scorer_attack': 0,
            'libero_avg_digs': 0,
            'libero_avg_reception': 0,
            'roster_depth': 0,
            'avg_sets_per_player': 0,
            'count_10plus_scorers': 0,
        }
    return {
        'starter_avg_attack': r[0] or 0,
        'starter_avg_block': r[1] or 0,
        'starter_avg_serve': r[2] or 0,
        'top_scorer_attack': r[3] or 0,
        'libero_avg_digs': r[4] or 0,
        'libero_avg_reception': r[5] or 0,
        'roster_depth': r[6] or 0,
        'avg_sets_per_player': r[7] or 0,
        'count_10plus_scorers': r[8] or 0,
    }


def _build_features_for_pair(conn: sqlite3.Connection, elo_map: dict, team_a_code: str, team_b_code: str, feature_names: list[str]) -> pd.DataFrame:
    """Construct a single-row feature frame matching the training schema."""
    ta_id = _get_team_id(conn, _normalize_code(team_a_code))
    tb_id = _get_team_id(conn, _normalize_code(team_b_code))
    if ta_id is None or tb_id is None:
        raise ValueError(f"Unknown team codes: {team_a_code}, {team_b_code}")
    # For aggregated stats (team + player), include all IDs that map to the canonical code
    ta_ids = _canonical_team_ids(conn, team_a_code)
    tb_ids = _canonical_team_ids(conn, team_b_code)

    # ELOs
    ta_elo = elo_map.get(ta_id, ELO_DEFAULT)
    tb_elo = elo_map.get(tb_id, ELO_DEFAULT)
    elo_diff = ta_elo - tb_elo
    elo_prob_a = 1.0 / (1.0 + 10 ** (-(elo_diff) / 400.0))

    # Team historical
    ta_hist = _team_hist_stats(conn, ta_ids)
    tb_hist = _team_hist_stats(conn, tb_ids)

    # Player aggregated
    ta_pl = _player_agg_stats(conn, ta_ids)
    tb_pl = _player_agg_stats(conn, tb_ids)

    f = {
        'team_a_elo': ta_elo,
        'team_b_elo': tb_elo,
        'elo_diff': elo_diff,
        'elo_prob_team_a': elo_prob_a,
        'team_a_matches_played': ta_hist['matches_played'],
        'team_a_win_rate': ta_hist['win_rate'],
        'team_a_avg_points': ta_hist['avg_points_scored'],
        'team_a_avg_attack': ta_hist['avg_attack_points'],
        'team_a_avg_block': ta_hist['avg_block_points'],
        'team_a_avg_serve': ta_hist['avg_serve_points'],
        'team_b_matches_played': tb_hist['matches_played'],
        'team_b_win_rate': tb_hist['win_rate'],
        'team_b_avg_points': tb_hist['avg_points_scored'],
        'team_b_avg_attack': tb_hist['avg_attack_points'],
        'team_b_avg_block': tb_hist['avg_block_points'],
        'team_b_avg_serve': tb_hist['avg_serve_points'],
        'team_a_starter_attack': ta_pl['starter_avg_attack'],
        'team_a_starter_block': ta_pl['starter_avg_block'],
        'team_a_starter_serve': ta_pl['starter_avg_serve'],
        'team_a_top_scorer': ta_pl['top_scorer_attack'],
        'team_a_libero_digs': ta_pl['libero_avg_digs'],
        'team_a_libero_reception': ta_pl['libero_avg_reception'],
        'team_a_roster_depth': ta_pl['roster_depth'],
        'team_a_avg_sets_per_player': ta_pl['avg_sets_per_player'],
        'team_a_10plus_scorers': ta_pl['count_10plus_scorers'],
        'team_b_starter_attack': tb_pl['starter_avg_attack'],
        'team_b_starter_block': tb_pl['starter_avg_block'],
        'team_b_starter_serve': tb_pl['starter_avg_serve'],
        'team_b_top_scorer': tb_pl['top_scorer_attack'],
        'team_b_libero_digs': tb_pl['libero_avg_digs'],
        'team_b_libero_reception': tb_pl['libero_avg_reception'],
        'team_b_roster_depth': tb_pl['roster_depth'],
        'team_b_avg_sets_per_player': tb_pl['avg_sets_per_player'],
        'team_b_10plus_scorers': tb_pl['count_10plus_scorers'],
    }

    X = pd.DataFrame([f])
    # Align to model's expected columns
    for col in feature_names:
        if col not in X.columns:
            X[col] = 0
    return X[feature_names]

def get_current_standings():
    """Get current match records from database with code normalization (e.g., CHD→CSS)."""
    conn = sqlite3.connect(DB_FILE_STR)
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
        code = _normalize_code(row[0])
        entry = standings.setdefault(code, {'name': row[1], 'wins': 0, 'losses': 0, 'games_played': 0})
        entry['wins'] += row[2]
        entry['losses'] += row[3]
        entry['games_played'] += row[4]
    
    conn.close()
    return standings

def identify_missing_matches():
    """Identify which first-round matches haven't been played yet."""
    # Based on tournament_format.md:
    # First round: two pools of six; each plays 5 matches in-pool.
    # Normalize CHD→CSS
    pool_a = ['HSH', 'FFF', 'CMF', 'CSS', 'CAP', 'NXL']
    pool_b = ['ZUS', 'CCS', 'AKA', 'PGA', 'CTC', 'GTH']
    
    conn = sqlite3.connect(DB_FILE_STR)
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
        team_a, team_b = _normalize_code(row[0]), _normalize_code(row[1])
        played_matchups.add(tuple(sorted([team_a, team_b])))
    
    conn.close()
    
    # Find missing Pool A matches
    missing_matches = []
    for pool, pool_name in [(pool_a, 'Pool A'), (pool_b, 'Pool B')]:
        for i, team_a in enumerate(pool):
            for team_b in pool[i+1:]:
                matchup = tuple(sorted([_normalize_code(team_a), _normalize_code(team_b)]))
                if matchup not in played_matchups:
                    missing_matches.append({
                        'team_a': _normalize_code(team_a),
                        'team_b': _normalize_code(team_b),
                        'pool': pool_name
                    })
    
    return missing_matches

def get_team_historical_features(team_code):
    """Deprecated helper (kept for reference). Use _build_features_for_pair."""
    conn = sqlite3.connect(DB_FILE_STR)
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
    """Deprecated helper for H2H stats (not used in player-aware model features)."""
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
    
    if not result or result[0] == 0:
        return {'h2h_matches': 0, 'h2h_team_a_wins': 0}
    
    return {
        'h2h_matches': result[0],
        'h2h_team_a_wins': result[1] or 0
    }

def predict_match(model, feature_names, conn, elo_map, team_a_code, team_b_code):
    """Predict the winner of a match using the trained model with player-aware features."""
    X = _build_features_for_pair(conn, elo_map, _normalize_code(team_a_code), _normalize_code(team_b_code), feature_names)
    proba = model.predict_proba(X)[0]
    pred = int(proba[1] >= 0.5)
    winner = team_a_code if pred == 1 else team_b_code
    confidence = max(proba[0], proba[1])
    return winner, confidence

def _played_matchups_in_tournament(conn) -> set[tuple[str, str]]:
    cur = conn.cursor()
    cur.execute('''
        SELECT ta.code, tb.code
        FROM matches m
        JOIN teams ta ON m.team_a_id = ta.id
        JOIN teams tb ON m.team_b_id = tb.id
        WHERE m.tournament_id = (SELECT id FROM tournaments WHERE code = 'TEST_PVLR25')
    ''')
    res = set()
    for a,b in cur.fetchall():
        aN, bN = _normalize_code(a), _normalize_code(b)
        res.add(tuple(sorted([aN, bN])))
    return res

def main(model_path: str | None = None, save_outputs: bool = False, keep_latest: int = 0, champion_analysis: bool = False):
    print("=" * 80)
    print(" " * 20 + "PVL REINFORCED CONFERENCE 2025 - TOURNAMENT SIMULATION")
    print("=" * 80)
    
    # Resolve model path
    candidate = None
    if model_path:
        candidate = Path(model_path)
    else:
        # Prefer calibrated player-aware default if present
        cal = Path(MODELS_DIR) / 'calibrated_xgboost_with_players.pkl'
        candidate = cal if cal.exists() else Path(BEST_MODEL_STR)
    if not candidate.exists():
        raise SystemExit(f"Model file not found: {candidate}")
    model_art = joblib.load(candidate)
    model = model_art['model']
    feature_names = model_art.get('feature_names')
    if feature_names is None:
        raise SystemExit("Model artifact missing 'feature_names'; cannot align inference features.")
    
    # Prepare DB connection and ELO map
    conn = sqlite3.connect(DB_FILE_STR)
    elo_map = _compute_current_elo(conn)
    
    # Get current standings
    print("\n[1] CURRENT STANDINGS (from database)")
    print("-" * 80)
    standings = get_current_standings()
    # Snapshot initial standings (normalized codes)
    initial_standings_snapshot = [
        {'team': code, 'wins': rec['wins'], 'losses': rec['losses'], 'games_played': rec['games_played']}
        for code, rec in sorted(standings.items())
    ]
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
    
    # Predict missing matches (First Round completion)
    print("\n[3] PREDICTING REMAINING FIRST-ROUND MATCHES")
    print("-" * 80)
    
    predictions = []  # first-round missing match predictions
    for match in missing_matches:
        winner, confidence = predict_match(model, feature_names, conn, elo_map, match['team_a'], match['team_b'])
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
    # Calculate win percentage and sort (display only)
    for code, team in standings.items():
        team['win_pct'] = team['wins'] / team['games_played'] if team['games_played'] > 0 else 0
    sorted_standings = sorted(standings.items(), key=lambda x: x[1]['win_pct'], reverse=True)
    projected_first_round = [
        {'rank': i, 'team': code, 'wins': rec['wins'], 'losses': rec['losses'], 'games_played': rec['games_played'], 'win_pct': rec['win_pct']}
        for i, (code, rec) in enumerate(sorted_standings, 1)
    ]
    print(f"{'Rank':<6} {'Team':<6} {'Record':<10} {'Win %':<10} {'Games'}")
    print("-" * 80)
    for i, (code, team) in enumerate(sorted_standings, 1):
        medal = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else "  "
        print(f"{medal} {i:<3} {code:<6} {team['wins']}-{team['losses']:<8} {team['win_pct']:.1%}      {team['games_played']}")
    
    # Second Round setup per tournament_format.md
    # Two new pools (carry over first-round results):
    # Pool C = Top3 of Pool A + Bottom3 of Pool B
    # Pool D = Top3 of Pool B + Bottom3 of Pool A
    pool_a = ['HSH', 'FFF', 'CMF', 'CSS', 'CAP', 'NXL']
    pool_b = ['ZUS', 'CCS', 'AKA', 'PGA', 'CTC', 'GTH']
    # Rank each pool using FIVB rules from first-round in-pool matches
    rows = _fetch_tournament_match_rows(conn)
    pool_a_set = set(pool_a)
    pool_b_set = set(pool_b)
    a_rec, b_rec = {}, {}
    for r in rows:
        aC, bC = r['a_code'], r['b_code']
        if aC in pool_a_set and bC in pool_a_set:
            _apply_match(a_rec, aC, bC, r['a_sets'], r['b_sets'], r['a_pts'], r['b_pts'])
        if aC in pool_b_set and bC in pool_b_set:
            _apply_match(b_rec, aC, bC, r['a_sets'], r['b_sets'], r['a_pts'], r['b_pts'])
    a_ranked = _rank_fivb(a_rec, pool_a, rows)
    b_ranked = _rank_fivb(b_rec, pool_b, rows)
    a_top3, a_bot3 = a_ranked[:3], a_ranked[3:]
    b_top3, b_bot3 = b_ranked[:3], b_ranked[3:]
    pool_c = a_top3 + b_bot3
    pool_d = b_top3 + a_bot3

    # Determine cross-pool matches to play (3 per team vs teams from other original pool)
    print("\n[5] SECOND ROUND (cross-pool) - simulating remaining 3 matches per team")
    print("-" * 80)
    played = _played_matchups_in_tournament(conn)

    # Build combined standings from all finished matches in DB (for proper FIVB ranking)
    combined = {}
    for r in rows:
        _apply_match(combined, r['a_code'], r['b_code'], r['a_sets'], r['b_sets'], r['a_pts'], r['b_pts'])

    def simulate_second_round(pool, origin_top3, origin_other_bot3, origin_label):
        # Teams from origin_top3 should play vs origin_other_bot3 (not faced in R1)
        for a in origin_top3:
            for b in origin_other_bot3:
                pair = tuple(sorted([a,b]))
                if pair in played:
                    continue
                w, conf = predict_match(model, feature_names, conn, elo_map, a, b)
                l = b if w == a else a
                # Approximate set score and points for FIVB match points and ratios
                w_sets, l_sets, w_pts, l_pts = _approx_simulated_sets(conf)
                # Apply to combined standings
                if w == a:
                    _apply_match(combined, a, b, w_sets, l_sets, w_pts, l_pts)
                else:
                    _apply_match(combined, a, b, l_sets, w_sets, l_pts, w_pts)
                played.add(pair)
                print(f"  {origin_label}: {a} vs {b} → {w} ({conf:.1%})")

    simulate_second_round(pool_c, a_top3, b_bot3, "Pool C")
    simulate_second_round(pool_c, b_bot3, a_top3, "Pool C")
    simulate_second_round(pool_d, b_top3, a_bot3, "Pool D")
    simulate_second_round(pool_d, a_bot3, b_top3, "Pool D")

    # Recompute combined standings after second round using FIVB rules
    _finalize_ratios(combined)
    ordered_codes = _rank_fivb(combined, match_rows=rows)
    sorted_combined = [(c, combined[c]) for c in ordered_codes]
    print("\n[6] COMBINED STANDINGS AFTER SECOND ROUND")
    print("-" * 80)
    print(f"{'Rank':<6} {'Team':<6} {'Record':<10} {'MPts':<6} {'SetRt':<8} {'PtRt':<8} {'Games'}")
    print("-" * 80)
    for i, (code, team) in enumerate(sorted_combined, 1):
        medal = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else "  "
        print(f"{medal} {i:<3} {code:<6} {team['wins']}-{team['losses']:<8} "
              f"{team.get('match_points',0):<6} {team.get('set_ratio',0):<8.3f} {team.get('point_ratio',0):<8.3f} {team['games_played']}")

    # Top 8 advance to quarterfinals
    print("\n[7] QUARTERFINAL MATCHUPS (Top 8 advance)")
    print("-" * 80)
    top_8 = sorted_combined[:8]
    matchups = [
        (top_8[0][0], top_8[7][0], "QF1"),
        (top_8[1][0], top_8[6][0], "QF2"),
        (top_8[2][0], top_8[5][0], "QF3"),
        (top_8[3][0], top_8[4][0], "QF4"),
    ]
    
    qf_winners = []
    qf_results = []
    for team_a, team_b, qf_name in matchups:
        winner, confidence = predict_match(model, feature_names, conn, elo_map, team_a, team_b)
        qf_winners.append(winner)
        print(f"{qf_name}: #{matchups.index((team_a, team_b, qf_name))+1} {team_a} vs "
              f"#{8-matchups.index((team_a, team_b, qf_name))} {team_b} → "
              f"Winner: {winner} ({confidence:.1%})")
        qf_results.append({
            'match': qf_name,
            'team_a': team_a,
            'team_b': team_b,
            'winner': winner,
            'confidence': confidence
        })
    
    # Semifinals
    print("\n[8] SEMIFINAL MATCHUPS")
    print("-" * 80)
    sf_matchups = [
        (qf_winners[0], qf_winners[3], "SF1"),  # QF1 winner vs QF4 winner
        (qf_winners[1], qf_winners[2], "SF2"),  # QF2 winner vs QF3 winner
    ]
    
    sf_winners = []
    sf_results = []
    for team_a, team_b, sf_name in sf_matchups:
        winner, confidence = predict_match(model, feature_names, conn, elo_map, team_a, team_b)
        sf_winners.append(winner)
        print(f"{sf_name}: {team_a} vs {team_b} → Winner: {winner} ({confidence:.1%})")
        sf_results.append({
            'match': sf_name,
            'team_a': team_a,
            'team_b': team_b,
            'winner': winner,
            'confidence': confidence
        })
    
    # Finals
    print("\n[9] CHAMPIONSHIP MATCH")
    print("-" * 80)
    champion, confidence = predict_match(model, feature_names, conn, elo_map, sf_winners[0], sf_winners[1])
    runner_up = sf_winners[1] if champion == sf_winners[0] else sf_winners[0]
    
    print(f"🏆 {sf_winners[0]} vs {sf_winners[1]}")
    print(f"\n🥇 PREDICTED CHAMPION: {champion} ({confidence:.1%} confidence)")
    print(f"🥈 PREDICTED RUNNER-UP: {runner_up}")
    
    # Optional champion analysis
    if champion_analysis:
        print("\n" + "=" * 80)
        print("CHAMPION ANALYSIS")
        print("=" * 80)
        # Determine bracket favorite (highest seed)
        bracket_favorite = top_8[0][0]
        champ_seed = next((i+1 for i, (c, _) in enumerate(top_8) if c == champion), None)
        runner_seed = next((i+1 for i, (c, _) in enumerate(top_8) if c == runner_up), None)
        
        print(f"Bracket Favorite (Seed #1): {bracket_favorite}")
        print(f"Champion Seed: #{champ_seed} ({champion})")
        print(f"Runner-Up Seed: #{runner_seed} ({runner_up})")
        
        if champion == bracket_favorite:
            print(f"\n✅ Favorite won as expected (confidence: {confidence:.1%})")
        else:
            print(f"\n⚠️  Upset! #{champ_seed} {champion} defeated favorite #{1} {bracket_favorite}")
        
        # Average bracket confidence
        all_bracket_confs = [r['confidence'] for r in qf_results + sf_results] + [confidence]
        avg_conf = sum(all_bracket_confs) / len(all_bracket_confs)
        print(f"\nAverage Bracket Confidence: {avg_conf:.1%}")
        
        if avg_conf > 0.70:
            print("High confidence bracket (low upset likelihood)")
        elif avg_conf > 0.60:
            print("Moderate confidence bracket")
        else:
            print("Low confidence bracket (high upset likelihood)")
    
    print("\n" + "=" * 80)
    print("SIMULATION COMPLETE!")
    print("=" * 80)

    # --------------------- Persist Outputs (optional) ---------------------
    if save_outputs:
        out_dir = Path('outputs')
        out_dir.mkdir(parents=True, exist_ok=True)
        # Use timezone-aware UTC timestamp to avoid deprecation warnings
        ts = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
        json_path = out_dir / f'tournament_simulation_{ts}.json'
        text_path = out_dir / f'tournament_simulation_{ts}.txt'
        result_payload = {
            'timestamp_utc': ts,
            'model_path': str(candidate),
            'initial_standings': initial_standings_snapshot,
            'missing_first_round_matches': missing_matches,
            'predicted_first_round_matches': predictions,
            'projected_first_round_standings': projected_first_round,
            'second_round_pool_c': pool_c,
            'second_round_pool_d': pool_d,
            'second_round_combined_rankings': [
                {
                    'rank': i,
                    'team': code,
                    'wins': rec['wins'],
                    'losses': rec['losses'],
                    'match_points': rec.get('match_points', 0),
                    'set_ratio': rec.get('set_ratio', 0.0),
                    'point_ratio': rec.get('point_ratio', 0.0),
                    'games_played': rec['games_played']
                } for i, (code, rec) in enumerate(sorted_combined, 1)
            ],
            'quarterfinals': qf_results,
            'semifinals': sf_results,
            'championship': {
                'team_a': sf_winners[0],
                'team_b': sf_winners[1],
                'champion': champion,
                'runner_up': runner_up,
                'confidence': confidence
            }
        }
        with open(json_path, 'w') as fjson:
            json.dump(result_payload, fjson, indent=2)
        # Minimal text summary
        with open(text_path, 'w') as ftxt:
            ftxt.write("PVL Tournament Simulation\n")
            ftxt.write(f"Timestamp (UTC): {ts}\nModel: {candidate}\n\n")
            ftxt.write("Initial Standings:\n")
            for s in initial_standings_snapshot:
                ftxt.write(f"  {s['team']}: {s['wins']}-{s['losses']} ({s['games_played']} gp)\n")
            ftxt.write("\nMissing First Round Matches:\n")
            for m in missing_matches:
                ftxt.write(f"  {m['pool']}: {m['team_a']} vs {m['team_b']}\n")
            ftxt.write("\nPredicted Remaining First Round Matches:\n")
            for p in predictions:
                ftxt.write(f"  {p['team_a']} vs {p['team_b']} → {p['winner']} ({p['confidence']:.1%})\n")
            ftxt.write("\nProjected First Round Standings (Top 10 shown):\n")
            for row in projected_first_round[:10]:
                ftxt.write(f"  #{row['rank']} {row['team']} {row['wins']}-{row['losses']} Win%={row['win_pct']:.1%}\n")
            ftxt.write("\nSecond Round Combined Rankings (Top 10 shown):\n")
            for row in result_payload['second_round_combined_rankings'][:10]:
                ftxt.write(f"  #{row['rank']} {row['team']} {row['wins']}-{row['losses']} MPts={row['match_points']} SR={row['set_ratio']:.3f} PR={row['point_ratio']:.3f}\n")
            ftxt.write("\nQuarterfinals:\n")
            for q in qf_results:
                ftxt.write(f"  {q['match']}: {q['team_a']} vs {q['team_b']} → {q['winner']} ({q['confidence']:.1%})\n")
            ftxt.write("\nSemifinals:\n")
            for sf in sf_results:
                ftxt.write(f"  {sf['match']}: {sf['team_a']} vs {sf['team_b']} → {sf['winner']} ({sf['confidence']:.1%})\n")
            ftxt.write("\nChampionship:\n")
            ftxt.write(f"  {sf_winners[0]} vs {sf_winners[1]} → Champion: {champion} ({confidence:.1%})\n")
            ftxt.write(f"Runner-Up: {runner_up}\n")
        print(f"\n✓ Outputs saved: {json_path.name}, {text_path.name}")
        # Optional housekeeping: keep only latest N simulation outputs
        if keep_latest and keep_latest > 0:
            from glob import glob
            def ts_from_name(name: str):
                # expects tournament_simulation_YYYYMMDD_HHMMSS.ext
                try:
                    base = Path(name).stem
                    ts_part = base.replace('tournament_simulation_', '')
                    return datetime.strptime(ts_part, '%Y%m%d_%H%M%S')
                except Exception:
                    return datetime.min
            files = []
            for pat in (str(out_dir / 'tournament_simulation_*.json'), str(out_dir / 'tournament_simulation_*.txt')):
                for fp in glob(pat):
                    files.append(Path(fp))
            files_sorted = sorted(files, key=lambda p: ts_from_name(p.name), reverse=True)
            to_remove = files_sorted[keep_latest:]
            removed = []
            for p in to_remove:
                try:
                    p.unlink()
                    removed.append(p.name)
                except Exception:
                    pass
            if removed:
                print(f"✓ Housekeeping: removed older outputs: {', '.join(removed[:5])}{'...' if len(removed) > 5 else ''}")

if __name__ == "__main__":
    # Allow direct execution with optional CLI arg --model and --save_outputs
    import argparse
    p = argparse.ArgumentParser(description="Tournament simulation with player-aware model")
    p.add_argument('--model', type=str, default=None, help='Path to model artifact (.pkl)')
    p.add_argument('--save_outputs', action='store_true', help='Persist simulation outputs to outputs/ directory')
    p.add_argument('--keep_latest', type=int, default=0, help='If >0, keep only the latest N simulation outputs (json+txt)')
    p.add_argument('--champion_analysis', action='store_true', help='Print champion analysis (favorite, seeds, upset likelihood)')
    args = p.parse_args()
    main(model_path=args.model, save_outputs=args.save_outputs, keep_latest=args.keep_latest, champion_analysis=args.champion_analysis)
