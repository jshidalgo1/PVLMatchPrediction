"""
Enhanced Feature Engineering with Player-Level Data
Combines team statistics with player performance metrics

Update:
- Adds ELO strength features computed chronologically per team
- Outputs are saved to configured CSV paths
"""

import sqlite3
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from config import DB_FILE_STR, CSV_DIR_STR, VOLLEYBALL_FEATURES_STR, X_FEATURES_STR, Y_TARGET_STR, canonicalize_team_code


class PlayerEnhancedFeatureExtractor:
    """Extract ML features including player-level statistics"""
    
    def __init__(self, db_path: str = DB_FILE_STR):
        self.db_path = db_path
        self.conn = None
        # ELO configuration
        self.elo_default = 1500.0
        self.elo_k = 20.0
        
    def connect(self):
        """Connect to database"""
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row
        
    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()
    
    def _expand_canonical_team_ids(self, team_id: int) -> List[int]:
        """Return all team IDs whose codes map to the same canonical code as the provided team_id."""
        cur = self.conn.cursor()
        cur.execute("SELECT code FROM teams WHERE id = ?", (team_id,))
        row = cur.fetchone()
        if not row:
            return [team_id]
        canon = canonicalize_team_code(row[0])
        cur.execute("SELECT id, code FROM teams")
        ids = []
        for tid, code in cur.fetchall():
            if canonicalize_team_code(code) == canon:
                ids.append(tid)
        return ids or [team_id]

    def get_team_historical_stats(self, team_id: int, before_match_id: int) -> Dict:
        """Get team's historical statistics before a given match"""
        team_ids = self._expand_canonical_team_ids(team_id)
        ph = ','.join(['?'] * len(team_ids))
        # Using matches join to access winner_id and restrict by time
        query = f'''
            SELECT 
                COUNT(DISTINCT m.id) as matches_played,
                SUM(CASE WHEN m.winner_id IN ({ph}) THEN 1 ELSE 0 END) as wins,
                AVG(CASE WHEN tms.team_id IN ({ph}) THEN tms.total_points END) as avg_points_scored,
                AVG(CASE WHEN tms.team_id NOT IN ({ph}) THEN tms.total_points END) as avg_points_conceded,
                AVG(CASE WHEN tms.team_id IN ({ph}) THEN tms.attack_points END) as avg_attack_points,
                AVG(CASE WHEN tms.team_id IN ({ph}) THEN tms.block_points END) as avg_block_points,
                AVG(CASE WHEN tms.team_id IN ({ph}) THEN tms.serve_points END) as avg_serve_points
            FROM matches m
            JOIN team_match_stats tms ON m.id = tms.match_id
            WHERE (m.team_a_id IN ({ph}) OR m.team_b_id IN ({ph}))
              AND m.id < ?
              AND m.status IS NOT NULL
        '''
        # Parameter order corresponds to each (ph) appearance sequentially
        params = (
            team_ids +  # winner_id IN
            team_ids +  # tms.team_id IN (points scored)
            team_ids +  # tms.team_id NOT IN (points conceded)
            team_ids +  # attack
            team_ids +  # block
            team_ids +  # serve
            team_ids +  # m.team_a_id IN
            team_ids +  # m.team_b_id IN
            [before_match_id]
        )
        cursor = self.conn.execute(query, params)
        row = cursor.fetchone()
        
        if not row or row['matches_played'] == 0:
            return {
                'matches_played': 0,
                'win_rate': 0.5,
                'avg_points_scored': 0,
                'avg_points_conceded': 0,
                'avg_attack_points': 0,
                'avg_block_points': 0,
                'avg_serve_points': 0
            }
        
        return {
            'matches_played': row['matches_played'],
            'win_rate': row['wins'] / row['matches_played'] if row['matches_played'] > 0 else 0.5,
            'avg_points_scored': row['avg_points_scored'] or 0,
            'avg_points_conceded': row['avg_points_conceded'] or 0,
            'avg_attack_points': row['avg_attack_points'] or 0,
            'avg_block_points': row['avg_block_points'] or 0,
            'avg_serve_points': row['avg_serve_points'] or 0
        }
    
    def get_player_historical_stats(self, team_id: int, before_match_id: int) -> Dict:
        """Get aggregated player statistics for a team before a given match"""
        team_ids = self._expand_canonical_team_ids(team_id)
        ph = ','.join(['?'] * len(team_ids))
        query = f'''
            SELECT 
                AVG(CASE WHEN is_starter = 1 THEN attack_points ELSE 0 END) as starter_avg_attack,
                AVG(CASE WHEN is_starter = 1 THEN block_points ELSE 0 END) as starter_avg_block,
                AVG(CASE WHEN is_starter = 1 THEN serve_points ELSE 0 END) as starter_avg_serve,
                MAX(attack_points) as top_scorer_attack,
                AVG(CASE WHEN is_libero = 1 THEN dig_excellent ELSE 0 END) as libero_avg_digs,
                AVG(CASE WHEN is_libero = 1 THEN reception_excellent ELSE 0 END) as libero_avg_reception,
                COUNT(DISTINCT jersey_number) as roster_depth,
                AVG(sets_played) as avg_sets_per_player,
                SUM(CASE WHEN attack_points >= 10 THEN 1 ELSE 0 END) as count_10plus_scorers
            FROM player_match_stats pms
            JOIN matches m ON pms.match_id = m.id
            WHERE pms.team_id IN ({ph})
              AND m.id < ?
              AND m.status IS NOT NULL
        '''
        params = team_ids + [before_match_id]
        cursor = self.conn.execute(query, params)
        row = cursor.fetchone()
        
        if not row:
            return {
                'starter_avg_attack': 0,
                'starter_avg_block': 0,
                'starter_avg_serve': 0,
                'top_scorer_attack': 0,
                'libero_avg_digs': 0,
                'libero_avg_reception': 0,
                'roster_depth': 0,
                'avg_sets_per_player': 0,
                'count_10plus_scorers': 0
            }
        
        return {
            'starter_avg_attack': row['starter_avg_attack'] or 0,
            'starter_avg_block': row['starter_avg_block'] or 0,
            'starter_avg_serve': row['starter_avg_serve'] or 0,
            'top_scorer_attack': row['top_scorer_attack'] or 0,
            'libero_avg_digs': row['libero_avg_digs'] or 0,
            'libero_avg_reception': row['libero_avg_reception'] or 0,
            'roster_depth': row['roster_depth'] or 0,
            'avg_sets_per_player': row['avg_sets_per_player'] or 0,
            'count_10plus_scorers': row['count_10plus_scorers'] or 0
        }
    
    def extract_features(self) -> pd.DataFrame:
        """Extract all features for all matches"""
        self.connect()
        
        # Get all completed matches
        matches_query = '''
            SELECT m.id, m.team_a_id, m.team_b_id, m.winner_id, m.tournament_id, m.match_no,
                   ta.code AS team_a_code, tb.code AS team_b_code, w.code AS winner_code
            FROM matches m
            JOIN teams ta ON m.team_a_id = ta.id
            JOIN teams tb ON m.team_b_id = tb.id
            LEFT JOIN teams w ON m.winner_id = w.id
            WHERE m.status IS NOT NULL
            ORDER BY m.id
        '''
        matches_df = pd.read_sql_query(matches_query, self.conn)
        
        features_list = []
        # Initialize ELO ratings per team_id
        elo: Dict[int, float] = {}
        
        def get_elo(team_id: int) -> float:
            return elo.get(team_id, self.elo_default)
        
        def expected_score(r_a: float, r_b: float) -> float:
            return 1.0 / (1.0 + 10 ** (-(r_a - r_b) / 400.0))
        
        def update_elo(team_a_id: int, team_b_id: int, winner_id: int):
            # Collapse ELO across canonical codes by using representative canonical id (lowest id among group)
            a_group = sorted(self._expand_canonical_team_ids(team_a_id))
            b_group = sorted(self._expand_canonical_team_ids(team_b_id))
            a_rep = a_group[0]
            b_rep = b_group[0]
            ra = get_elo(a_rep)
            rb = get_elo(b_rep)
            ea = expected_score(ra, rb)
            sa = 1.0 if winner_id in a_group else 0.0
            sb = 1.0 - sa
            ra_new = ra + self.elo_k * (sa - ea)
            rb_new = rb + self.elo_k * (sb - (1.0 - ea))
            elo[a_rep] = ra_new
            elo[b_rep] = rb_new
        
        for idx, match in matches_df.iterrows():
            match_id = match['id']
            team_a_id = match['team_a_id']
            team_b_id = match['team_b_id']
            winner_id = match['winner_id']
            
            # ELO before this match (no leakage)
            # Representative canonical IDs for ELO lookup
            a_rep = sorted(self._expand_canonical_team_ids(team_a_id))[0]
            b_rep = sorted(self._expand_canonical_team_ids(team_b_id))[0]
            team_a_elo = get_elo(a_rep)
            team_b_elo = get_elo(b_rep)
            elo_diff = team_a_elo - team_b_elo
            elo_prob_team_a = 1.0 / (1.0 + 10 ** (-(elo_diff) / 400.0))
            
            # Get historical stats for both teams
            team_a_stats = self.get_team_historical_stats(team_a_id, match_id)
            team_b_stats = self.get_team_historical_stats(team_b_id, match_id)
            
            # Get player-based stats
            team_a_players = self.get_player_historical_stats(team_a_id, match_id)
            team_b_players = self.get_player_historical_stats(team_b_id, match_id)
            
            # Create feature vector
            features = {
                # Match metadata
                'match_id': match_id,
                'team_a_id': team_a_id,
                'team_b_id': team_b_id,
                'tournament_id': match['tournament_id'],
                
                # ELO features (pre-match)
                'team_a_elo': team_a_elo,
                'team_b_elo': team_b_elo,
                'elo_diff': elo_diff,
                'elo_prob_team_a': elo_prob_team_a,
                
                # Team A historical stats
                'team_a_matches_played': team_a_stats['matches_played'],
                'team_a_win_rate': team_a_stats['win_rate'],
                'team_a_avg_points': team_a_stats['avg_points_scored'],
                'team_a_avg_attack': team_a_stats['avg_attack_points'],
                'team_a_avg_block': team_a_stats['avg_block_points'],
                'team_a_avg_serve': team_a_stats['avg_serve_points'],
                
                # Team B historical stats
                'team_b_matches_played': team_b_stats['matches_played'],
                'team_b_win_rate': team_b_stats['win_rate'],
                'team_b_avg_points': team_b_stats['avg_points_scored'],
                'team_b_avg_attack': team_b_stats['avg_attack_points'],
                'team_b_avg_block': team_b_stats['avg_block_points'],
                'team_b_avg_serve': team_b_stats['avg_serve_points'],
                
                # Team A player-based features
                'team_a_starter_attack': team_a_players['starter_avg_attack'],
                'team_a_starter_block': team_a_players['starter_avg_block'],
                'team_a_starter_serve': team_a_players['starter_avg_serve'],
                'team_a_top_scorer': team_a_players['top_scorer_attack'],
                'team_a_libero_digs': team_a_players['libero_avg_digs'],
                'team_a_libero_reception': team_a_players['libero_avg_reception'],
                'team_a_roster_depth': team_a_players['roster_depth'],
                'team_a_avg_sets_per_player': team_a_players['avg_sets_per_player'],
                'team_a_10plus_scorers': team_a_players['count_10plus_scorers'],
                
                # Team B player-based features
                'team_b_starter_attack': team_b_players['starter_avg_attack'],
                'team_b_starter_block': team_b_players['starter_avg_block'],
                'team_b_starter_serve': team_b_players['starter_avg_serve'],
                'team_b_top_scorer': team_b_players['top_scorer_attack'],
                'team_b_libero_digs': team_b_players['libero_avg_digs'],
                'team_b_libero_reception': team_b_players['libero_avg_reception'],
                'team_b_roster_depth': team_b_players['roster_depth'],
                'team_b_avg_sets_per_player': team_b_players['avg_sets_per_player'],
                'team_b_10plus_scorers': team_b_players['count_10plus_scorers'],
                
                # Target variable (1 if Team A wins, 0 if Team B wins)
                'team_a_wins': 1 if winner_id == team_a_id else 0
            }
            
            features_list.append(features)
            
            if (idx + 1) % 100 == 0:
                print(f"  Processed {idx + 1}/{len(matches_df)} matches...")
            
            # Update ELO AFTER recording features for this match
            update_elo(team_a_id, team_b_id, winner_id)
        
        self.close()
        
        return pd.DataFrame(features_list)


def main():
    """Extract features and save to files"""
    print("="*80)
    print(" "*20 + "ENHANCED FEATURE ENGINEERING WITH PLAYERS")
    print("="*80)
    
    print("\nExtracting features with player-level data...")
    extractor = PlayerEnhancedFeatureExtractor()
    features_df = extractor.extract_features()
    
    print(f"\n✓ Extracted features for {len(features_df)} matches")
    print(f"  Total features: {len(features_df.columns) - 5} (excluding metadata + target)")
    
    # Separate features and target
    metadata_cols = ['match_id', 'team_a_id', 'team_b_id', 'tournament_id']
    target_col = 'team_a_wins'
    feature_cols = [col for col in features_df.columns 
                   if col not in metadata_cols and col != target_col]
    
    X = features_df[feature_cols]
    y = features_df[target_col]
    
    # Save files
    # Ensure directory exists
    from pathlib import Path
    Path(CSV_DIR_STR).mkdir(parents=True, exist_ok=True)
    features_df.to_csv(VOLLEYBALL_FEATURES_STR, index=False)
    X.to_csv(X_FEATURES_STR, index=False)
    y.to_csv(Y_TARGET_STR, index=False)
    
    print("\n✓ Saved files:")
    print(f"  - {VOLLEYBALL_FEATURES_STR} (full dataset)")
    print(f"  - {X_FEATURES_STR} (features only)")
    print(f"  - {Y_TARGET_STR} (labels only)")
    
    print(f"\n📊 Feature Summary:")
    print(f"  Samples: {len(X)}")
    print(f"  Features: {len(feature_cols)}")
    print(f"  Team A wins: {y.sum()}")
    print(f"  Team B wins: {len(y) - y.sum()}")
    
    print("\n✓ Feature groups:")
    print(f"  - ELO features: 4 features (team_a_elo, team_b_elo, elo_diff, elo_prob_team_a)")
    print(f"  - Team historical stats: 12 features (6 per team)")
    print(f"  - Player-based stats: 18 features (9 per team)")
    print(f"  - Total: {len(feature_cols)} features")
    
    print("\n" + "="*80)
    print("FEATURE EXTRACTION COMPLETE!")
    print("="*80)
    print("\nNext step: Train XGBoost model with player features")


if __name__ == '__main__':
    main()
