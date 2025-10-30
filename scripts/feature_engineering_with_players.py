"""
Enhanced Feature Engineering with Player-Level Data
Combines team statistics with player performance metrics
"""

import sqlite3
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple


class PlayerEnhancedFeatureExtractor:
    """Extract ML features including player-level statistics"""
    
    def __init__(self, db_path: str = 'volleyball_data.db'):
        self.db_path = db_path
        self.conn = None
        
    def connect(self):
        """Connect to database"""
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row
        
    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()
    
    def get_team_historical_stats(self, team_id: int, before_match_id: int) -> Dict:
        """Get team's historical statistics before a given match"""
        query = '''
            SELECT 
                COUNT(*) as matches_played,
                SUM(CASE WHEN winner_id = ? THEN 1 ELSE 0 END) as wins,
                AVG(CASE WHEN team_id = ? THEN total_points ELSE 0 END) as avg_points_scored,
                AVG(CASE WHEN team_id != ? THEN total_points ELSE 0 END) as avg_points_conceded,
                AVG(CASE WHEN team_id = ? THEN attack_points ELSE 0 END) as avg_attack_points,
                AVG(CASE WHEN team_id = ? THEN block_points ELSE 0 END) as avg_block_points,
                AVG(CASE WHEN team_id = ? THEN serve_points ELSE 0 END) as avg_serve_points
            FROM matches m
            JOIN team_match_stats tms ON m.id = tms.match_id
            WHERE (team_a_id = ? OR team_b_id = ?)
                AND m.id < ?
                AND status IS NOT NULL
        '''
        
        cursor = self.conn.execute(query, (
            team_id, team_id, team_id, team_id, team_id, team_id,
            team_id, team_id, before_match_id
        ))
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
        query = '''
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
            WHERE pms.team_id = ?
                AND m.id < ?
                AND m.status IS NOT NULL
        '''
        
        cursor = self.conn.execute(query, (team_id, before_match_id))
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
            SELECT id, team_a_id, team_b_id, winner_id, tournament_id, match_no
            FROM matches
            WHERE status IS NOT NULL
            ORDER BY id
        '''
        
        matches_df = pd.read_sql_query(matches_query, self.conn)
        
        features_list = []
        
        for idx, match in matches_df.iterrows():
            match_id = match['id']
            team_a_id = match['team_a_id']
            team_b_id = match['team_b_id']
            winner_id = match['winner_id']
            
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
    features_df.to_csv('volleyball_features_with_players.csv', index=False)
    X.to_csv('X_features_with_players.csv', index=False)
    y.to_csv('y_target_with_players.csv', index=False)
    
    print("\n✓ Saved files:")
    print("  - volleyball_features_with_players.csv (full dataset)")
    print("  - X_features_with_players.csv (features only)")
    print("  - y_target_with_players.csv (labels only)")
    
    print(f"\n📊 Feature Summary:")
    print(f"  Samples: {len(X)}")
    print(f"  Features: {len(feature_cols)}")
    print(f"  Team A wins: {y.sum()}")
    print(f"  Team B wins: {len(y) - y.sum()}")
    
    print("\n✓ Feature groups:")
    print(f"  - Team historical stats: 12 features (6 per team)")
    print(f"  - Player-based stats: 18 features (9 per team)")
    print(f"  - Total: {len(feature_cols)} features")
    
    print("\n" + "="*80)
    print("FEATURE EXTRACTION COMPLETE!")
    print("="*80)
    print("\nNext step: Train XGBoost model with player features")


if __name__ == '__main__':
    main()
