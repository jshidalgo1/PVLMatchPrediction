"""
Feature Engineering for Volleyball Match Prediction
Transforms raw match data into features suitable for XGBoost training
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any
import json
from collections import defaultdict


class VolleyballFeatureEngineer:
    """Create ML features from volleyball match data"""
    
    def __init__(self, matches_data: List[Dict[str, Any]]):
        self.matches_data = matches_data
        self.team_history = defaultdict(lambda: {
            'matches_played': 0,
            'matches_won': 0,
            'sets_won': 0,
            'sets_lost': 0,
            'total_points_scored': 0,
            'total_points_conceded': 0,
            'stats': defaultdict(int)
        })
        
    def create_match_features(self) -> pd.DataFrame:
        """Create feature matrix for all matches"""
        features = []
        
        for match in self.matches_data:
            if 'teams' not in match or 'team_a' not in match['teams']:
                continue
                
            team_a_code = match['teams']['team_a']['code']
            team_b_code = match['teams']['team_b']['code']
            
            # Create features for this match
            match_features = self._create_single_match_features(match, team_a_code, team_b_code)
            
            if match_features:
                features.append(match_features)
            
            # Update team histories after processing match
            self._update_team_history(match, team_a_code, team_b_code)
        
        df = pd.DataFrame(features)
        return df
    
    def _create_single_match_features(self, match: Dict, team_a_code: str, team_b_code: str) -> Dict:
        """Create features for a single match - ONLY PRE-MATCH FEATURES"""
        team_a = match['teams']['team_a']
        team_b = match['teams']['team_b']
        
        features = {
            # Match identifiers
            'match_file': match['file_name'],
            'tournament': match['tournament'].get('code', 'unknown'),
            'team_a': team_a_code,
            'team_b': team_b_code,
            'date': match['match_info'].get('date'),
            
            # Target variable (1 if team_a won, 0 if team_b won)
            'team_a_won': 1 if match['winner'] == team_a_code else 0,
            
            # ⚠️ REMOVED: sets_won, set_scores, match statistics
            # These reveal the outcome and can't be used for prediction!
        }
        
        # Only add PRE-MATCH features (historical performance)
        features.update(self._get_historical_features(team_a_code, 'team_a'))
        features.update(self._get_historical_features(team_b_code, 'team_b'))
        
        # Add head-to-head features (from previous matches)
        features.update(self._get_head_to_head_features(team_a_code, team_b_code))
        
        return features
    
    def _extract_stat_features(self, statistics: Dict, prefix: str) -> Dict:
        """Extract statistical features for a team"""
        stats = statistics.get('total_stats', {})
        features = {}
        
        # Key volleyball statistics
        key_stats = [
            'AtkPoint', 'AtkFault', 'AtkCont',
            'BAtkPoint', 'BAtkFault', 'BAtkCont',
            'BlkPoint', 'BlkFault', 'BlkCont',
            'SrvPoint', 'SrvFault', 'SrvCont',
            'DigExcel', 'DigFault', 'DigCont',
            'SetExcel', 'SetFault', 'SetCont',
            'RecExcel', 'RecFault', 'RecCont',
            'team_OppError'
        ]
        
        for stat in key_stats:
            features[f'{prefix}_{stat}'] = stats.get(stat, 0)
        
        # Calculate derived metrics
        total_attacks = stats.get('AtkPoint', 0) + stats.get('AtkFault', 0) + stats.get('AtkCont', 0)
        if total_attacks > 0:
            features[f'{prefix}_attack_efficiency'] = stats.get('AtkPoint', 0) / total_attacks
            features[f'{prefix}_attack_error_rate'] = stats.get('AtkFault', 0) / total_attacks
        else:
            features[f'{prefix}_attack_efficiency'] = 0
            features[f'{prefix}_attack_error_rate'] = 0
        
        total_serves = stats.get('SrvPoint', 0) + stats.get('SrvFault', 0) + stats.get('SrvCont', 0)
        if total_serves > 0:
            features[f'{prefix}_serve_efficiency'] = stats.get('SrvPoint', 0) / total_serves
            features[f'{prefix}_serve_error_rate'] = stats.get('SrvFault', 0) / total_serves
        else:
            features[f'{prefix}_serve_efficiency'] = 0
            features[f'{prefix}_serve_error_rate'] = 0
        
        total_receptions = stats.get('RecExcel', 0) + stats.get('RecFault', 0) + stats.get('RecCont', 0)
        if total_receptions > 0:
            features[f'{prefix}_reception_efficiency'] = stats.get('RecExcel', 0) / total_receptions
            features[f'{prefix}_reception_error_rate'] = stats.get('RecFault', 0) / total_receptions
        else:
            features[f'{prefix}_reception_efficiency'] = 0
            features[f'{prefix}_reception_error_rate'] = 0
        
        return features
    
    def _extract_set_score_features(self, set_scores: List[int], prefix: str) -> Dict:
        """Extract features from set scores"""
        features = {
            f'{prefix}_total_points': sum(set_scores),
            f'{prefix}_avg_points_per_set': np.mean(set_scores) if set_scores else 0,
            f'{prefix}_sets_above_20': sum(1 for score in set_scores if score >= 20),
            f'{prefix}_sets_above_25': sum(1 for score in set_scores if score >= 25),
        }
        return features
    
    def _get_historical_features(self, team_code: str, prefix: str) -> Dict:
        """Get historical performance features for a team"""
        history = self.team_history[team_code]
        
        features = {
            f'{prefix}_prev_matches_played': history['matches_played'],
            f'{prefix}_prev_matches_won': history['matches_won'],
            f'{prefix}_prev_win_rate': history['matches_won'] / history['matches_played'] if history['matches_played'] > 0 else 0.5,
            f'{prefix}_prev_sets_won': history['sets_won'],
            f'{prefix}_prev_sets_lost': history['sets_lost'],
            f'{prefix}_prev_set_win_rate': history['sets_won'] / (history['sets_won'] + history['sets_lost']) if (history['sets_won'] + history['sets_lost']) > 0 else 0.5,
            f'{prefix}_prev_avg_points_scored': history['total_points_scored'] / history['matches_played'] if history['matches_played'] > 0 else 0,
            f'{prefix}_prev_avg_points_conceded': history['total_points_conceded'] / history['matches_played'] if history['matches_played'] > 0 else 0,
        }
        
        return features
    
    def _get_head_to_head_features(self, team_a_code: str, team_b_code: str) -> Dict:
        """Get head-to-head statistics between two teams"""
        # This is a simplified version - you could expand this with actual H2H history
        features = {
            'h2h_matches': 0,  # Placeholder
            'h2h_team_a_wins': 0,  # Placeholder
        }
        return features
    
    def _update_team_history(self, match: Dict, team_a_code: str, team_b_code: str):
        """Update team history after processing a match"""
        team_a = match['teams']['team_a']
        team_b = match['teams']['team_b']
        
        # Update Team A
        self.team_history[team_a_code]['matches_played'] += 1
        self.team_history[team_a_code]['sets_won'] += team_a['sets_won']
        self.team_history[team_a_code]['sets_lost'] += team_b['sets_won']
        self.team_history[team_a_code]['total_points_scored'] += sum(team_a['set_scores'])
        self.team_history[team_a_code]['total_points_conceded'] += sum(team_b['set_scores'])
        
        if match['winner'] == team_a_code:
            self.team_history[team_a_code]['matches_won'] += 1
        
        # Update Team B
        self.team_history[team_b_code]['matches_played'] += 1
        self.team_history[team_b_code]['sets_won'] += team_b['sets_won']
        self.team_history[team_b_code]['sets_lost'] += team_a['sets_won']
        self.team_history[team_b_code]['total_points_scored'] += sum(team_b['set_scores'])
        self.team_history[team_b_code]['total_points_conceded'] += sum(team_a['set_scores'])
        
        if match['winner'] == team_b_code:
            self.team_history[team_b_code]['matches_won'] += 1
    
    def get_feature_importance_ready_data(self) -> tuple:
        """Get features (X) and target (y) ready for ML"""
        df = self.create_match_features()
        
        # Separate features and target
        exclude_cols = ['match_file', 'tournament', 'team_a', 'team_b', 'date', 'team_a_won']
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        X = df[feature_cols]
        y = df['team_a_won']
        
        # Store metadata
        metadata = df[['match_file', 'tournament', 'team_a', 'team_b', 'date']]
        
        return X, y, metadata, feature_cols


def main():
    """Example usage"""
    # Load parsed match data
    try:
        with open('volleyball_matches.json', 'r') as f:
            matches_data = json.load(f)
    except FileNotFoundError:
        print("Error: volleyball_matches.json not found.")
        print("Please run parse_volleyball_data.py first.")
        return
    
    print(f"Loaded {len(matches_data)} matches\n")
    
    # Create features
    engineer = VolleyballFeatureEngineer(matches_data)
    X, y, metadata, feature_cols = engineer.get_feature_importance_ready_data()
    
    print(f"{'='*60}")
    print("FEATURE ENGINEERING COMPLETE")
    print(f"{'='*60}")
    print(f"Total samples: {len(X)}")
    print(f"Total features: {len(feature_cols)}")
    print(f"Target distribution: {y.value_counts().to_dict()}")
    print(f"\nFeature columns ({len(feature_cols)}):")
    for i, col in enumerate(feature_cols[:20], 1):
        print(f"  {i}. {col}")
    if len(feature_cols) > 20:
        print(f"  ... and {len(feature_cols) - 20} more")
    
    # Save to CSV for easy inspection
    full_df = pd.concat([metadata, X, y], axis=1)
    full_df.to_csv('volleyball_features.csv', index=False)
    print(f"\n✓ Features saved to: volleyball_features.csv")
    
    # Save just features and target for ML
    X.to_csv('X_features.csv', index=False)
    y.to_csv('y_target.csv', index=False)
    print(f"✓ ML-ready data saved to: X_features.csv, y_target.csv")


if __name__ == '__main__':
    main()
