#!/usr/bin/env python3
"""
Data Validation Tool
Review and verify the cleaned volleyball data
"""

import json
import pandas as pd
import sqlite3
from collections import defaultdict

def validate_json_data():
    """Validate the parsed JSON data"""
    print("=" * 70)
    print("1. VALIDATING JSON DATA (volleyball_matches.json)")
    print("=" * 70)
    
    with open('volleyball_matches.json', 'r') as f:
        matches = json.load(f)
    
    print(f"✓ Total matches: {len(matches)}")
    print()
    
    # Check for required fields
    issues = []
    tournaments = set()
    teams = set()
    
    for i, match in enumerate(matches):
        # Required fields
        if 'tournament' not in match:
            issues.append(f"Match {i}: Missing 'tournament' field")
        else:
            tournaments.add(match['tournament']['code'])
        
        if 'teams' not in match:
            issues.append(f"Match {i}: Missing 'teams' field")
        else:
            team_a = match['teams']['team_a']['code']
            team_b = match['teams']['team_b']['code']
            teams.add(team_a)
            teams.add(team_b)
            
            # Validate sets won
            sets_a = match['teams']['team_a']['sets_won']
            sets_b = match['teams']['team_b']['sets_won']
            
            if sets_a == sets_b:
                issues.append(f"Match {i} ({team_a} vs {team_b}): Tie game - sets {sets_a}-{sets_b}")
            
            # Validate score matches sets
            scores_a = match['teams']['team_a']['set_scores']
            scores_b = match['teams']['team_b']['set_scores']
            
            if len(scores_a) != len(scores_b):
                issues.append(f"Match {i}: Mismatched set score lengths")
    
    print(f"✓ Tournaments found: {len(tournaments)}")
    print(f"  {sorted(tournaments)}")
    print()
    print(f"✓ Teams found: {len(teams)}")
    print(f"  {sorted(teams)}")
    print()
    
    if issues:
        print(f"⚠️  Found {len(issues)} potential issues:")
        for issue in issues[:10]:  # Show first 10
            print(f"  - {issue}")
        if len(issues) > 10:
            print(f"  ... and {len(issues) - 10} more")
    else:
        print("✓ No structural issues found!")
    
    print()
    return matches

def show_sample_matches(matches):
    """Show sample matches for manual review"""
    print("=" * 70)
    print("2. SAMPLE MATCHES FOR REVIEW")
    print("=" * 70)
    
    # Show 3 random matches with key details
    import random
    samples = random.sample(matches, min(3, len(matches)))
    
    for match in samples:
        team_a = match['teams']['team_a']
        team_b = match['teams']['team_b']
        
        print(f"\n📋 {match['file_name']}")
        print(f"   Tournament: {match['tournament']['name']}")
        print(f"   Match: {team_a['code']} vs {team_b['code']}")
        print(f"   Sets: {team_a['sets_won']} - {team_b['sets_won']}")
        print(f"   Scores: {team_a['set_scores']} vs {team_b['set_scores']}")
        
        # Show key stats
        stats_a = team_a['statistics']['total_stats']
        stats_b = team_b['statistics']['total_stats']
        
        print(f"   {team_a['code']} Stats: {stats_a.get('AtkPoint', 0)} attacks, {stats_a.get('BlkPoint', 0)} blocks, {stats_a.get('SrvPoint', 0)} serves")
        print(f"   {team_b['code']} Stats: {stats_b.get('AtkPoint', 0)} attacks, {stats_b.get('BlkPoint', 0)} blocks, {stats_b.get('SrvPoint', 0)} serves")
    
    print()

def validate_database():
    """Validate the SQLite database"""
    print("=" * 70)
    print("3. VALIDATING DATABASE (volleyball_data.db)")
    print("=" * 70)
    
    conn = sqlite3.connect('volleyball_data.db')
    cursor = conn.cursor()
    
    # Check tables
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = cursor.fetchall()
    print(f"✓ Tables: {[t[0] for t in tables]}")
    print()
    
    # Count records
    for table_name in ['tournaments', 'teams', 'matches', 'players']:
        cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
        count = cursor.fetchone()[0]
        print(f"  {table_name}: {count} records")
    
    print()
    
    # Sample match from database
    cursor.execute("""
        SELECT m.match_code, t1.team_code, t2.team_code, m.team_a_sets_won, m.team_b_sets_won
        FROM matches m
        JOIN teams t1 ON m.team_a_id = t1.team_id
        JOIN teams t2 ON m.team_b_id = t2.team_id
        LIMIT 5
    """)
    
    print("Sample matches from database:")
    for row in cursor.fetchall():
        print(f"  {row[0]}: {row[1]} ({row[3]}) vs {row[2]} ({row[4]})")
    
    conn.close()
    print()

def validate_features():
    """Validate the feature data"""
    print("=" * 70)
    print("4. VALIDATING FEATURES (volleyball_features.csv)")
    print("=" * 70)
    
    features = pd.read_csv('volleyball_features.csv')
    
    print(f"✓ Total samples: {len(features)}")
    print(f"✓ Total features: {len(features.columns)}")
    print()
    
    # Check for missing values
    missing = features.isnull().sum()
    missing_cols = missing[missing > 0]
    
    if len(missing_cols) > 0:
        print(f"⚠️  Columns with missing values:")
        for col, count in missing_cols.items():
            print(f"  - {col}: {count} missing ({count/len(features)*100:.1f}%)")
    else:
        print("✓ No missing values!")
    
    print()
    
    # Check target variable distribution
    print("Target variable (team_a_won) distribution:")
    print(features['team_a_won'].value_counts())
    print(f"  Balance: {features['team_a_won'].value_counts().min() / features['team_a_won'].value_counts().max() * 100:.1f}%")
    print()
    
    # Show feature statistics
    print("Sample feature statistics:")
    key_features = ['team_a_sets_won', 'team_b_sets_won', 'team_a_AtkPoint', 'team_b_AtkPoint']
    print(features[key_features].describe().round(2))
    print()

def check_tournament_specific(matches, tournament_code='TEST_PVLR25'):
    """Check specific tournament data"""
    print("=" * 70)
    print(f"5. TOURNAMENT-SPECIFIC CHECK: {tournament_code}")
    print("=" * 70)
    
    tournament_matches = [m for m in matches if m['tournament']['code'] == tournament_code]
    
    if not tournament_matches:
        print(f"⚠️  No matches found for tournament: {tournament_code}")
        return
    
    print(f"✓ Found {len(tournament_matches)} matches")
    print()
    
    # Team participation
    team_stats = defaultdict(lambda: {'wins': 0, 'losses': 0, 'matches': []})
    
    for match in tournament_matches:
        team_a = match['teams']['team_a']
        team_b = match['teams']['team_b']
        
        team_a_code = team_a['code']
        team_b_code = team_b['code']
        
        team_stats[team_a_code]['matches'].append(match['file_name'])
        team_stats[team_b_code]['matches'].append(match['file_name'])
        
        if team_a['sets_won'] > team_b['sets_won']:
            team_stats[team_a_code]['wins'] += 1
            team_stats[team_b_code]['losses'] += 1
        else:
            team_stats[team_b_code]['wins'] += 1
            team_stats[team_a_code]['losses'] += 1
    
    print("Team records in tournament:")
    for team in sorted(team_stats.keys()):
        stats = team_stats[team]
        total = stats['wins'] + stats['losses']
        print(f"  {team}: {stats['wins']}-{stats['losses']} ({total} matches)")
    
    print()
    print("All matches in tournament:")
    for i, match in enumerate(tournament_matches, 1):
        team_a = match['teams']['team_a']
        team_b = match['teams']['team_b']
        print(f"  {i}. {team_a['code']} ({team_a['sets_won']}) vs {team_b['code']} ({team_b['sets_won']}) - {match['file_name']}")
    
    print()

def main():
    print("\n" + "=" * 70)
    print("  VOLLEYBALL DATA VALIDATION REPORT")
    print("=" * 70)
    print()
    
    # Run all validations
    matches = validate_json_data()
    show_sample_matches(matches)
    validate_database()
    validate_features()
    check_tournament_specific(matches)
    
    print("=" * 70)
    print("VALIDATION COMPLETE!")
    print("=" * 70)
    print()
    print("📁 Files to review manually:")
    print("  1. volleyball_matches.json - Raw parsed data")
    print("  2. volleyball_features.csv - ML feature matrix")
    print("  3. volleyball_data.db - SQLite database (use DB Browser)")
    print()
    print("💡 To inspect specific matches, use:")
    print("  - Open volleyball_matches.json in a text editor")
    print("  - Use 'DB Browser for SQLite' to browse the database")
    print("  - Open volleyball_features.csv in Excel/Numbers")
    print()

if __name__ == '__main__':
    main()
