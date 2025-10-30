#!/usr/bin/env python3
"""
Predict Tournament Winner
Analyzes team performance and predicts the tournament champion
"""

import json
import joblib
import pandas as pd
import numpy as np
from collections import defaultdict

def load_tournament_data(tournament_code):
    """Load matches for a specific tournament"""
    with open('volleyball_matches.json', 'r') as f:
        all_matches = json.load(f)
    
    tournament_matches = [m for m in all_matches if m['tournament']['code'] == tournament_code]
    return tournament_matches

def calculate_team_statistics(matches):
    """Calculate comprehensive team statistics from tournament matches"""
    team_stats = defaultdict(lambda: {
        'matches_played': 0,
        'matches_won': 0,
        'sets_won': 0,
        'sets_lost': 0,
        'total_points_scored': 0,
        'total_points_conceded': 0,
        'total_attacks': 0,
        'total_attack_points': 0,
        'total_attack_errors': 0,
        'total_blocks': 0,
        'total_serves': 0,
        'total_serve_points': 0,
        'total_serve_errors': 0,
        'opponents_beaten': []
    })
    
    for match in matches:
        team_a = match['teams']['team_a']['code']
        team_b = match['teams']['team_b']['code']
        
        sets_a = match['teams']['team_a']['sets_won']
        sets_b = match['teams']['team_b']['sets_won']
        
        # Team A stats
        team_stats[team_a]['matches_played'] += 1
        team_stats[team_a]['sets_won'] += sets_a
        team_stats[team_a]['sets_lost'] += sets_b
        
        if sets_a > sets_b:
            team_stats[team_a]['matches_won'] += 1
            team_stats[team_a]['opponents_beaten'].append(team_b)
        
        # Team B stats
        team_stats[team_b]['matches_played'] += 1
        team_stats[team_b]['sets_won'] += sets_b
        team_stats[team_b]['sets_lost'] += sets_a
        
        if sets_b > sets_a:
            team_stats[team_b]['matches_won'] += 1
            team_stats[team_b]['opponents_beaten'].append(team_a)
        
        # Points and detailed stats for Team A
        team_a_stats = match['teams']['team_a']['statistics']['total_stats']
        team_a_scores = match['teams']['team_a']['set_scores']
        team_b_scores = match['teams']['team_b']['set_scores']
        
        team_stats[team_a]['total_points_scored'] += sum(team_a_scores)
        team_stats[team_a]['total_points_conceded'] += sum(team_b_scores)
        team_stats[team_a]['total_attack_points'] += team_a_stats.get('AtkPoint', 0)
        team_stats[team_a]['total_attack_errors'] += team_a_stats.get('AtkFault', 0)
        team_stats[team_a]['total_attacks'] += team_a_stats.get('AtkCont', 0)
        team_stats[team_a]['total_blocks'] += team_a_stats.get('BlkPoint', 0)
        team_stats[team_a]['total_serves'] += team_a_stats.get('SrvCont', 0)
        team_stats[team_a]['total_serve_points'] += team_a_stats.get('SrvPoint', 0)
        team_stats[team_a]['total_serve_errors'] += team_a_stats.get('SrvFault', 0)
        
        # Points and detailed stats for Team B
        team_b_stats = match['teams']['team_b']['statistics']['total_stats']
        
        team_stats[team_b]['total_points_scored'] += sum(team_b_scores)
        team_stats[team_b]['total_points_conceded'] += sum(team_a_scores)
        team_stats[team_b]['total_attack_points'] += team_b_stats.get('AtkPoint', 0)
        team_stats[team_b]['total_attack_errors'] += team_b_stats.get('AtkFault', 0)
        team_stats[team_b]['total_attacks'] += team_b_stats.get('AtkCont', 0)
        team_stats[team_b]['total_blocks'] += team_b_stats.get('BlkPoint', 0)
        team_stats[team_b]['total_serves'] += team_b_stats.get('SrvCont', 0)
        team_stats[team_b]['total_serve_points'] += team_b_stats.get('SrvPoint', 0)
        team_stats[team_b]['total_serve_errors'] += team_b_stats.get('SrvFault', 0)
    
    return dict(team_stats)

def calculate_team_metrics(team_stats):
    """Calculate derived metrics for ranking teams"""
    metrics = []
    
    for team, stats in team_stats.items():
        if stats['matches_played'] == 0:
            continue
        
        win_rate = stats['matches_won'] / stats['matches_played']
        set_win_rate = stats['sets_won'] / (stats['sets_won'] + stats['sets_lost']) if (stats['sets_won'] + stats['sets_lost']) > 0 else 0
        avg_points_per_match = stats['total_points_scored'] / stats['matches_played']
        point_differential = stats['total_points_scored'] - stats['total_points_conceded']
        
        attack_efficiency = ((stats['total_attack_points'] - stats['total_attack_errors']) / stats['total_attacks'] * 100) if stats['total_attacks'] > 0 else 0
        serve_efficiency = (stats['total_serve_points'] / stats['total_serves'] * 100) if stats['total_serves'] > 0 else 0
        
        metrics.append({
            'team': team,
            'matches_played': stats['matches_played'],
            'matches_won': stats['matches_won'],
            'matches_lost': stats['matches_played'] - stats['matches_won'],
            'win_rate': win_rate,
            'sets_won': stats['sets_won'],
            'sets_lost': stats['sets_lost'],
            'set_win_rate': set_win_rate,
            'total_points': stats['total_points_scored'],
            'avg_points_per_match': avg_points_per_match,
            'point_differential': point_differential,
            'attack_efficiency': attack_efficiency,
            'serve_efficiency': serve_efficiency,
            'blocks': stats['total_blocks'],
            'serve_points': stats['total_serve_points'],
            'opponents_beaten': len(set(stats['opponents_beaten']))
        })
    
    return sorted(metrics, key=lambda x: (x['win_rate'], x['set_win_rate'], x['point_differential']), reverse=True)

def main():
    print("=" * 60)
    print("  PVL REINFORCED CONFERENCE 2025 - TOURNAMENT ANALYSIS")
    print("=" * 60)
    print()
    
    # Load tournament data
    tournament_code = 'TEST_PVLR25'
    matches = load_tournament_data(tournament_code)
    
    if not matches:
        print(f"❌ No matches found for tournament: {tournament_code}")
        print("\nAvailable tournaments:")
        with open('volleyball_matches.json', 'r') as f:
            all_matches = json.load(f)
        tournaments = set(m['tournament']['code'] for m in all_matches)
        for t in sorted(tournaments):
            t_matches = [m for m in all_matches if m['tournament']['code'] == t]
            t_name = t_matches[0]['tournament']['name']
            print(f"  - {t} ({t_name}): {len(t_matches)} matches")
        return
    
    print(f"✓ Loaded {len(matches)} matches from {matches[0]['tournament']['name']}")
    print()
    
    # Calculate team statistics
    team_stats = calculate_team_statistics(matches)
    team_metrics = calculate_team_metrics(team_stats)
    
    # Display standings
    print("=" * 60)
    print("  TOURNAMENT STANDINGS & PREDICTIONS")
    print("=" * 60)
    print()
    print(f"{'Rank':<6}{'Team':<10}{'W-L':<10}{'Win%':<10}{'Sets':<12}{'Pts Diff':<12}{'Atk%':<8}")
    print("-" * 60)
    
    for i, team in enumerate(team_metrics, 1):
        rank_symbol = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else f"{i}."
        print(f"{rank_symbol:<6}{team['team']:<10}{team['matches_won']}-{team['matches_lost']:<8}"
              f"{team['win_rate']*100:>6.1f}%   {team['sets_won']}-{team['sets_lost']:<9}"
              f"{team['point_differential']:>+6.0f}      {team['attack_efficiency']:>5.1f}%")
    
    print()
    print("=" * 60)
    print("  CHAMPION PREDICTION")
    print("=" * 60)
    print()
    
    if team_metrics:
        champion = team_metrics[0]
        runner_up = team_metrics[1] if len(team_metrics) > 1 else None
        
        print(f"🏆 PREDICTED CHAMPION: {champion['team']}")
        print(f"   Win Rate: {champion['win_rate']*100:.1f}%")
        print(f"   Match Record: {champion['matches_won']}-{champion['matches_lost']}")
        print(f"   Set Record: {champion['sets_won']}-{champion['sets_lost']}")
        print(f"   Average Points/Match: {champion['avg_points_per_match']:.1f}")
        print(f"   Attack Efficiency: {champion['attack_efficiency']:.1f}%")
        print(f"   Point Differential: {champion['point_differential']:+.0f}")
        
        if runner_up:
            print()
            print(f"🥈 Runner-up: {runner_up['team']}")
            print(f"   Win Rate: {runner_up['win_rate']*100:.1f}%")
            print(f"   Match Record: {runner_up['matches_won']}-{runner_up['matches_lost']}")
    
    print()
    print("=" * 60)
    print("  DETAILED TEAM ANALYSIS")
    print("=" * 60)
    print()
    
    for team in team_metrics[:5]:  # Top 5 teams
        print(f"📊 {team['team']}:")
        print(f"   Matches: {team['matches_won']}-{team['matches_lost']} ({team['win_rate']*100:.1f}% win rate)")
        print(f"   Sets: {team['sets_won']}-{team['sets_lost']} ({team['set_win_rate']*100:.1f}% set win rate)")
        print(f"   Total Points: {team['total_points']:.0f} (avg {team['avg_points_per_match']:.1f}/match)")
        print(f"   Point Differential: {team['point_differential']:+.0f}")
        print(f"   Attack Efficiency: {team['attack_efficiency']:.1f}%")
        print(f"   Serve Efficiency: {team['serve_efficiency']:.1f}%")
        print(f"   Total Blocks: {team['blocks']}")
        print(f"   Different opponents beaten: {team['opponents_beaten']}")
        print()

if __name__ == '__main__':
    main()
