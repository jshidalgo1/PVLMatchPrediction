#!/usr/bin/env python3
"""
Player Statistics Analysis Example

This script demonstrates how to extract and analyze player statistics
from the database.
"""

import sys
from pathlib import Path
import sqlite3
import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def analyze_player_stats():
    """Analyze player performance statistics."""
    
    print("=" * 70)
    print("PLAYER STATISTICS ANALYSIS EXAMPLE")
    print("=" * 70)
    print()
    
    # Connect to database
    db_path = project_root / "data" / "databases" / "volleyball_data.db"
    
    if not db_path.exists():
        print(f"❌ Database not found: {db_path}")
        print()
        print("Please process match data first:")
        print("  python scripts/batch_processor.py")
        return
    
    conn = sqlite3.connect(db_path)
    
    # Query: Top 10 scorers
    print("🏆 TOP 10 SCORERS (All Matches)")
    print("-" * 70)
    
    query_top_scorers = """
    SELECT 
        player_name,
        team_code,
        COUNT(DISTINCT match_id) as matches_played,
        SUM(attack_points) as total_attack,
        SUM(block_points) as total_blocks,
        SUM(serve_points) as total_serves,
        SUM(total_points) as total_points,
        ROUND(AVG(total_points), 2) as avg_points_per_match
    FROM player_match_stats
    GROUP BY player_name, team_code
    HAVING matches_played >= 3
    ORDER BY total_points DESC
    LIMIT 10
    """
    
    top_scorers = pd.read_sql_query(query_top_scorers, conn)
    print(top_scorers.to_string(index=False))
    print()
    
    # Query: Top blockers
    print("🛡️  TOP 10 BLOCKERS")
    print("-" * 70)
    
    query_top_blockers = """
    SELECT 
        player_name,
        team_code,
        COUNT(DISTINCT match_id) as matches_played,
        SUM(block_points) as total_blocks,
        ROUND(AVG(block_points), 2) as avg_blocks_per_match
    FROM player_match_stats
    GROUP BY player_name, team_code
    HAVING matches_played >= 3
    ORDER BY total_blocks DESC
    LIMIT 10
    """
    
    top_blockers = pd.read_sql_query(query_top_blockers, conn)
    print(top_blockers.to_string(index=False))
    print()
    
    # Query: Top servers
    print("⚡ TOP 10 SERVERS")
    print("-" * 70)
    
    query_top_servers = """
    SELECT 
        player_name,
        team_code,
        COUNT(DISTINCT match_id) as matches_played,
        SUM(serve_points) as total_serves,
        ROUND(AVG(serve_points), 2) as avg_serves_per_match
    FROM player_match_stats
    GROUP BY player_name, team_code
    HAVING matches_played >= 3
    ORDER BY total_serves DESC
    LIMIT 10
    """
    
    top_servers = pd.read_sql_query(query_top_servers, conn)
    print(top_servers.to_string(index=False))
    print()
    
    # Query: Team statistics
    print("📊 TEAM PERFORMANCE SUMMARY")
    print("-" * 70)
    
    query_team_stats = """
    SELECT 
        t.team_code,
        t.team_name,
        COUNT(DISTINCT m.match_id) as matches_played,
        SUM(CASE WHEN tms.outcome = 'W' THEN 1 ELSE 0 END) as wins,
        SUM(CASE WHEN tms.outcome = 'L' THEN 1 ELSE 0 END) as losses,
        ROUND(
            100.0 * SUM(CASE WHEN tms.outcome = 'W' THEN 1 ELSE 0 END) / 
            COUNT(DISTINCT m.match_id), 
            2
        ) as win_percentage,
        ROUND(AVG(tms.total_points), 2) as avg_points
    FROM teams t
    JOIN team_match_stats tms ON t.team_id = tms.team_id
    JOIN matches m ON tms.match_id = m.match_id
    GROUP BY t.team_code, t.team_name
    ORDER BY win_percentage DESC
    """
    
    team_stats = pd.read_sql_query(query_team_stats, conn)
    print(team_stats.to_string(index=False))
    print()
    
    conn.close()
    
    print("=" * 70)
    print("ANALYSIS COMPLETE!")
    print("=" * 70)
    print()
    print("💡 Tips:")
    print("  - Modify SQL queries to analyze different metrics")
    print("  - Use pandas for advanced data manipulation")
    print("  - Export results to CSV for further analysis")
    print()


if __name__ == "__main__":
    try:
        analyze_player_stats()
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)
