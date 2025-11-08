"""
Database Storage for Volleyball Match Data
SQLite database schema and operations
"""

import sqlite3
import json
from typing import List, Dict, Any
try:
    from .config import canonicalize_team_code
except Exception:
    from config import canonicalize_team_code
from pathlib import Path


class VolleyballDatabase:
    """Manage volleyball match data in SQLite database"""
    
    def __init__(self, db_path: str = 'volleyball_data.db'):
        self.db_path = db_path
        self.conn = None
        
    def connect(self):
        """Connect to database"""
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row
        return self.conn
    
    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()
    
    def create_schema(self):
        """Create database schema"""
        cursor = self.conn.cursor()
        
        # Tournaments table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS tournaments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                code TEXT UNIQUE NOT NULL,
                name TEXT
            )
        ''')
        
        # Teams table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS teams (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                code TEXT UNIQUE NOT NULL,
                name TEXT,
                coach TEXT,
                assistant_coach TEXT
            )
        ''')
        
        # Matches table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS matches (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                file_name TEXT UNIQUE,
                tournament_id INTEGER,
                match_no TEXT,
                date TEXT,
                time TEXT,
                city TEXT,
                hall TEXT,
                team_a_id INTEGER,
                team_b_id INTEGER,
                team_a_sets_won INTEGER,
                team_b_sets_won INTEGER,
                winner_id INTEGER,
                status TEXT,
                FOREIGN KEY (tournament_id) REFERENCES tournaments(id),
                FOREIGN KEY (team_a_id) REFERENCES teams(id),
                FOREIGN KEY (team_b_id) REFERENCES teams(id),
                FOREIGN KEY (winner_id) REFERENCES teams(id)
            )
        ''')
        
        # Team match statistics table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS team_match_stats (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                match_id INTEGER,
                team_id INTEGER,
                is_team_a BOOLEAN,
                total_points INTEGER,
                sets_won INTEGER,
                attack_points INTEGER,
                attack_faults INTEGER,
                attack_continues INTEGER,
                block_points INTEGER,
                block_faults INTEGER,
                serve_points INTEGER,
                serve_faults INTEGER,
                reception_excellent INTEGER,
                reception_faults INTEGER,
                dig_excellent INTEGER,
                dig_faults INTEGER,
                set_excellent INTEGER,
                set_faults INTEGER,
                opponent_errors INTEGER,
                stats_json TEXT,
                FOREIGN KEY (match_id) REFERENCES matches(id),
                FOREIGN KEY (team_id) REFERENCES teams(id)
            )
        ''')
        
        # Set scores table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS set_scores (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                match_id INTEGER,
                set_number INTEGER,
                team_a_score INTEGER,
                team_b_score INTEGER,
                FOREIGN KEY (match_id) REFERENCES matches(id)
            )
        ''')
        
        # Players table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS players (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                first_name TEXT,
                last_name TEXT,
                full_name TEXT UNIQUE
            )
        ''')
        
        # Player match statistics table (aggregated across all sets)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS player_match_stats (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                match_id INTEGER,
                team_id INTEGER,
                player_id INTEGER,
                jersey_number INTEGER,
                is_starter BOOLEAN,
                is_libero BOOLEAN,
                sets_played INTEGER,
                attack_points INTEGER DEFAULT 0,
                attack_faults INTEGER DEFAULT 0,
                attack_continues INTEGER DEFAULT 0,
                back_attack_points INTEGER DEFAULT 0,
                back_attack_faults INTEGER DEFAULT 0,
                back_attack_continues INTEGER DEFAULT 0,
                block_points INTEGER DEFAULT 0,
                block_faults INTEGER DEFAULT 0,
                block_continues INTEGER DEFAULT 0,
                serve_points INTEGER DEFAULT 0,
                serve_faults INTEGER DEFAULT 0,
                serve_continues INTEGER DEFAULT 0,
                reception_excellent INTEGER DEFAULT 0,
                reception_faults INTEGER DEFAULT 0,
                reception_continues INTEGER DEFAULT 0,
                dig_excellent INTEGER DEFAULT 0,
                dig_faults INTEGER DEFAULT 0,
                dig_continues INTEGER DEFAULT 0,
                set_excellent INTEGER DEFAULT 0,
                set_faults INTEGER DEFAULT 0,
                set_continues INTEGER DEFAULT 0,
                FOREIGN KEY (match_id) REFERENCES matches(id),
                FOREIGN KEY (team_id) REFERENCES teams(id),
                FOREIGN KEY (player_id) REFERENCES players(id)
            )
        ''')
        
        # Match lineups table (starting roster for each set)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS match_lineups (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                match_id INTEGER,
                team_id INTEGER,
                set_number INTEGER,
                position INTEGER,
                player_jersey INTEGER,
                role TEXT,
                FOREIGN KEY (match_id) REFERENCES matches(id),
                FOREIGN KEY (team_id) REFERENCES teams(id)
            )
        ''')
        
        self.conn.commit()
        print("✓ Database schema created")
    
    def insert_tournament(self, code: str, name: str) -> int:
        """Insert or get tournament"""
        cursor = self.conn.cursor()
        cursor.execute('SELECT id FROM tournaments WHERE code = ?', (code,))
        row = cursor.fetchone()
        
        if row:
            return row[0]
        
        cursor.execute('INSERT INTO tournaments (code, name) VALUES (?, ?)', (code, name))
        self.conn.commit()
        return cursor.lastrowid
    
    def insert_team(self, code: str, name: str, coach: str = None, assistant_coach: str = None) -> int:
        """Insert or get team"""
        # Canonicalize code before any lookup/insert to unify aliases (e.g., CHD -> CSS)
        code = canonicalize_team_code(code)
        cursor = self.conn.cursor()
        cursor.execute('SELECT id FROM teams WHERE code = ?', (code,))
        row = cursor.fetchone()
        
        if row:
            return row[0]
        
        cursor.execute(
            'INSERT INTO teams (code, name, coach, assistant_coach) VALUES (?, ?, ?, ?)',
            (code, name, coach, assistant_coach)
        )
        self.conn.commit()
        return cursor.lastrowid
    
    def insert_player(self, first_name: str, last_name: str, full_name: str) -> int:
        """Insert or get player"""
        cursor = self.conn.cursor()
        cursor.execute('SELECT id FROM players WHERE full_name = ?', (full_name,))
        row = cursor.fetchone()
        
        if row:
            return row[0]
        
        cursor.execute(
            'INSERT INTO players (first_name, last_name, full_name) VALUES (?, ?, ?)',
            (first_name, last_name, full_name)
        )
        self.conn.commit()
        return cursor.lastrowid
    
    def insert_match(self, match_data: Dict[str, Any]) -> int:
        """Insert complete match data"""
        cursor = self.conn.cursor()
        
        # Check if match already exists
        cursor.execute('SELECT id FROM matches WHERE file_name = ?', (match_data['file_name'],))
        if cursor.fetchone():
            print(f"  Match {match_data['file_name']} already exists, skipping")
            return None
        
        # Insert tournament
        tournament_id = None
        if match_data['tournament'].get('code'):
            tournament_id = self.insert_tournament(
                match_data['tournament']['code'],
                match_data['tournament'].get('name')
            )
        
        # Insert teams
        team_ids = {}
        for team_key, team_info in match_data['team_rosters'].items():
            # Ensure canonical code in roster info
            canon_code = canonicalize_team_code(team_info['code'])
            team_info = dict(team_info)
            team_info['code'] = canon_code
            team_ids[team_key] = self.insert_team(
                team_info['code'],
                team_info['name'],
                team_info.get('coach'),
                team_info.get('assistant_coach')
            )
        
        # Insert players
        for player in match_data['players']:
            self.insert_player(
                player['first_name'],
                player['last_name'],
                player['full_name']
            )
        # Build code->id mapping for canonical codes as well
        # Existing team_ids currently keyed by original roster loop key (team_code). Extend with canonical code.
        for orig_code, tid in list(team_ids.items()):
            canon_code = canonicalize_team_code(orig_code)
            team_ids[canon_code] = tid

        # Get canonical team codes
        team_a_code = canonicalize_team_code(match_data['teams']['team_a']['code'])
        team_b_code = canonicalize_team_code(match_data['teams']['team_b']['code'])
        team_a_id = team_ids.get(team_a_code)
        team_b_id = team_ids.get(team_b_code)

        # Determine winner ID using canonical code
        winner_code = canonicalize_team_code(match_data['winner']) if match_data['winner'] else None
        winner_id = team_ids.get(winner_code) if winner_code else None

        # Insert match row
        cursor.execute('''
            INSERT INTO matches (
                file_name, tournament_id, match_no, date, time, city, hall,
                team_a_id, team_b_id, team_a_sets_won, team_b_sets_won, winner_id, status
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            match_data['file_name'],
            tournament_id,
            match_data['match_info'].get('match_no'),
            match_data['match_info'].get('date'),
            match_data['match_info'].get('time'),
            match_data['match_info'].get('city'),
            match_data['match_info'].get('hall'),
            team_a_id,
            team_b_id,
            match_data['teams']['team_a']['sets_won'],
            match_data['teams']['team_b']['sets_won'],
            winner_id,
            match_data['match_info'].get('status')
        ))
        match_id = cursor.lastrowid
        
        # Insert team statistics
        for team_key, is_team_a in [('team_a', True), ('team_b', False)]:
            team_data = match_data['teams'][team_key]
            # Canonical stats assignment
            team_code_canon = canonicalize_team_code(team_data['code'])
            team_id = team_a_id if is_team_a else team_b_id
            stats = team_data['statistics']['total_stats']
            
            cursor.execute('''
                INSERT INTO team_match_stats (
                    match_id, team_id, is_team_a, total_points, sets_won,
                    attack_points, attack_faults, attack_continues,
                    block_points, block_faults,
                    serve_points, serve_faults,
                    reception_excellent, reception_faults,
                    dig_excellent, dig_faults,
                    set_excellent, set_faults,
                    opponent_errors, stats_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                match_id, team_id, is_team_a,
                sum(team_data['set_scores']),
                team_data['sets_won'],
                stats.get('AtkPoint', 0), stats.get('AtkFault', 0), stats.get('AtkCont', 0),
                stats.get('BlkPoint', 0), stats.get('BlkFault', 0),
                stats.get('SrvPoint', 0), stats.get('SrvFault', 0),
                stats.get('RecExcel', 0), stats.get('RecFault', 0),
                stats.get('DigExcel', 0), stats.get('DigFault', 0),
                stats.get('SetExcel', 0), stats.get('SetFault', 0),
                stats.get('team_OppError', 0),
                json.dumps(stats)
            ))
        
        # Insert set scores
        team_a_scores = match_data['teams']['team_a']['set_scores']
        team_b_scores = match_data['teams']['team_b']['set_scores']
        max_sets = max(len(team_a_scores), len(team_b_scores))
        
        for i in range(max_sets):
            cursor.execute('''
                INSERT INTO set_scores (match_id, set_number, team_a_score, team_b_score)
                VALUES (?, ?, ?, ?)
            ''', (
                match_id,
                i + 1,
                team_a_scores[i] if i < len(team_a_scores) else 0,
                team_b_scores[i] if i < len(team_b_scores) else 0
            ))
        
        # Insert player statistics
        for team_key, is_team_a in [('team_a', True), ('team_b', False)]:
            team_data = match_data['teams'][team_key]
            team_id = team_a_id if is_team_a else team_b_id
            
            # Insert player match stats
            if 'player_stats' in team_data:
                for player_stat in team_data['player_stats']:
                    # No need to look up player_id for now - we'll add that later if needed
                    self.insert_player_stats(match_id, team_id, player_stat)
            
            # Insert lineups
            if 'lineups' in team_data:
                for set_lineup in team_data['lineups']:
                    set_number = set_lineup['set_number']
                    
                    # Insert all roster positions
                    for player in set_lineup['starters'] + set_lineup['substitutes'] + set_lineup['liberos']:
                        self.insert_lineup(match_id, team_id, set_number, [player])
        
        self.conn.commit()
        return match_id
    
    def insert_player_stats(self, match_id: int, team_id: int, player_stats: Dict[str, Any]):
        """Insert player match statistics"""
        cursor = self.conn.cursor()
        
        cursor.execute('''
            INSERT INTO player_match_stats (
                match_id, team_id, player_id, jersey_number,
                is_starter, is_libero, sets_played,
                attack_points, attack_faults, attack_continues,
                back_attack_points, back_attack_faults, back_attack_continues,
                block_points, block_faults, block_continues,
                serve_points, serve_faults, serve_continues,
                reception_excellent, reception_faults, reception_continues,
                dig_excellent, dig_faults, dig_continues,
                set_excellent, set_faults, set_continues
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            match_id, team_id,
            player_stats.get('player_id'),
            player_stats.get('jersey_number'),
            player_stats.get('is_starter', False),
            player_stats.get('is_libero', False),
            player_stats.get('sets_played', 0),
            player_stats.get('attack_points', 0),
            player_stats.get('attack_faults', 0),
            player_stats.get('attack_continues', 0),
            player_stats.get('back_attack_points', 0),
            player_stats.get('back_attack_faults', 0),
            player_stats.get('back_attack_continues', 0),
            player_stats.get('block_points', 0),
            player_stats.get('block_faults', 0),
            player_stats.get('block_continues', 0),
            player_stats.get('serve_points', 0),
            player_stats.get('serve_faults', 0),
            player_stats.get('serve_continues', 0),
            player_stats.get('reception_excellent', 0),
            player_stats.get('reception_faults', 0),
            player_stats.get('reception_continues', 0),
            player_stats.get('dig_excellent', 0),
            player_stats.get('dig_faults', 0),
            player_stats.get('dig_continues', 0),
            player_stats.get('set_excellent', 0),
            player_stats.get('set_faults', 0),
            player_stats.get('set_continues', 0)
        ))
        
        self.conn.commit()
    
    def insert_lineup(self, match_id: int, team_id: int, set_number: int, lineup_data: List[Dict[str, Any]]):
        """Insert lineup data for a set"""
        cursor = self.conn.cursor()
        
        for player in lineup_data:
            cursor.execute('''
                INSERT INTO match_lineups (
                    match_id, team_id, set_number, position, player_jersey, role
                ) VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                match_id, team_id, set_number,
                player.get('position'),
                player.get('jersey'),
                player.get('role')  # 'starter', 'substitute', 'libero'
            ))
        
        self.conn.commit()
    
    def load_from_json(self, json_file: str):
        """Load matches from JSON file into database"""
        with open(json_file, 'r') as f:
            matches_data = json.load(f)
        
        print(f"Loading {len(matches_data)} matches into database...")
        
        for i, match in enumerate(matches_data, 1):
            try:
                self.insert_match(match)
                print(f"  ✓ [{i}/{len(matches_data)}] Inserted: {match['file_name']}")
            except Exception as e:
                print(f"  ✗ [{i}/{len(matches_data)}] Error: {match['file_name']} - {str(e)}")
        
        print("\n✓ Database loading complete")
    
    def get_summary(self) -> Dict[str, Any]:
        """Get database summary statistics"""
        cursor = self.conn.cursor()
        
        summary = {}
        
        cursor.execute('SELECT COUNT(*) FROM tournaments')
        summary['total_tournaments'] = cursor.fetchone()[0]
        
        cursor.execute('SELECT COUNT(*) FROM teams')
        summary['total_teams'] = cursor.fetchone()[0]
        
        cursor.execute('SELECT COUNT(*) FROM matches')
        summary['total_matches'] = cursor.fetchone()[0]
        
        cursor.execute('SELECT COUNT(*) FROM players')
        summary['total_players'] = cursor.fetchone()[0]
        
        return summary


def main():
    """Example usage"""
    db = VolleyballDatabase()
    db.connect()
    
    # Create schema
    db.create_schema()
    
    # Load data from JSON
    try:
        db.load_from_json('volleyball_matches.json')
    except FileNotFoundError:
        print("Error: volleyball_matches.json not found.")
        print("Please run parse_volleyball_data.py first.")
        db.close()
        return
    
    # Print summary
    summary = db.get_summary()
    print(f"\n{'='*50}")
    print("DATABASE SUMMARY")
    print(f"{'='*50}")
    print(f"Tournaments: {summary['total_tournaments']}")
    print(f"Teams: {summary['total_teams']}")
    print(f"Matches: {summary['total_matches']}")
    print(f"Players: {summary['total_players']}")
    print(f"\nDatabase saved to: volleyball_data.db")
    
    db.close()


if __name__ == '__main__':
    main()
