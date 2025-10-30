"""
Volleyball Match Data Parser
Parses PVL XML match files and extracts features for machine learning
"""

import xml.etree.ElementTree as ET
from typing import Dict, List, Any
import json
from pathlib import Path
from collections import defaultdict


class VolleyballDataParser:
    """Parser for volleyball match XML files"""
    
    def __init__(self):
        self.match_data = []
        
    def parse_xml_file(self, xml_file_path: str) -> Dict[str, Any]:
        """Parse a single XML file and extract match data"""
        tree = ET.parse(xml_file_path)
        root = tree.getroot()
        
        match_data = {
            'file_name': Path(xml_file_path).name,
            'tournament': {},
            'teams': {},
            'match_info': {},
            'players': [],
            'team_rosters': {},
            'set_results': [],
            'winner': None
        }
        
        # Parse tournament info
        tournament = root.find('Tournament')
        if tournament is not None:
            match_data['tournament'] = {
                'code': tournament.get('Code'),
                'name': tournament.find('Name').text if tournament.find('Name') is not None else None
            }
        
        # Parse all players
        for player in root.findall('Player'):
            first_name = player.find('FirstName')
            last_name = player.find('LastName')
            if first_name is not None and last_name is not None:
                match_data['players'].append({
                    'first_name': first_name.text,
                    'last_name': last_name.text,
                    'full_name': f"{first_name.text} {last_name.text}"
                })
        
        # Parse team information
        for team in root.findall('Team'):
            team_code = team.get('Code')
            team_info = {
                'code': team_code,
                'name': team.find('Name').text if team.find('Name') is not None else None,
                'coach': None,
                'assistant_coach': None,
                'player_numbers': []
            }
            
            coach = team.find('Coach/Name')
            if coach is not None:
                team_info['coach'] = coach.text
                
            assistant = team.find('AssistantCoach/Name')
            if assistant is not None:
                team_info['assistant_coach'] = assistant.text
            
            # Get player roster numbers
            for player in team.findall('Player'):
                shirt_no = player.get('NoShirt')
                if shirt_no:
                    team_info['player_numbers'].append(int(shirt_no))
            
            match_data['team_rosters'][team_code] = team_info
        
        # Parse match information
        match = root.find('Match')
        if match is not None:
            match_data['match_info'] = {
                'match_no': match.find('Match_No').text if match.find('Match_No') is not None else match.get('No'),
                'status': match.find('Status').text if match.find('Status') is not None else None,
                'phase': match.find('Phase').text if match.find('Phase') is not None else None,
                'date': match.find('Date').text if match.find('Date') is not None else None,
                'time': match.find('Time').text if match.find('Time') is not None else None,
                'city': match.find('City').text if match.find('City') is not None else None,
                'hall': match.find('Hall').text if match.find('Hall') is not None else None
            }
            
            # Parse team match data
            team_a = match.find('TeamA')
            team_b = match.find('TeamB')
            
            if team_a is not None and team_b is not None:
                team_a_code = team_a.get('Code')
                team_b_code = team_b.get('Code')
                
                # Extract set scores for both teams
                team_a_scores = self._extract_set_scores(team_a)
                team_b_scores = self._extract_set_scores(team_b)
                
                # Calculate sets won by comparing scores
                team_a_sets_won = 0
                team_b_sets_won = 0
                
                for score_a, score_b in zip(team_a_scores, team_b_scores):
                    if score_a > score_b:
                        team_a_sets_won += 1
                    elif score_b > score_a:
                        team_b_sets_won += 1
                
                # Determine winner (team that won more sets)
                if team_a_sets_won > team_b_sets_won:
                    match_data['winner'] = team_a_code
                elif team_b_sets_won > team_a_sets_won:
                    match_data['winner'] = team_b_code
                else:
                    # This should never happen in volleyball, but handle it
                    match_data['winner'] = None
                
                match_data['teams'] = {
                    'team_a': {
                        'code': team_a_code,
                        'sets_won': team_a_sets_won,
                        'set_scores': team_a_scores,
                        'statistics': self._extract_team_statistics(team_a),
                        'player_stats': self._extract_player_match_stats(team_a),
                        'lineups': self._extract_lineups(team_a)
                    },
                    'team_b': {
                        'code': team_b_code,
                        'sets_won': team_b_sets_won,
                        'set_scores': team_b_scores,
                        'statistics': self._extract_team_statistics(team_b),
                        'player_stats': self._extract_player_match_stats(team_b),
                        'lineups': self._extract_lineups(team_b)
                    }
                }
        
        return match_data
    
    def _extract_set_scores(self, team_elem) -> List[int]:
        """Extract scores for each set"""
        scores = []
        for set_elem in team_elem.findall('Set'):
            score_elem = set_elem.find('Score')
            if score_elem is not None and score_elem.text:
                scores.append(int(score_elem.text))
        return scores
    
    def _extract_team_statistics(self, team_elem) -> Dict[str, Any]:
        """Extract aggregated team statistics from all sets"""
        total_stats = defaultdict(int)
        player_stats = defaultdict(lambda: defaultdict(int))
        
        for set_elem in team_elem.findall('Set'):
            set_no = set_elem.get('No')
            if set_no and set_no != '5':  # Skip empty set 5 placeholders
                stats = set_elem.find('Statistics')
                if stats is not None:
                    # Team level stats
                    team_stats = stats.find('Team')
                    if team_stats is not None:
                        for attr, value in team_stats.attrib.items():
                            total_stats[f'team_{attr}'] += int(value)
                    
                    # Player level stats
                    for player in stats.findall('Player'):
                        shirt_no = player.get('NoShirt')
                        if shirt_no:
                            for attr, value in player.attrib.items():
                                if attr != 'NoShirt':
                                    player_stats[shirt_no][attr] += int(value)
                                    total_stats[attr] += int(value)
        
        return {
            'total_stats': dict(total_stats),
            'player_stats': dict(player_stats)
        }
    
    def _extract_player_match_stats(self, team_elem) -> List[Dict[str, Any]]:
        """Extract individual player statistics aggregated across all sets"""
        player_totals = defaultdict(lambda: defaultdict(int))
        player_sets_played = defaultdict(int)
        
        # Get libero numbers from team element
        liberos = set()
        for libero in team_elem.findall('Libero'):
            if libero.text:
                liberos.add(libero.text)
        
        # Get starting lineup from first set
        starters = set()
        first_set = team_elem.find('Set[@No="1"]')
        if first_set is not None:
            roster = first_set.find('Roster')
            if roster is not None:
                # p1-p6 are starters
                for i in range(1, 7):
                    starter = roster.get(f'p{i}')
                    if starter:
                        starters.add(starter)
        
        # Aggregate stats across all sets
        for set_elem in team_elem.findall('Set'):
            set_no = set_elem.get('No')
            if set_no and set_no != '5':  # Skip empty set 5 placeholders
                stats = set_elem.find('Statistics')
                if stats is not None:
                    for player in stats.findall('Player'):
                        shirt_no = player.get('NoShirt')
                        if shirt_no and len(player.attrib) > 1:  # Has stats beyond NoShirt
                            player_sets_played[shirt_no] += 1
                            
                            for attr, value in player.attrib.items():
                                if attr != 'NoShirt':
                                    player_totals[shirt_no][attr] += int(value)
        
        # Convert to list format
        player_list = []
        for jersey, stats in player_totals.items():
            player_list.append({
                'jersey_number': int(jersey),
                'is_starter': jersey in starters,
                'is_libero': jersey in liberos,
                'sets_played': player_sets_played[jersey],
                'attack_points': stats.get('AtkPoint', 0),
                'attack_faults': stats.get('AtkFault', 0),
                'attack_continues': stats.get('AtkCont', 0),
                'back_attack_points': stats.get('BAtkPoint', 0),
                'back_attack_faults': stats.get('BAtkFault', 0),
                'back_attack_continues': stats.get('BAtkCont', 0),
                'block_points': stats.get('BlkPoint', 0),
                'block_faults': stats.get('BlkFault', 0),
                'block_continues': stats.get('BlkCont', 0),
                'serve_points': stats.get('SrvPoint', 0),
                'serve_faults': stats.get('SrvFault', 0),
                'serve_continues': stats.get('SrvCont', 0),
                'reception_excellent': stats.get('RecExcel', 0),
                'reception_faults': stats.get('RecFault', 0),
                'reception_continues': stats.get('RecCont', 0),
                'dig_excellent': stats.get('DigExcel', 0),
                'dig_faults': stats.get('DigFault', 0),
                'dig_continues': stats.get('DigCont', 0),
                'set_excellent': stats.get('SetExcel', 0),
                'set_faults': stats.get('SetFault', 0),
                'set_continues': stats.get('SetCont', 0)
            })
        
        return player_list
    
    def _extract_lineups(self, team_elem) -> List[Dict[str, Any]]:
        """Extract lineup information for each set"""
        lineups = []
        
        for set_elem in team_elem.findall('Set'):
            set_no = set_elem.get('No')
            if set_no and set_no != '5':  # Skip empty set 5 placeholders
                roster = set_elem.find('Roster')
                if roster is not None:
                    set_lineup = {
                        'set_number': int(set_no),
                        'starters': [],
                        'substitutes': [],
                        'liberos': []
                    }
                    
                    # Parse starters (p1-p6)
                    for i in range(1, 7):
                        player = roster.get(f'p{i}')
                        if player:
                            set_lineup['starters'].append({
                                'position': i,
                                'jersey': int(player),
                                'role': 'starter'
                            })
                    
                    # Parse substitutes (r1-r6)
                    for i in range(1, 7):
                        player = roster.get(f'r{i}')
                        if player:
                            set_lineup['substitutes'].append({
                                'position': i,
                                'jersey': int(player),
                                'role': 'substitute'
                            })
                    
                    # Parse liberos (l1-l3)
                    for i in range(1, 4):
                        player = roster.get(f'l{i}')
                        if player:
                            set_lineup['liberos'].append({
                                'position': i,
                                'jersey': int(player),
                                'role': 'libero'
                            })
                    
                    lineups.append(set_lineup)
        
        return lineups
    
    def parse_multiple_files(self, file_paths: List[str]) -> List[Dict[str, Any]]:
        """Parse multiple XML files"""
        self.match_data = []
        for file_path in file_paths:
            try:
                data = self.parse_xml_file(file_path)
                self.match_data.append(data)
                print(f"✓ Parsed: {file_path}")
            except Exception as e:
                print(f"✗ Error parsing {file_path}: {str(e)}")
        
        return self.match_data
    
    def save_to_json(self, output_file: str):
        """Save parsed data to JSON file"""
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(self.match_data, f, indent=2, ensure_ascii=False)
        print(f"\n✓ Saved to JSON: {output_file}")
    
    def get_feature_summary(self) -> Dict[str, Any]:
        """Get summary of extracted features"""
        if not self.match_data:
            return {}
        
        summary = {
            'total_matches': len(self.match_data),
            'tournaments': set(),
            'teams': set(),
            'stat_types': set()
        }
        
        for match in self.match_data:
            if match['tournament'].get('code'):
                summary['tournaments'].add(match['tournament']['code'])
            
            if 'teams' in match:
                for team_key in ['team_a', 'team_b']:
                    if team_key in match['teams']:
                        summary['teams'].add(match['teams'][team_key]['code'])
                        
                        # Collect stat types
                        stats = match['teams'][team_key]['statistics']['total_stats']
                        summary['stat_types'].update(stats.keys())
        
        summary['tournaments'] = list(summary['tournaments'])
        summary['teams'] = list(summary['teams'])
        summary['stat_types'] = sorted(list(summary['stat_types']))
        
        return summary


def main():
    """Example usage"""
    import glob
    from config import XML_DIR_STR
    
    # Find all XML files in data/xml_files directory
    xml_files = glob.glob(f'{XML_DIR_STR}/*.xml')
    
    if not xml_files:
        print(f"No XML files found in {XML_DIR_STR}")
        print(f"Usage: Place XML files in {XML_DIR_STR} and run this script.")
        return
    
    print(f"Found {len(xml_files)} XML file(s)\n")
    
    # Parse all files
    parser = VolleyballDataParser()
    parsed_data = parser.parse_multiple_files(xml_files)
    
    # Save to JSON
    parser.save_to_json('volleyball_matches.json')
    
    # Print summary
    summary = parser.get_feature_summary()
    print(f"\n{'='*50}")
    print("DATA SUMMARY")
    print(f"{'='*50}")
    print(f"Total Matches: {summary['total_matches']}")
    print(f"Tournaments: {', '.join(summary['tournaments'])}")
    print(f"Teams: {', '.join(summary['teams'])}")
    print(f"\nStatistics Available ({len(summary['stat_types'])} types):")
    for stat in summary['stat_types'][:20]:  # Show first 20
        print(f"  - {stat}")
    if len(summary['stat_types']) > 20:
        print(f"  ... and {len(summary['stat_types']) - 20} more")


if __name__ == '__main__':
    main()
