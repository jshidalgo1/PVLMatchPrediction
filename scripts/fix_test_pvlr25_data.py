"""
Fix corrupted data in TEST_PVLR25 tournament.
The XML files have incorrect sets_won values - we need to recalculate from actual set scores.
"""

import sqlite3

def fix_match_from_set_scores(conn, match_id):
    """Recalculate sets_won from actual set scores."""
    cursor = conn.cursor()
    
    # Get set scores
    cursor.execute('''
        SELECT team_a_score, team_b_score
        FROM set_scores
        WHERE match_id = ?
        ORDER BY set_number
    ''', (match_id,))
    
    set_scores = cursor.fetchall()
    
    if not set_scores:
        print(f'  ⚠️  Match {match_id}: No set scores found')
        return
    
    # Calculate sets won (volleyball format: first to 3 wins)
    team_a_sets = sum(1 for a, b in set_scores if a > b)
    team_b_sets = sum(1 for a, b in set_scores if b > a)
    
    # In volleyball, match ends when one team wins 3 sets
    # The sets_won should be min(sets_won, 3) for winner and actual for loser
    if team_a_sets > team_b_sets:
        team_a_sets_won = 3  # Winner always shows 3
        team_b_sets_won = team_b_sets  # Loser shows how many they actually won
        winner_team = 'A'
    elif team_b_sets > team_a_sets:
        team_a_sets_won = team_a_sets
        team_b_sets_won = 3  # Winner always shows 3
        winner_team = 'B'
    else:
        print(f'  ❌ Match {match_id}: Still a tie after recalculation!')
        return
    
    # Get team IDs and set winner
    cursor.execute('SELECT team_a_id, team_b_id FROM matches WHERE id = ?', (match_id,))
    team_a_id, team_b_id = cursor.fetchone()
    winner_id = team_a_id if winner_team == 'A' else team_b_id
    
    # Update match
    cursor.execute('''
        UPDATE matches 
        SET team_a_sets_won = ?, team_b_sets_won = ?, winner_id = ?
        WHERE id = ?
    ''', (team_a_sets_won, team_b_sets_won, winner_id, match_id))
    
    return team_a_sets_won, team_b_sets_won

def main():
    conn = sqlite3.connect('volleyball_data.db')
    cursor = conn.cursor()
    
    print('=' * 80)
    print('FIXING TEST_PVLR25 TOURNAMENT DATA')
    print('=' * 80)
    
    # Get all matches in TEST_PVLR25
    cursor.execute('''
        SELECT id FROM matches 
        WHERE tournament_id = (SELECT id FROM tournaments WHERE code = 'TEST_PVLR25')
    ''')
    
    match_ids = [row[0] for row in cursor.fetchall()]
    print(f'\nFound {len(match_ids)} matches to fix\n')
    
    fixed = 0
    for match_id in match_ids:
        result = fix_match_from_set_scores(conn, match_id)
        if result:
            fixed += 1
    
    conn.commit()
    
    print(f'\n✅ Fixed {fixed}/{len(match_ids)} matches')
    
    # Verify - check for remaining issues
    print('\n' + '=' * 80)
    print('VERIFICATION')
    print('=' * 80)
    
    # Check for ties
    cursor.execute('''
        SELECT COUNT(*) FROM matches
        WHERE tournament_id = (SELECT id FROM tournaments WHERE code = 'TEST_PVLR25')
          AND team_a_sets_won = team_b_sets_won
    ''')
    ties = cursor.fetchone()[0]
    print(f'\nTie games: {ties} (should be 0)')
    
    # Check for invalid set totals
    cursor.execute('''
        SELECT COUNT(*) FROM matches
        WHERE tournament_id = (SELECT id FROM tournaments WHERE code = 'TEST_PVLR25')
          AND (team_a_sets_won + team_b_sets_won < 3 OR team_a_sets_won + team_b_sets_won > 5)
    ''')
    invalid = cursor.fetchone()[0]
    print(f'Invalid set totals: {invalid} (should be 0)')
    
    # Check winner consistency
    cursor.execute('''
        SELECT COUNT(*) FROM matches
        WHERE tournament_id = (SELECT id FROM tournaments WHERE code = 'TEST_PVLR25')
          AND NOT (
            (winner_id = team_a_id AND team_a_sets_won = 3) OR
            (winner_id = team_b_id AND team_b_sets_won = 3)
          )
    ''')
    winner_errors = cursor.fetchone()[0]
    print(f'Winner inconsistencies: {winner_errors} (should be 0)')
    
    if ties == 0 and invalid == 0 and winner_errors == 0:
        print('\n🎉 ALL DATA FIXED SUCCESSFULLY!')
    else:
        print('\n⚠️  Some issues remain')
    
    conn.close()

if __name__ == '__main__':
    main()
