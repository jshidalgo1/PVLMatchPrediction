# Player Statistics Documentation

Comprehensive guide to player statistics tracking, calculation, and display in the PVL Match Prediction system.

## Overview

The system tracks individual player performance across matches, calculating both cumulative totals and per-set averages for various skills.

## Available Metrics

### Total Metrics
- **Total Points**: Sum of attack, block, and serve points
- **Attack Points**: Points scored from attacks
- **Sets Played**: Number of sets player participated in

### Per-Set Average Metrics
- **Block Points per Set**: Block points / sets played
- **Serve Points per Set**: Serve points / sets played  
- **Digs per Set**: Excellent digs / sets played
- **Receptions per Set**: Excellent receptions / sets played
- **Sets per Set**: Excellent sets / sets played

## Sets Played Calculation

**Critical Metric**: Sets played is the denominator for all per-set averages. Accuracy is essential.

### The Challenge

A player can be on the court without recording any statistics in a set. This affects:
- **Liberos**: Never attack or block, may not serve
- **Defensive Specialists**: May only dig or receive
- **Substitutes**: May enter briefly without touching the ball

Previous implementations counted sets only if a player had stats, leading to inflated per-set averages.

### Solution: Roster-Based Calculation

**Implementation**: Parse the `<Roster>` element from each `<Set>` in the match XML.

**XML Structure**:
```xml
<Set no="1">
  <Roster p1="5" p2="11" p3="20" p4="16" p5="1" p6="4" 
          r1="8" r2="15" r3="" r4="" r5="" r6="" 
          l1="6" l2=""/>
  <Statistics>
    <!-- Player stats here -->
  </Statistics>
</Set>
```

**Roster Attributes**:
- `p1-p6`: Starting six players (jersey numbers)
- `r1-r6`: Substitute players who entered
- `l1-l2`: Libero players

**Algorithm** (`parse_volleyball_data.py`):

```python
def _extract_player_match_stats(match_element, roster_map):
    sets_by_player = {}  # jersey -> set count
    
    # For each set in match
    for set_element in match_element.findall('.//Set'):
        roster_element = set_element.find('Roster')
        if roster_element is None:
            continue
        
        # Extract all jersey numbers present
        starters = [roster_element.get(f'p{i}') for i in range(1, 7)]
        substitutes = [roster_element.get(f'r{i}') for i in range(1, 7)]
        liberos = [roster_element.get(f'l{i}') for i in range(1, 3)]
        
        # Combine and filter empty
        all_jerseys = set(filter(None, starters + substitutes + liberos))
        
        # Increment set count for each present player
        for jersey in all_jerseys:
            sets_by_player[jersey] = sets_by_player.get(jersey, 0) + 1
    
    # Assign to player stats
    for stat in player_stats:
        stat['sets_played'] = sets_by_player.get(stat['jersey_number'], 0)
    
    return player_stats
```

**Benefits**:
- Accurate even when player has zero stats
- Correctly tracks liberos and defensive specialists
- Enables meaningful per-set comparisons

### Database Storage

```sql
CREATE TABLE player_match_stats (
    ...
    sets_played INTEGER,  -- From roster parsing, NOT stat counting
    ...
);
```

Updated by `parse_volleyball_data.py`, stored via `database_manager.py`.

## Per-Set Average Calculation

### Formula

```
Stat Per Set = Total Stat / Sets Played
```

### Dashboard Implementation

**Location**: `dashboard/src/app/page.tsx`

```typescript
const getPerSet = (value: number, sets: number): number => {
  return sets > 0 ? value / sets : 0;
};

const formatStat = (value: number, sets: number, isAverage: boolean): string => {
  if (isAverage) {
    return getPerSet(value, sets).toFixed(2);
  }
  return value.toString();
};
```

**Usage**:
```typescript
// Display block points per set
<td>{formatStat(player.stats.block_points, player.stats.sets_played, true)}</td>

// Display total attack points
<td>{formatStat(player.stats.attack_points, player.stats.sets_played, false)}</td>
```

### Export Process

**Location**: `scripts/export_dashboard_data.py`

```python
def get_players_data(conn, tournament_id=None):
    query = """
        SELECT 
            pms.player_id,
            SUM(pms.sets_played) as total_sets,
            SUM(pms.attack_points) as attack_total,
            SUM(pms.block_points) as block_total,
            SUM(pms.serve_points) as serve_total,
            SUM(pms.dig_excellent) as dig_total,
            SUM(pms.reception_excellent) as reception_total,
            SUM(pms.set_excellent) as set_total
        FROM player_match_stats pms
        JOIN matches m ON pms.match_id = m.id
        WHERE m.tournament_id = ? AND pms.player_id IS NOT NULL
        GROUP BY pms.player_id
    """
    
    # ... aggregate and format for dashboard
    players_map[player_id]['stats'] = {
        'sets_played': total_sets,
        'total_points': attack + block + serve,
        'attack_points': attack,
        'block_points': block,
        'serve_points': serve,
        'dig_excellent': dig,
        'reception_excellent': reception,
        'set_excellent': set_stat
    }
```

Dashboard calculates per-set averages on the fly from these totals.

## Data Flow

```
XML File
  └─> parse_volleyball_data.py
      └─> Extract Roster tags per set
          └─> Count jersey appearances
              └─> Set sets_played in parsed data
                  └─> database_manager.py
                      └─> Insert into player_match_stats
                          └─> export_dashboard_data.py
                              └─> Aggregate by tournament
                                  └─> dashboard/public/data.json
                                      └─> Dashboard UI
                                          └─> Calculate per-set averages
                                              └─> Display to user
```

## Sorting and Filtering

### Dashboard Sorting

**Implementation**: Client-side JavaScript sorting

```typescript
const sortedPlayers = [...filteredPlayers].sort((a, b) => {
  switch (sortMetric) {
    case 'total':
      return b.stats.total_points - a.stats.total_points;
    case 'attack':
      return b.stats.attack_points - a.stats.attack_points;
    case 'block':
      return getPerSet(b.stats.block_points, b.stats.sets_played) - 
             getPerSet(a.stats.block_points, a.stats.sets_played);
    case 'serve':
      return getPerSet(b.stats.serve_points, b.stats.sets_played) - 
             getPerSet(a.stats.serve_points, a.stats.sets_played);
    // ... etc for dig, reception, set
  }
});
```

**Notes**:
- Totals (points, attack) sort by raw value
- Averages (block, serve, dig, reception, set) sort by calculated per-set value
- Both ascending and descending supported

### Tournament Filtering

Currently filtered to one tournament (TEST_PVLR25) in export:

```python
# export_dashboard_data.py
cursor.execute("SELECT id FROM tournaments WHERE code = 'TEST_PVLR25'")
current_tournament_id = cursor.fetchone()[0]

players = get_players_data(conn, tournament_id=current_tournament_id)
```

**To change tournament**:
Modify query to use different tournament code.

**Multi-tournament view**:
Remove `tournament_id` parameter to aggregate across all tournaments.

## Player Identification

### Name-to-ID Mapping

**Challenge**: XML provides names separately from jersey numbers.

**Solution**: Positional mapping + database lookup

**Step 1**: Extract names and jerseys from XML
```python
# Top-level player names (in order)
names = ['Player A', 'Player B', 'Player C', ...]

# Team roster jersey numbers (in order)
jerseys = ['5', '11', '20', '16', ...]

# Create roster map
roster_map = {jersey: name for jersey, name in zip(jerseys, names)}
```

**Step 2**: Insert players by name
```python
for name in names:
    cursor.execute("INSERT OR IGNORE INTO players (full_name) VALUES (?)", (name,))
```

**Step 3**: Map stats to player IDs
```python
cursor.execute("SELECT id, full_name FROM players WHERE full_name IN (...)")
name_to_id = {name: pid for pid, name in cursor.fetchall()}

for stat in player_stats:
    stat['player_id'] = name_to_id.get(stat['full_name'])
```

This links statistics (by jersey) to player names (by ID).

## Data Validation

### Checks to Run

**1. Verify sets_played > 0 for players with stats**:
```sql
SELECT COUNT(*) 
FROM player_match_stats 
WHERE (attack_points > 0 OR block_points > 0 OR serve_points > 0) 
AND sets_played = 0;
```
Should return 0.

**2. Check player_id linkage**:
```sql
SELECT COUNT(*) 
FROM player_match_stats 
WHERE player_id IS NULL;
```
Should be 0 or minimal (only for players without database entry).

**3. Validate per-set averages**:
```sql
SELECT 
    full_name,
    SUM(block_points) as total_block,
    SUM(sets_played) as total_sets,
    ROUND(CAST(SUM(block_points) AS FLOAT) / SUM(sets_played), 2) as block_per_set
FROM player_match_stats pms
JOIN players p ON pms.player_id = p.id
GROUP BY p.id
ORDER BY block_per_set DESC
LIMIT 10;
```

**4. Compare with XML source**:
Manually verify a few players' stats match raw XML data.

## Common Issues

### Issue 1: Player has stats but sets_played = 0

**Cause**: XML missing `<Roster>` tags, or parser not finding them

**Solution**:
- Check raw XML file for `<Roster>` presence
- Verify parser is finding `.//Set/Roster` elements
- Re-run `batch_processor.py` with updated parser

### Issue 2: Sets_played too high

**Cause**: Player counted multiple times due to duplicate jersey numbers

**Solution**:
- Use `set()` to ensure unique jerseys per set
- Verify positional mapping didn't create duplicates

### Issue 3: Libero has 0 sets

**Cause**: Libero listings in `l1/l2` not being parsed

**Solution**:
```python
# Ensure libero extraction
liberos = [roster_element.get(f'l{i}') for i in range(1, 3)]
all_jerseys = set(filter(None, starters + substitutes + liberos))
```

### Issue 4: Dashboard shows 0.00 for all per-set stats

**Cause**: sets_played = 0 for all players

**Solution**:
- Check database: `SELECT MAX(sets_played) FROM player_match_stats;`
- If 0, re-process data with roster-based calculation
- Run: `python scripts/batch_processor.py`

## Future Enhancements

- [ ] Track additional stats (aces, errors, efficiency rates)
- [ ] Per-set granularity (Set 1 vs Set 5 performance)
- [ ] Momentum indicators (recent form, streaks)
- [ ] Position-specific benchmarks
- [ ] Clutch performance metrics (tie-breaking points)
- [ ] Historical trends (improvement over season)
- [ ] Head-to-head player matchups

## Example Queries

### Top Blockers (Per-Set)

```sql
SELECT 
    p.full_name,
    t.name as team,
    SUM(pms.block_points) as total_blocks,
    SUM(pms.sets_played) as sets,
    ROUND(CAST(SUM(pms.block_points) AS FLOAT) / SUM(pms.sets_played), 2) as blocks_per_set
FROM player_match_stats pms
JOIN players p ON pms.player_id = p.id
JOIN teams t ON pms.team_id = t.id
GROUP BY p.id
HAVING sets > 0
ORDER BY blocks_per_set DESC
LIMIT 10;
```

### Best Servers

```sql
SELECT 
    p.full_name,
    SUM(pms.serve_points) as aces,
    SUM(pms.sets_played) as sets,
    ROUND(CAST(SUM(pms.serve_points) AS FLOAT) / SUM(pms.sets_played), 2) as aces_per_set
FROM player_match_stats pms
JOIN players p ON pms.player_id = p.id
GROUP BY p.id
HAVING sets > 0 AND aces > 0
ORDER BY aces_per_set DESC
LIMIT 10;
```

### Defensive Specialists

```sql
SELECT 
    p.full_name,
    SUM(pms.dig_excellent) as digs,
    SUM(pms.reception_excellent) as receptions,
    SUM(pms.sets_played) as sets,
    ROUND(CAST(SUM(pms.dig_excellent) AS FLOAT) / SUM(pms.sets_played), 2) as digs_per_set
FROM player_match_stats pms
JOIN players p ON pms.player_id = p.id
WHERE pms.is_libero = 1
GROUP BY p.id
ORDER BY digs_per_set DESC
LIMIT 10;
```

---

**Last Updated**: November 23, 2025
