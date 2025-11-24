# Dashboard Guide

Complete guide to the PVL Match Prediction Dashboard - a Next.js web application for visualizing tournament predictions, player statistics, and playoff brackets.

## Overview

The dashboard provides an interactive interface for exploring:
- Tournament standings and projections
- Player performance statistics
- Match predictions by phase
- Playoff bracket visualization

## Features

### 1. Tournament Standings

**Current Standings**
- Real-time win-loss records from database
- Teams sorted by FIVB ranking rules
- Match points, set ratio, and point ratio

**Projected Standings**
- Final standings after simulating remaining matches
- Playoff qualification indicators
- Ranking based on combined preliminary and second-round results

### 2. Player Statistics

**Metrics Display**
- **Totals**: Total Points, Attack Points
- **Per-Set Averages**: Block, Serve, Dig, Reception, Set stats divided by sets played

**Interactive Features**
- Click column headers to sort by any metric
- Sorting highlights active metric
- Per-tournament filtering (currently TEST_PVLR25)
- Top 10 players displayed by selected metric

**How Per-Set Averages Work**
```
Block/Set = Block Points / Sets Played
Serve/Set = Serve Points / Sets Played
Dig/Set = Dig Excellent / Sets Played
Rec/Set = Reception Excellent / Sets Played
Set/Set = Set Excellent / Sets Played
```

Sets played calculated from XML `<Roster>` tags (starters + substitutes + liberos per set).

### 3. Match Predictions

**Organization**
- Grouped by tournament phase
- Preliminary Round 1 (Groups A & B)
- Preliminary Round 2 (Groups C & D)
- Quarterfinals, Semifinals
- Championship and Third Place

**Match Cards**
- Team names and codes
- Predicted winner (highlighted)
- Confidence percentage
- "Actual" vs "Predicted" indicators

### 4. Playoff Bracket

**Visual Components**
- Quarterfinals (4 matches)
- Semifinals (2 matches)
- Championship match
- Third place match

**Bracket Display**
- Connecting lines showing bracket flow
- Winner highlighting with green background
- Confidence scores displayed
- Responsive layout

## Data Flow

### Backend → Dashboard

1. **Data Generation**
   ```bash
   # Run simulation
   python scripts/simulate_tournament.py --save_outputs
   
   # Export to dashboard
   python scripts/export_dashboard_data.py
   ```

2. **Export Process**
   - Loads latest simulation from `outputs/tournament_simulation_*.json`
   - Queries database for standings, matches, player stats
   - Filters player stats by tournament
   - Generates `dashboard/public/data.json`

3. **Dashboard Consumption**
   - Next.js reads `data.json` at build/runtime
   - React components render tournament data
   - Client-side sorting and filtering

### Data Structure

```typescript
interface DashboardData {
  tournaments: Tournament[]      // Multi-tournament support
  players: PlayerStats[]         // All players with stats
  teams: Team[]                  // Team information
  last_updated: string           // ISO timestamp
}

interface Tournament {
  id: number
  code: string
  name: string
  simulation: {
    current_standings: TeamStanding[]
    final_standings: TeamStanding[]
    predictions_by_phase: PhaseMatch[]
    playoffs: PlayoffBracket | null
  }
  history: TournamentHistory
}

interface PlayerStats {
  id: number
  full_name: string
  teams: string[]               // Team codes
  stats: {
    sets_played: number
    total_points: number
    attack_points: number
    block_points: number
    serve_points: number
    dig_excellent: number
    reception_excellent: number
    set_excellent: number
  }
}
```

## Technical Details

### Per-Set Average Calculation

**Challenge**: Calculate meaningful per-set statistics for players.

**Current Implementation**: Stats are divided by **team's total sets** in the tournament, not individual player's sets played.

**Rationale**: 
- Provides consistent baseline across all players on same team
- Simplifies comparison between players
- Avoids complexity of tracking individual player set appearances

**Location**: `scripts/parse_volleyball_data.py` → `_extract_player_match_stats()`

**Dashboard Display**: `dashboard/src/app/page.tsx`

```typescript
const getPerSet = (val: number | undefined, teamTotalSets: number | undefined) => {
  return (val && teamTotalSets && teamTotalSets > 0) ? val / teamTotalSets : 0;
};

const formatStat = (val: number | undefined, teamTotalSets: number | undefined, 
                   isAvg: boolean, isEfficiency: boolean = false, totalAttempts?: number) => {
  if (isEfficiency && totalAttempts) {
    return ((val || 0) / totalAttempts * 100).toFixed(1) + '%';
  }
  if (isAvg) {
    return getPerSet(val, teamTotalSets).toFixed(2);
  }
  return val?.toString() || '0';
};
```

**Note**: Reception stat displays as efficiency % (excellent receptions / total attempts).

### Playoff Bracket Loading

**Source Priority**:
1. Load from latest `outputs/tournament_simulation_*.json` (if exists)
2. Fall back to database queries + on-the-fly prediction
3. Return null if insufficient data

**Third Place Match**:
- Calculates losers from semifinals
- Predicts winner between SF losers
- Included in simulation output since Nov 2025

## Customization

### Changing Displayed Metrics

Edit `PlayerStatsTable` in `dashboard/src/app/page.tsx`:

```typescript
// Add new metric to sortMetric type
type SortMetric = 'total' | 'attack' | 'block' | 'serve' | 'dig' | 'reception' | 'set' | 'NEW_METRIC';

// Add sorting logic
const sortedPlayers = [...filteredPlayers].sort((a, b) => {
  if (sortMetric === 'NEW_METRIC') {
    return getPerSet(b.stats.new_stat, b.stats.sets_played) - 
           getPerSet(a.stats.new_stat, a.stats.sets_played);
  }
  // ... existing sorts
});

// Add table column
<th className="px-4 py-3 text-center">New</th>

// Add table cell
<td className="px-4 py-3 text-center">
  {formatStat(player.stats.new_stat, player.stats.sets_played, true)}
</td>
```

### Filtering by Different Tournament

Currently hardcoded to `TEST_PVLR25`. To change:

```python
# In scripts/export_dashboard_data.py
current_tournament_id = row[0] if row else None
# Change tournament code in query
cursor.execute("SELECT id FROM tournaments WHERE code = 'YOUR_CODE'")
```

### Styling Changes

Dashboard uses Tailwind CSS. Modify classes in JSX:

```tsx
// Change card background
<div className="bg-white">  // → bg-gray-50

// Change table header color
<thead className="bg-gray-50">  // → bg-blue-100

// Change highlight color
<tr className="bg-blue-50">  // → bg-green-50
```

## Troubleshooting

### Player Stats Empty

**Symptoms**: "No player statistics available"

**Causes**:
1. No players in database with `sets_played > 0`
2. `player_id` is NULL in `player_match_stats`
3. Tournament filter returns no matches

**Solutions**:
```bash
# Check database
sqlite3 data/databases/volleyball_data.db
SELECT COUNT(*) FROM player_match_stats WHERE player_id IS NOT NULL;

# Re-process data with updated parser
python scripts/batch_processor.py

# Re-export
python scripts/export_dashboard_data.py
```

### Bracket Not Showing

**Symptoms**: Playoff bracket section empty

**Causes**:
1. No simulation output file in `/outputs`
2. `playoffs` is `null` in data.json
3. Fewer than 8 teams in tournament

**Solutions**:
```bash
# Run simulation with --save_outputs
python scripts/simulate_tournament.py --save_outputs

# Verify outputs directory
ls -lt outputs/ | grep tournament_simulation

# Re-export
python scripts/export_dashboard_data.py
```

### Development Server Issues

**Port already in use**:
```bash
# Kill existing process
lsof -ti:3000 | xargs kill -9

# Or use different port
PORT=3001 npm run dev
```

**Module not found**:
```bash
# Clear cache and reinstall
rm -rf node_modules .next
npm install
```

## Performance

### Data Size
- Typical `data.json`: ~500KB - 2MB
- 500+ matches, 200+ players, 12 teams
- Loads in <100ms on modern hardware

### Optimization Tips
- Limit player stats to top 50 per metric
- Paginate match predictions if >100 matches
- Use React.memo for expensive components
- Virtualize long lists with react-window

## Future Enhancements

- [ ] Charts for player performance trends
- [ ] Historical tournament comparison
- [ ] Live match updates (WebSocket)
- [ ] Downloadable reports (PDF/CSV)
- [ ] Mobile-optimized layout
- [ ] Dark mode toggle
- [ ] Advanced filtering (by team, position, date range)

---

**Last Updated**: November 23, 2025
