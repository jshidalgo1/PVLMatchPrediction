export interface Team {
    id: number;
    code: string;
    name: string;
    coach: string | null;
    assistant_coach: string | null;
}

export interface PlayerStats {
    sets_played: number;
    total_points: number;
    attack_points: number;
    block_points: number;
    serve_points: number;
    dig_excellent: number;
    reception_excellent: number;
    set_excellent: number;
}

export interface Player {
    id: number;
    first_name: string;
    last_name: string;
    full_name: string;
    stats: PlayerStats;
    teams: string[];
}

export interface Standing {
    team: string;
    name?: string;
    wins: number;
    losses: number;
    games_played: number;
    rank?: number;
    win_pct?: number;
    match_points?: number;
    set_ratio?: number;
    point_ratio?: number;
}

export interface MatchPrediction {
    team_a: string;
    team_b: string;
    winner: string;
    confidence: number;
    pool: string;
    round?: string;
}

export interface PhasePredictions {
    phase_description: string;
    predictions: MatchPrediction[];
}

export interface PlayoffMatch {
    matchup: string;
    team_a: string;
    team_b: string;
    winner?: string;
    confidence?: number;
    is_actual: boolean;
}

export interface PlayoffBracket {
    quarterfinals: PlayoffMatch[];
    semifinals: PlayoffMatch[];
    championship: PlayoffMatch | null;
    third_place: PlayoffMatch | null;
    top_8_seeds: any[];
}

export interface SimulationResults {
    current_standings: Standing[];
    predictions_by_phase?: PhasePredictions[];
    final_standings: Standing[];
    matches_by_phase?: {
        [phase: string]: {
            completed_count: number;
            pending_count: number;
        };
    };
    playoffs?: PlayoffBracket | null;
    // Old fields for backward compatibility
    first_round_predictions?: MatchPrediction[];
    projected_standings_r1?: Standing[];
    second_round_predictions?: MatchPrediction[];
    pools?: any;
}

export interface Tournament {
    id: number;
    code: string;
    name: string;
    simulation: SimulationResults;
    history: TournamentHistory;
}

export interface HistoricalMatch {
    match_no: string;
    date: string;
    phase_no: number;
    phase_description: string;
    team_a: string;
    team_b: string;
    winner: string | null;
    team_a_sets: number;
    team_b_sets: number;
}

export interface PhaseMatches {
    phase_description: string;
    phase_no: number;
    matches: HistoricalMatch[];
}

export interface TournamentHistory {
    champion: string | null;
    all_matches: HistoricalMatch[];
    phases: PhaseMatches[];
}

export interface DashboardData {
    teams: Team[];
    players: Player[];
    simulation?: SimulationResults; // Old structure (temporary)
    tournaments?: Tournament[]; // New structure (for future)
    last_updated: string;
}
