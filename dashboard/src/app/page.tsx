"use client";

import { useEffect, useState } from "react";
import { Standing, DashboardData, PhasePredictions } from "@/types";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { PlayoffBracketView } from "@/components/PlayoffBracket";
import { Trophy, TrendingUp, CheckCircle2 } from "lucide-react";

export default function Home() {
    const [data, setData] = useState<DashboardData | null>(null);
    const [loading, setLoading] = useState(true);

    useEffect(() => {
        fetch("/data.json?t=" + Date.now())
            .then(res => res.json())
            .then((jsonData: DashboardData) => {
                setData(jsonData);
                setLoading(false);
            })
            .catch(err => {
                console.error("Failed to load data:", err);
                setLoading(false);
            });
    }, []);

    if (loading) {
        return (
            <div className="flex items-center justify-center min-h-screen">
                <div className="text-lg font-medium text-slate-600">Loading simulation...</div>
            </div>
        );
    }

    if (!data || !data.tournaments || data.tournaments.length === 0) {
        return (
            <Card className="max-w-md mx-auto mt-20">
                <CardHeader>
                    <CardTitle className="text-red-600">Data not found</CardTitle>
                    <CardDescription>Please run the simulation script to generate data.</CardDescription>
                </CardHeader>
            </Card>
        );
    }

    // Find TEST_PVLR25 (current tournament)
    const currentTournament = data.tournaments.find(t => t.code === "TEST_PVLR25");

    if (!currentTournament || !currentTournament.simulation) {
        return (
            <Card className="max-w-md mx-auto mt-20">
                <CardHeader>
                    <CardTitle className="text-red-600">Current tournament unavailable</CardTitle>
                    <CardDescription>No simulation data available for the current tournament.</CardDescription>
                </CardHeader>
            </Card>
        );
    }

    const simulation = currentTournament.simulation;
    const hasPredictions = simulation.predictions_by_phase && simulation.predictions_by_phase.length > 0;

    return (
        <div className="container mx-auto p-4 space-y-8">
            <div className="text-center space-y-4 pb-8 border-b">
                <div className="flex items-center justify-center gap-3">
                    <Trophy className="h-10 w-10 text-indigo-600" />
                    <h1 className="text-4xl font-extrabold text-slate-900 tracking-tight">
                        Current Tournament Simulation
                    </h1>
                </div>
                <p className="text-lg text-slate-600">AI-Powered Match Predictions</p>
                <Badge variant="outline" className="text-xs">
                    Last Updated: {new Date(data.last_updated).toLocaleString()}
                </Badge>
            </div>

            {/* Current Tournament Display */}
            <div className="text-center pb-4">
                <h2 className="text-2xl font-bold text-indigo-900">{currentTournament.name}</h2>
            </div>

            {/* Tournament Phase Status */}
            {simulation.matches_by_phase && (
                <Card>
                    <CardHeader>
                        <CardTitle>Tournament Progress</CardTitle>
                        <CardDescription>Match completion by phase</CardDescription>
                    </CardHeader>
                    <CardContent>
                        <div className="grid gap-4 md:grid-cols-2">
                            {Object.entries(simulation.matches_by_phase).map(([phase, stats]) => (
                                <div key={phase} className="flex items-center justify-between p-4 rounded-lg border bg-white">
                                    <div className="flex-1">
                                        <p className="font-semibold text-slate-900">{phase}</p>
                                        <p className="text-sm text-slate-500">
                                            {stats.completed_count} played, {stats.pending_count} remaining
                                        </p>
                                    </div>
                                    {stats.pending_count === 0 && (
                                        <CheckCircle2 className="h-6 w-6 text-green-600" />
                                    )}
                                </div>
                            ))}
                        </div>
                    </CardContent>
                </Card>
            )}


            <Card>
                <CardHeader>
                    <CardTitle className="flex items-center gap-2">
                        <TrendingUp className="h-5 w-5" />
                        Current Standings
                    </CardTitle>
                    <CardDescription>Live standings from completed matches</CardDescription>
                </CardHeader>
                <CardContent>
                    <StandingsTable standings={simulation.current_standings} />
                </CardContent>
            </Card>

            {/* Playoff Bracket */}
            {simulation.playoffs && (
                <div className="mt-8 mb-8">
                    <PlayoffBracketView bracket={simulation.playoffs} />
                </div>
            )}

            {/* Predictions by Phase */}
            {hasPredictions ? (
                <div className="space-y-6">
                    <h2 className="text-2xl font-bold text-slate-900">Upcoming Match Predictions</h2>
                    {simulation.predictions_by_phase!.map((phaseData, idx) => (
                        <Card key={idx}>
                            <CardHeader>
                                <CardTitle>{phaseData.phase_description}</CardTitle>
                                <CardDescription>
                                    {phaseData.predictions.length} {phaseData.predictions.length === 1 ? 'match' : 'matches'} to be played
                                </CardDescription>
                            </CardHeader>
                            <CardContent>
                                <PredictionsList predictions={phaseData.predictions} />
                            </CardContent>
                        </Card>
                    ))}
                </div>
            ) : (
                <Card className="border-green-200 bg-green-50/50">
                    <CardHeader>
                        <CardTitle className="text-green-900 flex items-center gap-2">
                            <CheckCircle2 className="h-6 w-6" />
                            Tournament Complete
                        </CardTitle>
                        <CardDescription>All matches have been played. Check the final standings below.</CardDescription>
                    </CardHeader>
                </Card>
            )}

            <Card className="border-indigo-200 bg-indigo-50/50">
                <CardHeader>
                    <CardTitle className="text-indigo-900">Final Standings</CardTitle>
                    <CardDescription>Current tournament rankings</CardDescription>
                </CardHeader>
                <CardContent>
                    <StandingsTable standings={simulation.final_standings} />
                </CardContent>
            </Card>
        </div>
    );
}

function StandingsTable({ standings }: { standings: Standing[] }) {
    return (
        <div className="rounded-md border">
            <div className="overflow-x-auto">
                <table className="w-full text-sm">
                    <thead>
                        <tr className="border-b bg-slate-50/50">
                            <th className="h-12 px-4 text-left align-middle font-medium text-slate-500">Team</th>
                            <th className="h-12 px-4 text-left align-middle font-medium text-slate-500">Record</th>
                            <th className="h-12 px-4 text-left align-middle font-medium text-slate-500">Games</th>
                        </tr>
                    </thead>
                    <tbody>
                        {standings.map((team, idx) => (
                            <tr key={team.team} className="border-b transition-colors hover:bg-slate-50/50">
                                <td className="p-4 align-middle">
                                    <span className="font-semibold text-indigo-600">{team.team}</span>
                                </td>
                                <td className="p-4 align-middle">
                                    <Badge variant="outline">{team.wins}-{team.losses}</Badge>
                                </td>
                                <td className="p-4 align-middle text-slate-600">{team.games_played}</td>
                            </tr>
                        ))}
                    </tbody>
                </table>
            </div>
        </div>
    );
}

function PredictionsList({ predictions }: { predictions: any[] }) {
    return (
        <div className="space-y-3">
            {predictions.map((match, idx) => (
                <div key={idx} className="flex items-center justify-between p-4 rounded-lg border bg-white hover:bg-slate-50/50 transition-colors">
                    <div className="flex items-center gap-4 flex-1">
                        <span className="font-medium text-slate-700 min-w-[60px] text-right">{match.team_a}</span>
                        <span className="text-xs text-slate-400 font-semibold">VS</span>
                        <span className="font-medium text-slate-700 min-w-[60px]">{match.team_b}</span>
                    </div>
                    <div className="flex items-center gap-3">
                        <Badge className="bg-green-100 text-green-800 hover:bg-green-100">
                            {match.winner}
                        </Badge>
                        <span className="text-xs text-slate-500 tabular-nums">
                            {(match.confidence * 100).toFixed(0)}%
                        </span>
                    </div>
                </div>
            ))}
        </div>
    );
}

function PlayerStatsTable({ players }: { players: any[] }) {
    const [sortMetric, setSortMetric] = useState<'total' | 'attack' | 'block' | 'serve' | 'dig' | 'reception' | 'set'>('total');

    const getPerSet = (val: number | undefined, teamTotalSets: number | undefined) => {
        if (!val || !teamTotalSets || teamTotalSets === 0) return 0;
        return val / teamTotalSets;
    };

    const getEfficiency = (excellent: number | undefined, totalAttempts: number | undefined) => {
        if (!excellent || !totalAttempts || totalAttempts === 0) return 0;
        return (excellent / totalAttempts) * 100;
    };

    // Sort based on selected metric
    const sortedPlayers = [...players].sort((a, b) => {
        const statsA = a.stats || {};
        const statsB = b.stats || {};
        const teamSetsA = statsA.team_total_sets || 1;
        const teamSetsB = statsB.team_total_sets || 1;

        switch (sortMetric) {
            case 'attack':
                return (statsB.attack_points || 0) - (statsA.attack_points || 0);
            case 'block':
                return getPerSet(statsB.block_points, teamSetsB) - getPerSet(statsA.block_points, teamSetsA);
            case 'serve':
                return getPerSet(statsB.serve_points, teamSetsB) - getPerSet(statsA.serve_points, teamSetsA);
            case 'dig':
                return getPerSet(statsB.dig_excellent, teamSetsB) - getPerSet(statsA.dig_excellent, teamSetsA);
            case 'reception':
                return getEfficiency(statsB.reception_excellent, statsB.reception_total_attempts) - getEfficiency(statsA.reception_excellent, statsA.reception_total_attempts);
            case 'set':
                return getPerSet(statsB.set_excellent, teamSetsB) - getPerSet(statsA.set_excellent, teamSetsA);
            case 'total':
            default:
                return (statsB.total_points || 0) - (statsA.total_points || 0);
        }
    }).slice(0, 20); // Top 20

    const getMetricLabel = (metric: string) => {
        switch (metric) {
            case 'attack': return 'Best Spikers (Total)';
            case 'block': return 'Best Blockers (Avg/Set)';
            case 'serve': return 'Best Servers (Avg/Set)';
            case 'dig': return 'Best Diggers (Avg/Set)';
            case 'reception': return 'Best Receivers (Efficiency %)';
            case 'set': return 'Best Setters (Avg/Set)';
            default: return 'Top Scorers (Total)';
        }
    };

    const formatStat = (val: number | undefined, teamTotalSets: number | undefined, isAvg: boolean, isEfficiency: boolean = false, totalAttempts?: number) => {
        if (isEfficiency && totalAttempts !== undefined) {
            return getEfficiency(val, totalAttempts).toFixed(2) + '%';
        }
        if (isAvg) {
            return getPerSet(val, teamTotalSets).toFixed(2);
        }
        return val || 0;
    };

    return (
        <div className="space-y-4">
            <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-4">
                <h3 className="text-sm font-medium text-slate-500">{getMetricLabel(sortMetric)}</h3>
                <div className="flex flex-wrap gap-1 bg-slate-100 p-1 rounded-lg">
                    {(['total', 'attack', 'block', 'serve', 'dig', 'reception', 'set'] as const).map((metric) => (
                        <button
                            key={metric}
                            onClick={() => setSortMetric(metric)}
                            className={`px-3 py-1.5 text-xs font-medium rounded-md transition-all ${sortMetric === metric
                                ? 'bg-white text-indigo-600 shadow-sm'
                                : 'text-slate-500 hover:text-slate-700'
                                }`}
                        >
                            {metric.charAt(0).toUpperCase() + metric.slice(1)}
                        </button>
                    ))}
                </div>
            </div>

            <div className="rounded-md border">
                <div className="overflow-x-auto">
                    <table className="w-full text-sm">
                        <thead>
                            <tr className="border-b bg-slate-50/50">
                                <th className="h-12 px-4 text-left align-middle font-medium text-slate-500">Rank</th>
                                <th className="h-12 px-4 text-left align-middle font-medium text-slate-500">Player</th>
                                <th className="h-12 px-4 text-left align-middle font-medium text-slate-500">Team</th>
                                <th className={`h-12 px-4 text-right align-middle font-medium ${sortMetric === 'total' ? 'text-indigo-600 bg-indigo-50/30' : 'text-slate-500'}`}>Points</th>
                                <th className={`h-12 px-4 text-right align-middle font-medium ${sortMetric === 'attack' ? 'text-indigo-600 bg-indigo-50/30' : 'text-slate-500'}`}>Attack</th>
                                <th className={`h-12 px-4 text-right align-middle font-medium ${sortMetric === 'block' ? 'text-indigo-600 bg-indigo-50/30' : 'text-slate-500'}`}>Block/S</th>
                                <th className={`h-12 px-4 text-right align-middle font-medium ${sortMetric === 'serve' ? 'text-indigo-600 bg-indigo-50/30' : 'text-slate-500'}`}>Serve/S</th>
                                <th className={`h-12 px-4 text-right align-middle font-medium ${sortMetric === 'dig' ? 'text-indigo-600 bg-indigo-50/30' : 'text-slate-500'}`}>Dig/S</th>
                                <th className={`h-12 px-4 text-right align-middle font-medium ${sortMetric === 'reception' ? 'text-indigo-600 bg-indigo-50/30' : 'text-slate-500'}`}>Receive</th>
                                <th className={`h-12 px-4 text-right align-middle font-medium ${sortMetric === 'set' ? 'text-indigo-600 bg-indigo-50/30' : 'text-slate-500'}`}>Set/S</th>
                            </tr>
                        </thead>
                        <tbody>
                            {sortedPlayers.map((player, idx) => (
                                <tr key={player.id} className="border-b transition-colors hover:bg-slate-50/50">
                                    <td className="p-4 align-middle text-slate-500">{idx + 1}</td>
                                    <td className="p-4 align-middle font-medium text-slate-900">{player.full_name}</td>
                                    <td className="p-4 align-middle">
                                        {player.teams.map((t: string) => (
                                            <Badge key={t} variant="outline" className="mr-1">{t}</Badge>
                                        ))}
                                    </td>
                                    <td className={`p-4 align-middle text-right font-bold ${sortMetric === 'total' ? 'text-indigo-600 bg-indigo-50/30' : 'text-slate-600'}`}>{formatStat(player.stats?.total_points, player.stats?.team_total_sets, false)}</td>
                                    <td className={`p-4 align-middle text-right ${sortMetric === 'attack' ? 'text-indigo-600 font-bold bg-indigo-50/30' : 'text-slate-600'}`}>{formatStat(player.stats?.attack_points, player.stats?.team_total_sets, false)}</td>
                                    <td className={`p-4 align-middle text-right ${sortMetric === 'block' ? 'text-indigo-600 font-bold bg-indigo-50/30' : 'text-slate-600'}`}>{formatStat(player.stats?.block_points, player.stats?.team_total_sets, true)}</td>
                                    <td className={`p-4 align-middle text-right ${sortMetric === 'serve' ? 'text-indigo-600 font-bold bg-indigo-50/30' : 'text-slate-600'}`}>{formatStat(player.stats?.serve_points, player.stats?.team_total_sets, true)}</td>
                                    <td className={`p-4 align-middle text-right ${sortMetric === 'dig' ? 'text-indigo-600 font-bold bg-indigo-50/30' : 'text-slate-600'}`}>{formatStat(player.stats?.dig_excellent, player.stats?.team_total_sets, true)}</td>
                                    <td className={`p-4 align-middle text-right ${sortMetric === 'reception' ? 'text-indigo-600 font-bold bg-indigo-50/30' : 'text-slate-600'}`}>{formatStat(player.stats?.reception_excellent, player.stats?.team_total_sets, false, true, player.stats?.reception_total_attempts)}</td>
                                    <td className={`p-4 align-middle text-right ${sortMetric === 'set' ? 'text-indigo-600 font-bold bg-indigo-50/30' : 'text-slate-600'}`}>{formatStat(player.stats?.set_excellent, player.stats?.team_total_sets, true)}</td>
                                </tr>
                            ))}
                        </tbody>
                    </table>
                </div>
            </div>
        </div>
    );
}

