"use client";

import { useEffect, useState } from "react";
import { DashboardData, Team } from "@/types";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Trophy } from "lucide-react";

export default function TeamsPage() {
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
                <div className="text-lg font-medium text-slate-600">Loading teams...</div>
            </div>
        );
    }

    if (!data || !data.teams) {
        return (
            <Card className="max-w-md mx-auto mt-20">
                <CardHeader>
                    <CardTitle className="text-red-600">Data not found</CardTitle>
                    <CardDescription>No team data available.</CardDescription>
                </CardHeader>
            </Card>
        );
    }

    return (
        <div className="container mx-auto p-4 space-y-8">
            <div className="text-center space-y-4 pb-8 border-b">
                <div className="flex items-center justify-center gap-3">
                    <Trophy className="h-10 w-10 text-indigo-600" />
                    <h1 className="text-4xl font-extrabold text-slate-900 tracking-tight">
                        Team Statistics
                    </h1>
                </div>
                <p className="text-lg text-slate-600">Performance metrics, top players, and recent form</p>
                <Badge variant="outline" className="text-xs">
                    Last Updated: {new Date(data.last_updated).toLocaleString()}
                </Badge>
            </div>

            <TeamsGrid teams={data.teams || []} />
        </div>
    );
}

function TeamsGrid({ teams }: { teams: Team[] }) {
    const teamsWithStats = teams.filter(t => t.statistics).sort((a, b) => (b.statistics?.wins || 0) - (a.statistics?.wins || 0));
    if (teamsWithStats.length === 0) return <div className="text-center text-slate-500 py-8">No team statistics available</div>;

    // Rank-based styling
    const getRankStyle = (index: number) => {
        if (index === 0) return "border-2 border-amber-400 bg-gradient-to-br from-amber-50 to-white shadow-lg"; // Gold
        if (index === 1) return "border-2 border-slate-300 bg-gradient-to-br from-slate-50 to-white shadow-md"; // Silver
        if (index === 2) return "border-2 border-orange-300 bg-gradient-to-br from-orange-50 to-white shadow-md"; // Bronze
        return "border border-slate-200 bg-white shadow-sm";
    };

    const getRankBadge = (index: number) => {
        if (index === 0) return <Badge className="bg-amber-500 text-white hover:bg-amber-600 font-bold">🏆 #1</Badge>;
        if (index === 1) return <Badge className="bg-slate-400 text-white hover:bg-slate-500 font-bold">🥈 #2</Badge>;
        if (index === 2) return <Badge className="bg-orange-400 text-white hover:bg-orange-500 font-bold">🥉 #3</Badge>;
        return <Badge variant="secondary" className="font-semibold">#{index + 1}</Badge>;
    };

    return (
        <div className="grid gap-6 md:grid-cols-2 lg:grid-cols-3">
            {teamsWithStats.map((team, index) => (
                <div key={team.id} className={`${getRankStyle(index)} rounded-xl p-5 hover:shadow-xl hover:scale-[1.02] transition-all duration-300`}>
                    {/* Header with Rank Badge */}
                    <div className="flex items-start justify-between mb-4">
                        <div className="flex-1">
                            <h3 className="font-bold text-xl text-slate-900">{team.code}</h3>
                            <p className="text-xs text-slate-500 line-clamp-1 mt-0.5">{team.name}</p>
                        </div>
                        <div className="flex flex-col items-end gap-1">
                            {getRankBadge(index)}
                            <Badge variant="outline" className="text-sm font-bold border-2">{team.statistics!.wins}-{team.statistics!.losses}</Badge>
                        </div>
                    </div>

                    {/* Stats Grid */}
                    <div className="grid grid-cols-2 gap-4 mb-4 pb-4 border-b-2 border-slate-100">
                        <div>
                            <p className="text-xs font-medium text-slate-500 uppercase tracking-wide">Win %</p>
                            <p className="text-2xl font-bold text-indigo-600">{team.statistics!.win_percentage.toFixed(1)}%</p>
                        </div>
                        {team.statistics!.current_elo && (
                            <div>
                                <p className="text-xs font-medium text-slate-500 uppercase tracking-wide">ELO</p>
                                <p className="text-2xl font-bold text-slate-800">{team.statistics!.current_elo.toFixed(0)}</p>
                            </div>
                        )}
                        <div>
                            <p className="text-xs font-medium text-slate-500 uppercase tracking-wide">Set Ratio</p>
                            <p className="text-lg font-semibold text-slate-700">{team.statistics!.set_ratio.toFixed(2)}</p>
                        </div>
                        <div>
                            <p className="text-xs font-medium text-slate-500 uppercase tracking-wide">Matches</p>
                            <p className="text-lg font-semibold text-slate-700">{team.statistics!.total_matches}</p>
                        </div>
                    </div>

                    {/* Last 5 Results */}
                    {team.statistics!.last_5_results && team.statistics!.last_5_results.length > 0 && (
                        <div className="mb-4">
                            <p className="text-xs font-medium text-slate-500 uppercase tracking-wide mb-2">Recent Form</p>
                            <div className="flex gap-1.5">
                                {team.statistics!.last_5_results.map((result: string, idx: number) => (
                                    <Badge
                                        key={idx}
                                        className={`text-sm px-3 py-1 font-bold ${result === 'W' ? 'bg-green-500 text-white hover:bg-green-600' : 'bg-red-500 text-white hover:bg-red-600'}`}
                                    >
                                        {result}
                                    </Badge>
                                ))}
                            </div>
                        </div>
                    )}

                    {/* Top Scorers */}
                    {team.top_players && team.top_players.length > 0 && (
                        <div>
                            <p className="text-xs font-medium text-slate-500 uppercase tracking-wide mb-2">Top Scorers</p>
                            <div className="space-y-2">
                                {team.top_players.slice(0, 3).map((player, idx: number) => (
                                    <div key={idx} className="flex justify-between items-center text-sm bg-slate-50 rounded-md px-3 py-1.5">
                                        <span className="text-slate-700 font-medium truncate">{player.name}</span>
                                        <span className="font-bold text-indigo-600 ml-2">{player.total_points}</span>
                                    </div>
                                ))}
                            </div>
                        </div>
                    )}
                </div>
            ))}
        </div>
    );
}
