"use client";

import { useEffect, useState } from "react";
import { DashboardData, Tournament, PhaseMatches, HistoricalMatch } from "@/types";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import {
    Select,
    SelectContent,
    SelectItem,
    SelectTrigger,
    SelectValue,
} from "@/components/ui/select";
import {
    Accordion,
    AccordionContent,
    AccordionItem,
    AccordionTrigger,
} from "@/components/ui/accordion";
import { Trophy } from "lucide-react";

export default function HistoryPage() {
    const [data, setData] = useState<DashboardData | null>(null);
    const [selectedTournamentId, setSelectedTournamentId] = useState<number | null>(null);
    const [loading, setLoading] = useState(true);

    useEffect(() => {
        fetch("/data.json")
            .then(res => res.json())
            .then((jsonData: DashboardData) => {
                setData(jsonData);
                // Default to first completed tournament (not TEST_PVLR25)
                if (jsonData.tournaments && jsonData.tournaments.length > 0) {
                    const completedTournaments = jsonData.tournaments.filter(t => t.code !== "TEST_PVLR25");
                    if (completedTournaments.length > 0) {
                        setSelectedTournamentId(completedTournaments[0].id);
                    }
                }
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
                <div className="text-lg font-medium text-slate-600">Loading tournament history...</div>
            </div>
        );
    }

    if (!data || !data.tournaments) {
        return (
            <Card className="max-w-md mx-auto mt-20">
                <CardHeader>
                    <CardTitle className="text-red-600">Data not found</CardTitle>
                    <CardDescription>No tournament history available.</CardDescription>
                </CardHeader>
            </Card>
        );
    }

    // Filter out current tournament (TEST_PVLR25)
    const completedTournaments = data.tournaments.filter(t => t.code !== "TEST_PVLR25");

    if (completedTournaments.length === 0) {
        return (
            <Card className="max-w-md mx-auto mt-20">
                <CardHeader>
                    <CardTitle>No Completed Tournaments</CardTitle>
                    <CardDescription>Only the current tournament is available. Check back later for historical results!</CardDescription>
                </CardHeader>
            </Card>
        );
    }

    const selectedTournament = completedTournaments.find(t => t.id === selectedTournamentId);

    if (!selectedTournament) {
        return null;
    }

    const history = selectedTournament.history;

    return (
        <div className="space-y-8">
            {/* Header */}
            <div className="text-center space-y-4 pb-8 border-b">
                <div className="flex items-center justify-center gap-3">
                    <Trophy className="h-10 w-10 text-amber-600" />
                    <h1 className="text-4xl font-extrabold text-slate-900 tracking-tight">
                        Tournament History
                    </h1>
                </div>
                <p className="text-lg text-slate-600">Explore past tournament results and champions</p>

                {/* Tournament Selector */}
                <div className="flex items-center justify-center gap-4 pt-4">
                    <label className="text-sm font-medium text-slate-700">Select Tournament:</label>
                    <Select
                        value={selectedTournamentId?.toString()}
                        onValueChange={(value) => setSelectedTournamentId(Number(value))}
                    >
                        <SelectTrigger className="w-[350px]">
                            <SelectValue placeholder="Select a completed tournament" />
                        </SelectTrigger>
                        <SelectContent>
                            {completedTournaments.map((tournament) => (
                                <SelectItem key={tournament.id} value={tournament.id.toString()}>
                                    {tournament.name}
                                </SelectItem>
                            ))}
                        </SelectContent>
                    </Select>
                </div>
            </div>

            {/* Tournament Name & Champion */}
            <div className="text-center space-y-4">
                <h2 className="text-3xl font-bold text-indigo-900">{selectedTournament.name}</h2>
                <p className="text-sm text-slate-500">Code: {selectedTournament.code}</p>

                {history.champion && (
                    <Card className="max-w-md mx-auto border-amber-500 bg-amber-50">
                        <CardHeader>
                            <div className="flex items-center justify-center gap-3">
                                <Trophy className="h-8 w-8 text-amber-600" />
                                <div>
                                    <CardTitle className="text-amber-900">Champion</CardTitle>
                                    <CardDescription className="text-3xl font-bold text-amber-700 mt-2">
                                        {history.champion}
                                    </CardDescription>
                                </div>
                                <Trophy className="h-8 w-8 text-amber-600" />
                            </div>
                        </CardHeader>
                    </Card>
                )}
            </div>

            {/* Match History by Phase */}
            <Card>
                <CardHeader>
                    <CardTitle>Tournament Phases</CardTitle>
                    <CardDescription>
                        {history.all_matches.length} total matches across {history.phases.length} phases
                    </CardDescription>
                </CardHeader>
                <CardContent>
                    {history.phases.length === 0 ? (
                        <p className="text-center text-slate-500 py-8">No match data available</p>
                    ) : (
                        <Accordion type="single" collapsible className="w-full">
                            {history.phases.map((phase, idx) => (
                                <AccordionItem key={idx} value={`phase-${idx}`}>
                                    <AccordionTrigger className="hover:no-underline">
                                        <div className="flex items-center justify-between w-full pr-4">
                                            <span className="font-semibold text-lg">
                                                {phase.phase_description}
                                            </span>
                                            <Badge variant="outline">
                                                {phase.matches.length} {phase.matches.length === 1 ? 'match' : 'matches'}
                                            </Badge>
                                        </div>
                                    </AccordionTrigger>
                                    <AccordionContent>
                                        <div className="space-y-2 pt-4">
                                            {phase.matches.map((match, matchIdx) => (
                                                <MatchCard key={matchIdx} match={match} />
                                            ))}
                                        </div>
                                    </AccordionContent>
                                </AccordionItem>
                            ))}
                        </Accordion>
                    )}
                </CardContent>
            </Card>
        </div>
    );
}

function MatchCard({ match }: { match: HistoricalMatch }) {
    const formatDate = (dateStr: string) => {
        if (!dateStr) return "TBD";
        // Format YYYYMMDD to Month DD, YYYY
        const year = dateStr.substring(0, 4);
        const month = dateStr.substring(4, 6);
        const day = dateStr.substring(6, 8);
        const date = new Date(parseInt(year), parseInt(month) - 1, parseInt(day));
        return date.toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' });
    };

    return (
        <div className="flex items-center justify-between p-4 rounded-lg border bg-white hover:bg-slate-50/50 transition-colors">
            <div className="flex items-center gap-4 flex-1">
                <span className="text-xs text-slate-500 font-mono min-w-[80px]">
                    {formatDate(match.date)}
                </span>
                <div className="flex items-center gap-4 flex-1">
                    <div className={`font-semibold text-right min-w-[60px] ${match.winner === match.team_a ? 'text-green-700' : 'text-slate-600'}`}>
                        {match.team_a}
                    </div>
                    <Badge variant="outline" className="font-mono">
                        {match.team_a_sets} - {match.team_b_sets}
                    </Badge>
                    <div className={`font-semibold min-w-[60px] ${match.winner === match.team_b ? 'text-green-700' : 'text-slate-600'}`}>
                        {match.team_b}
                    </div>
                </div>
            </div>
            {match.winner && (
                <Badge className="bg-green-100 text-green-800 hover:bg-green-100 ml-4">
                    {match.winner} wins
                </Badge>
            )}
        </div>
    );
}
