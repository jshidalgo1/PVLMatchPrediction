import React from 'react';
import { PlayoffMatch, PlayoffBracket } from '@/types';
import { cn } from '@/lib/utils';
import { Trophy, Medal } from 'lucide-react';

interface PlayoffBracketProps {
    bracket: PlayoffBracket;
}

export function PlayoffBracketView({ bracket }: PlayoffBracketProps) {
    if (!bracket) return null;

    // Helper to find match by ID/Label
    const getMatch = (list: PlayoffMatch[], id: string) => list.find(m => m.matchup.startsWith(id));

    const qf1 = getMatch(bracket.quarterfinals, 'QF1');
    const qf4 = getMatch(bracket.quarterfinals, 'QF4');
    const qf2 = getMatch(bracket.quarterfinals, 'QF2');
    const qf3 = getMatch(bracket.quarterfinals, 'QF3');

    const sf1 = getMatch(bracket.semifinals, 'SF1');
    const sf2 = getMatch(bracket.semifinals, 'SF2');

    return (
        <div className="w-full overflow-x-auto py-12 px-8 bg-slate-50/50 rounded-xl border border-slate-100">
            <div className="min-w-[900px] flex items-center justify-center">

                {/* COLUMN 1: QUARTERFINALS */}
                <div className="flex flex-col gap-10">
                    <div className="flex flex-col gap-10">
                        {/* Group 1 */}
                        <div className="flex flex-col gap-6">
                            <BracketMatch match={qf1} />
                            <BracketMatch match={qf4} />
                        </div>
                        {/* Group 2 */}
                        <div className="flex flex-col gap-6">
                            <BracketMatch match={qf2} />
                            <BracketMatch match={qf3} />
                        </div>
                    </div>
                </div>

                {/* CONNECTOR COLUMN 1 (QF -> SF) */}
                <div className="flex flex-col gap-10 mx-4">
                    {/* Connector for SF1 */}
                    <div className="h-[196px] w-[40px] relative">
                        <svg className="absolute inset-0 w-full h-full" style={{ overflow: 'visible' }}>
                            {/* Top Fork */}
                            <path d="M0,45 H20 Q30,45 30,55 V88" fill="none" stroke="#cbd5e1" strokeWidth="2" />
                            {/* Bottom Fork */}
                            <path d="M0,151 H20 Q30,151 30,141 V108" fill="none" stroke="#cbd5e1" strokeWidth="2" />
                            {/* Horizontal to SF */}
                            <path d="M30,98 H40" fill="none" stroke="#cbd5e1" strokeWidth="2" />
                        </svg>
                    </div>

                    {/* Connector for SF2 */}
                    <div className="h-[196px] w-[40px] relative">
                        <svg className="absolute inset-0 w-full h-full" style={{ overflow: 'visible' }}>
                            {/* Top Fork */}
                            <path d="M0,45 H20 Q30,45 30,55 V88" fill="none" stroke="#cbd5e1" strokeWidth="2" />
                            {/* Bottom Fork */}
                            <path d="M0,151 H20 Q30,151 30,141 V108" fill="none" stroke="#cbd5e1" strokeWidth="2" />
                            {/* Horizontal to SF */}
                            <path d="M30,98 H40" fill="none" stroke="#cbd5e1" strokeWidth="2" />
                        </svg>
                    </div>
                </div>

                {/* COLUMN 2: SEMIFINALS */}
                <div className="flex flex-col gap-[136px]">
                    <BracketMatch match={sf1} />
                    <BracketMatch match={sf2} />
                </div>

                {/* CONNECTOR COLUMN 2 (SF -> FINALS) */}
                <div className="w-[60px] h-[430px] relative mx-4">
                    <svg className="absolute inset-0 w-full h-full" style={{ overflow: 'visible' }}>
                        {/* Top Path from SF1 */}
                        <path d="M0,45 H30 Q40,45 40,55 V205" fill="none" stroke="#cbd5e1" strokeWidth="2" />
                        {/* Bottom Path from SF2 */}
                        <path d="M0,385 H30 Q40,385 40,375 V225" fill="none" stroke="#cbd5e1" strokeWidth="2" />
                        {/* Horizontal to Final */}
                        <path d="M40,215 H60" fill="none" stroke="#cbd5e1" strokeWidth="2" />
                    </svg>
                </div>

                {/* COLUMN 3: FINALS */}
                <div className="flex flex-col gap-16">
                    <div className="flex flex-col items-center gap-2">
                        <div className="flex items-center gap-2 text-amber-600 font-bold text-sm uppercase tracking-wider">
                            <Trophy className="h-4 w-4" /> Championship
                        </div>
                        <BracketMatch match={bracket.championship} isChampionship />
                    </div>

                    <div className="flex flex-col items-center gap-2 mt-8">
                        <div className="flex items-center gap-2 text-slate-500 font-bold text-xs uppercase tracking-wider">
                            <Medal className="h-3 w-3" /> 3rd Place
                        </div>
                        <BracketMatch match={bracket.third_place} />
                    </div>
                </div>

            </div>
        </div>
    );
}

function BracketMatch({ match, isChampionship = false }: { match?: PlayoffMatch | null; isChampionship?: boolean }) {
    if (!match) return (
        <div className="h-[90px] w-[220px] border-2 border-dashed border-slate-200 rounded-xl flex items-center justify-center text-slate-300 text-xs bg-white/50">
            TBD
        </div>
    );

    const confidence = match.confidence ? (match.confidence * 100).toFixed(0) : null;

    return (
        <div className={cn(
            "relative bg-white rounded-xl border shadow-sm transition-all hover:shadow-md w-[220px] z-10 overflow-hidden group",
            isChampionship ? "border-amber-200 shadow-amber-100/50 ring-1 ring-amber-100" : "border-slate-200"
        )}>
            {/* Header */}
            <div className={cn(
                "px-3 py-2 text-[10px] font-bold uppercase tracking-wider border-b flex justify-between items-center",
                isChampionship ? "bg-gradient-to-r from-amber-50 to-orange-50 text-amber-700 border-amber-100" : "bg-slate-50 text-slate-500 border-slate-100"
            )}>
                <span>{match.matchup}</span>
                {match.is_actual ? (
                    <span className="bg-slate-900 text-white px-1.5 py-0.5 rounded text-[9px] font-bold">FINAL</span>
                ) : (
                    <span className="text-slate-400 font-medium">Proj</span>
                )}
            </div>

            {/* Teams */}
            <div className="p-3 space-y-2">
                <TeamRow team={match.team_a} winner={match.winner} />
                <TeamRow team={match.team_b} winner={match.winner} />
            </div>

            {/* Footer / Confidence */}
            {!match.is_actual && match.winner && (
                <div className="px-3 py-2 border-t text-[10px] flex justify-between items-center bg-slate-50/30">
                    <span className="text-slate-400 font-medium">Win Probability</span>
                    <span className={cn("font-bold", Number(confidence) > 70 ? "text-green-600" : "text-slate-600")}>
                        {confidence}%
                    </span>
                </div>
            )}

            {/* Hover Effect Line */}
            <div className={cn("absolute left-0 top-0 bottom-0 w-1 opacity-0 group-hover:opacity-100 transition-opacity",
                isChampionship ? "bg-amber-400" : "bg-indigo-500"
            )} />
        </div>
    );
}

function TeamRow({ team, winner }: { team: string; winner?: string }) {
    const isWinner = winner === team;
    const isLoser = winner && winner !== team;

    return (
        <div className={cn(
            "flex justify-between items-center px-2 py-1 rounded transition-colors",
            isWinner ? "bg-green-50 text-green-700 font-bold" : "text-slate-700 font-medium",
            isLoser && "opacity-50 grayscale"
        )}>
            <span className="text-sm">{team}</span>
            {isWinner && <CheckIcon className="h-3.5 w-3.5 text-green-600" />}
        </div>
    );
}

function CheckIcon(props: React.SVGProps<SVGSVGElement>) {
    return (
        <svg
            {...props}
            xmlns="http://www.w3.org/2000/svg"
            width="24"
            height="24"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth="3"
            strokeLinecap="round"
            strokeLinejoin="round"
        >
            <polyline points="20 6 9 17 4 12" />
        </svg>
    );
}
