import { getDashboardData } from "@/lib/data";
import Link from "next/link";

export const revalidate = 0;

export default async function TeamsPage() {
    const data = await getDashboardData();

    if (!data) return <div>Loading...</div>;

    const { teams, tournaments } = data;

    // Get current tournament (TEST_PVLR25)
    const currentTournament = tournaments?.find(t => t.code === "TEST_PVLR25");
    const simulation = currentTournament?.simulation;
    const pools = simulation?.pools;

    // Helper to find pool (only if pools exist)
    const getPool = (code: string) => {
        if (!pools) return null;
        if (pools.pool_a?.includes(code)) return "Pool A";
        if (pools.pool_b?.includes(code)) return "Pool B";
        return null;
    };

    return (
        <div className="space-y-8">
            <h1 className="text-3xl font-bold text-slate-900">Teams</h1>

            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                {teams.map((team) => {
                    const pool = getPool(team.code);
                    return (
                        <div key={team.id} className="bg-white rounded-lg shadow-md overflow-hidden hover:shadow-lg transition-shadow">
                            <div className="bg-indigo-600 px-6 py-4">
                                <div className="flex justify-between items-center">
                                    <h2 className="text-xl font-bold text-white">{team.name}</h2>
                                    <span className="bg-indigo-800 text-indigo-100 text-xs px-2 py-1 rounded-full">{team.code}</span>
                                </div>
                                {pool && <p className="text-indigo-200 text-sm mt-1">{pool}</p>}
                            </div>
                            <div className="p-6 space-y-4">
                                <div>
                                    <p className="text-xs text-slate-500 uppercase font-semibold">Coach</p>
                                    <p className="text-slate-800">{team.coach || "N/A"}</p>
                                </div>
                                {team.assistant_coach && (
                                    <div>
                                        <p className="text-xs text-slate-500 uppercase font-semibold">Assistant Coach</p>
                                        <p className="text-slate-800">{team.assistant_coach}</p>
                                    </div>
                                )}
                                <div className="pt-4 border-t border-slate-100 flex justify-end">
                                    {/* Placeholder for individual team details page link if we implement it */}
                                    <span className="text-indigo-600 text-sm font-medium cursor-not-allowed opacity-50">View Details</span>
                                </div>
                            </div>
                        </div>
                    );
                })}
            </div>
        </div>
    );
}
