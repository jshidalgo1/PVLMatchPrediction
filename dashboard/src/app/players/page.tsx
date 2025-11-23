import { getDashboardData } from "@/lib/data";
import PlayersTable from "@/components/PlayersTable";

export const revalidate = 0;

export default async function PlayersPage() {
    const data = await getDashboardData();

    if (!data) return <div>Loading...</div>;

    const { players, teams } = data;

    return (
        <div className="space-y-8">
            <h1 className="text-3xl font-bold text-slate-900">Players</h1>
            <PlayersTable players={players} teams={teams} />
        </div>
    );
}
