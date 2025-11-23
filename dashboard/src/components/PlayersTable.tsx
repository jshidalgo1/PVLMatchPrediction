'use client';

import { useState, useMemo } from 'react';
import { Player, Team } from '@/types';
import { ChevronDown, ChevronUp, Search } from 'lucide-react';

// Client component because it needs interactivity (sorting/filtering)
// We'll fetch data in a server component wrapper or just pass it down.
// For simplicity, let's make this a client component that takes data as props, 
// but since we can't easily pass data from a server page to a client page without prop drilling,
// we'll fetch it via an API route or just use a server component that renders a client component.

// Let's create the client component part here.

type SortField = keyof Player['stats'] | 'full_name' | 'team';
type SortDirection = 'asc' | 'desc';

interface PlayersTableProps {
    players: Player[];
    teams: Team[];
}

export default function PlayersTable({ players, teams }: PlayersTableProps) {
    const [searchTerm, setSearchTerm] = useState('');
    const [teamFilter, setTeamFilter] = useState('All');
    const [sortField, setSortField] = useState<SortField>('attack_points');
    const [sortDirection, setSortDirection] = useState<SortDirection>('desc');

    const handleSort = (field: SortField) => {
        if (sortField === field) {
            setSortDirection(sortDirection === 'asc' ? 'desc' : 'asc');
        } else {
            setSortField(field);
            setSortDirection('desc');
        }
    };

    const filteredPlayers = useMemo(() => {
        return players.filter(player => {
            const matchesSearch = player.full_name.toLowerCase().includes(searchTerm.toLowerCase());
            const matchesTeam = teamFilter === 'All' || player.teams.includes(teamFilter);
            return matchesSearch && matchesTeam;
        });
    }, [players, searchTerm, teamFilter]);

    const sortedPlayers = useMemo(() => {
        return [...filteredPlayers].sort((a, b) => {
            let aValue: any = a.stats[sortField as keyof typeof a.stats];
            let bValue: any = b.stats[sortField as keyof typeof b.stats];

            if (sortField === 'full_name') {
                aValue = a.full_name;
                bValue = b.full_name;
            } else if (sortField === 'team') {
                aValue = a.teams[0] || '';
                bValue = b.teams[0] || '';
            }

            if (aValue < bValue) return sortDirection === 'asc' ? -1 : 1;
            if (aValue > bValue) return sortDirection === 'asc' ? 1 : -1;
            return 0;
        });
    }, [filteredPlayers, sortField, sortDirection]);

    return (
        <div className="space-y-6">
            <div className="flex flex-col md:flex-row gap-4 justify-between items-center bg-white p-4 rounded-lg shadow">
                <div className="relative w-full md:w-64">
                    <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-slate-400 h-4 w-4" />
                    <input
                        type="text"
                        placeholder="Search players..."
                        className="pl-10 pr-4 py-2 w-full border border-slate-300 rounded-md focus:ring-indigo-500 focus:border-indigo-500"
                        value={searchTerm}
                        onChange={(e) => setSearchTerm(e.target.value)}
                    />
                </div>
                <div className="flex items-center gap-2 w-full md:w-auto">
                    <label className="text-sm font-medium text-slate-700">Team:</label>
                    <select
                        className="border border-slate-300 rounded-md px-3 py-2 focus:ring-indigo-500 focus:border-indigo-500"
                        value={teamFilter}
                        onChange={(e) => setTeamFilter(e.target.value)}
                    >
                        <option value="All">All Teams</option>
                        {teams.map(t => (
                            <option key={t.code} value={t.code}>{t.name} ({t.code})</option>
                        ))}
                    </select>
                </div>
            </div>

            <div className="bg-white rounded-lg shadow overflow-hidden">
                <div className="overflow-x-auto">
                    <table className="min-w-full divide-y divide-slate-200">
                        <thead className="bg-slate-50">
                            <tr>
                                <SortHeader label="Player" field="full_name" currentSort={sortField} currentDir={sortDirection} onSort={handleSort} />
                                <SortHeader label="Team" field="team" currentSort={sortField} currentDir={sortDirection} onSort={handleSort} />
                                <SortHeader label="Sets" field="sets_played" currentSort={sortField} currentDir={sortDirection} onSort={handleSort} />
                                <SortHeader label="Attack" field="attack_points" currentSort={sortField} currentDir={sortDirection} onSort={handleSort} />
                                <SortHeader label="Block" field="block_points" currentSort={sortField} currentDir={sortDirection} onSort={handleSort} />
                                <SortHeader label="Serve" field="serve_points" currentSort={sortField} currentDir={sortDirection} onSort={handleSort} />
                                <SortHeader label="Dig (Exc)" field="dig_excellent" currentSort={sortField} currentDir={sortDirection} onSort={handleSort} />
                                <SortHeader label="Rec (Exc)" field="reception_excellent" currentSort={sortField} currentDir={sortDirection} onSort={handleSort} />
                                <SortHeader label="Set (Exc)" field="set_excellent" currentSort={sortField} currentDir={sortDirection} onSort={handleSort} />
                            </tr>
                        </thead>
                        <tbody className="bg-white divide-y divide-slate-200">
                            {sortedPlayers.map((player) => (
                                <tr key={player.id} className="hover:bg-slate-50">
                                    <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-slate-900">{player.full_name}</td>
                                    <td className="px-6 py-4 whitespace-nowrap text-sm text-slate-500">{player.teams.join(', ')}</td>
                                    <td className="px-6 py-4 whitespace-nowrap text-sm text-slate-500">{player.stats.sets_played}</td>
                                    <td className="px-6 py-4 whitespace-nowrap text-sm text-slate-900 font-semibold">{player.stats.attack_points}</td>
                                    <td className="px-6 py-4 whitespace-nowrap text-sm text-slate-500">{player.stats.block_points}</td>
                                    <td className="px-6 py-4 whitespace-nowrap text-sm text-slate-500">{player.stats.serve_points}</td>
                                    <td className="px-6 py-4 whitespace-nowrap text-sm text-slate-500">{player.stats.dig_excellent}</td>
                                    <td className="px-6 py-4 whitespace-nowrap text-sm text-slate-500">{player.stats.reception_excellent}</td>
                                    <td className="px-6 py-4 whitespace-nowrap text-sm text-slate-500">{player.stats.set_excellent}</td>
                                </tr>
                            ))}
                        </tbody>
                    </table>
                </div>
                <div className="bg-slate-50 px-4 py-3 border-t border-slate-200 text-sm text-slate-500">
                    Showing {sortedPlayers.length} players
                </div>
            </div>
        </div>
    );
}

function SortHeader({ label, field, currentSort, currentDir, onSort }: { label: string, field: SortField, currentSort: SortField, currentDir: SortDirection, onSort: (f: SortField) => void }) {
    return (
        <th
            className="px-6 py-3 text-left text-xs font-medium text-slate-500 uppercase tracking-wider cursor-pointer hover:bg-slate-100 transition-colors select-none"
            onClick={() => onSort(field)}
        >
            <div className="flex items-center gap-1">
                {label}
                {currentSort === field && (
                    currentDir === 'asc' ? <ChevronUp className="h-3 w-3" /> : <ChevronDown className="h-3 w-3" />
                )}
            </div>
        </th>
    );
}
