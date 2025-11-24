"use client";

import { useEffect, useState } from "react";
import { DashboardData } from "@/types";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { TrendingUp } from "lucide-react";
import PlayersTable from "@/components/PlayersTable";

export default function PlayersPage() {
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
                <div className="text-lg font-medium text-slate-600">Loading players...</div>
            </div>
        );
    }

    if (!data || !data.players) {
        return (
            <Card className="max-w-md mx-auto mt-20">
                <CardHeader>
                    <CardTitle className="text-red-600">Data not found</CardTitle>
                    <CardDescription>No player data available.</CardDescription>
                </CardHeader>
            </Card>
        );
    }

    return (
        <div className="container mx-auto p-4 space-y-8">
            <div className="text-center space-y-4 pb-8 border-b">
                <div className="flex items-center justify-center gap-3">
                    <TrendingUp className="h-10 w-10 text-indigo-600" />
                    <h1 className="text-4xl font-extrabold text-slate-900 tracking-tight">
                        Player Statistics
                    </h1>
                </div>
                <p className="text-lg text-slate-600">Individual player performance metrics and leaderboards</p>
                <Badge variant="outline" className="text-xs">
                    Last Updated: {new Date(data.last_updated).toLocaleString()}
                </Badge>
            </div>

            <PlayersTable players={data.players} teams={data.teams} />
        </div>
    );
}
