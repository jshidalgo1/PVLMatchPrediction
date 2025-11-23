import type { Metadata } from "next";
import { Inter } from "next/font/google";
import "./globals.css";
import Link from "next/link";

const inter = Inter({ subsets: ["latin"] });

export const metadata: Metadata = {
    title: "PVL Match Prediction Dashboard",
    description: "Simulation results and statistics for PVL Reinforced Conference 2025",
};

export default function RootLayout({
    children,
}: Readonly<{
    children: React.ReactNode;
}>) {
    return (
        <html lang="en">
            <body className={`${inter.className} bg-slate-50 text-slate-900 min-h-screen`}>
                <nav className="bg-indigo-600 text-white shadow-lg">
                    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
                        <div className="flex items-center justify-between h-16">
                            <div className="flex items-center">
                                <Link href="/" className="font-bold text-xl tracking-tight">
                                    PVL Predictor
                                </Link>
                                <nav className="ml-10 flex gap-6">
                                    <Link
                                        href="/"
                                        className="text-white hover:text-slate-200 transition-colors font-medium"
                                    >
                                        Simulation
                                    </Link>
                                    <Link
                                        href="/history"
                                        className="text-white hover:text-slate-200 transition-colors font-medium"
                                    >
                                        History
                                    </Link>
                                    <Link
                                        href="/teams"
                                        className="text-white hover:text-slate-200 transition-colors font-medium"
                                    >
                                        Teams
                                    </Link>
                                    <Link
                                        href="/players"
                                        className="text-white hover:text-slate-200 transition-colors font-medium"
                                    >
                                        Players
                                    </Link>
                                </nav>
                            </div>
                        </div>
                    </div>
                </nav>
                <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
                    {children}
                </main>
            </body>
        </html >
    );
}
