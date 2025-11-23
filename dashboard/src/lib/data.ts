import fs from 'fs/promises';
import path from 'path';
import { DashboardData } from '@/types';

export async function getDashboardData(): Promise<DashboardData | null> {
    try {
        const filePath = path.join(process.cwd(), 'public', 'data.json');
        const fileContents = await fs.readFile(filePath, 'utf8');
        return JSON.parse(fileContents);
    } catch (error) {
        console.error("Error reading data.json:", error);
        console.log("Current working directory:", process.cwd());
        console.log("Attempted path:", path.join(process.cwd(), 'public', 'data.json'));
        return null;
    }
}
