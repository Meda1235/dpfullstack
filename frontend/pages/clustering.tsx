// pages/cluster.tsx
import React from 'react';
import SidebarNavigation from '../components/SidebarNav';
import ClusterAnalysisDialog from '../components/ClusterAnalysis';

export default function ClusterPage() {
    return (
        <div className="flex min-h-screen bg-gray-50 dark:bg-gray-950 text-gray-800 dark:text-gray-100">
            <SidebarNavigation />

            <main className="flex-1 p-6 overflow-y-auto">
                <ClusterAnalysisDialog />
            </main>
        </div>
    );
}
