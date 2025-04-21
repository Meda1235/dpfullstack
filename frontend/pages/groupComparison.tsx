// pages/group-comparison.tsx
import React from 'react';
import SidebarNav from '@/components/SidebarNav';
import GroupComparisonDialog from '@/components/GroupComparisonDialog';

export default function GroupComparisonPage() {
    return (
        <div className="flex">
            <SidebarNav />
            <main className="flex-1 p-6 space-y-6">
                <h1 className="text-2xl font-bold">🧪 Porovnání skupin</h1>
                <p className="text-gray-600 dark:text-gray-300">
                    Vyberte číselné a kategoriální proměnné, mezi jejichž skupinami chcete provést statistické porovnání.
                </p>

                <GroupComparisonDialog />
            </main>
        </div>
    );
}
