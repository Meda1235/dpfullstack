// pages/relationships.tsx
import React from 'react';
import SidebarNavigation from '../components/SidebarNav';
import VariableRelationshipDialog from '../components/VariableRelationshipDialog';

export default function RelationshipsPage() {
    return (
        <div className="flex min-h-screen bg-gray-50 dark:bg-gray-950 text-gray-800 dark:text-gray-100">
            <SidebarNavigation />

            <main className="flex-1 p-6 overflow-y-auto">
                <VariableRelationshipDialog />
            </main>
        </div>
    );
}
