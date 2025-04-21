// pages/preanalysis/missing.tsx
import React from 'react';
import SidebarNav from '../../components/SidebarNav';
import MissingValues from '../../components/MissingValues';

export default function MissingPage() {
    return (
        <div className="flex">
            <SidebarNav />
            <main className="flex-1 p-6">
                <MissingValues />
            </main>
        </div>
    );
}
