// pages/preanalysis/columns.tsx
import React, { useEffect, useState } from 'react';
import ColumnAnalysis from '@/components/ColumnAnalysis';
import SidebarNav from '@/components/SidebarNav';

interface UploadedData {
    headers: string[];
    rows: (string | number | null)[][];
    selectedColumns: boolean[];
}

export default function ColumnAnalysisPage() {
    const [data, setData] = useState<UploadedData | null>(null);

    useEffect(() => {
        fetch('http://localhost:8000/api/get_stored_data')
            .then((res) => res.json())
            .then((res) => {
                if (res && res.headers && res.data) {
                    setData({
                        headers: res.headers,
                        rows: res.data,
                        selectedColumns: res.headers.map(() => true),
                    });
                }
            });
    }, []);

    if (!data) {
        return (
            <div className="flex">
                <SidebarNav />
                <div className="p-6 text-gray-500 w-full">Načítání dat...</div>
            </div>
        );
    }

    return (
        <div className="flex">
            <SidebarNav />
            <div className="flex-1 p-6">
                <ColumnAnalysis data={data} />
            </div>
        </div>
    );
}
