import React, { createContext, useContext, useState } from 'react';

type UploadedData = {
    headers: string[];
    rows: (string | number | null)[][];
    selectedColumns: boolean[];
};

type DataContextType = {
    uploadedData: UploadedData | null;
    setUploadedData: (data: UploadedData) => void;
};

const DataContext = createContext<DataContextType>({
    uploadedData: null,
    setUploadedData: () => {},
});

export const useDataContext = () => useContext(DataContext);

export const DataProvider = ({ children }: { children: React.ReactNode }) => {
    const [uploadedData, setUploadedData] = useState<UploadedData | null>(null);

    return (
        <DataContext.Provider value={{ uploadedData, setUploadedData }}>
            {children}
        </DataContext.Provider>
    );
};
