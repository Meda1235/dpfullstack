import React, { useEffect, useState, useMemo, useCallback } from 'react';
import { useRouter } from 'next/router';
// Removed: import SidebarNav from '@/components/SidebarNav'; // Should be in layout

// Interfaces for clarity
interface StoredData {
    headers: string[];
    data: (string | number | null)[][];
    column_types?: { [key: string]: { type: string; missing: string } }; // From /api/analyze
    filled_mask?: boolean[][];
}

interface ColumnTypeInfo {
    name: string;
    type: 'Číselný' | 'Kategorie' | 'Neznámý'; // Keep consistent with ColumnAnalysis
}

// Helper to check if a value is considered "missing"
const isValueMissing = (value: any): boolean => {
    return value === null || value === '' || value === undefined || String(value).toLowerCase() === 'nan';
}

export default function MissingValues() {
    // --- State Definitions ---
    const [headers, setHeaders] = useState<string[]>([]);
    const [data, setData] = useState<(string | number | null)[][]>([]);
    const [columnTypeInfos, setColumnTypeInfos] = useState<ColumnTypeInfo[]>([]);
    const [filledMask, setFilledMask] = useState<boolean[][]>([]);
    const [methodMap, setMethodMap] = useState<Record<string, string>>({});
    const [showOnlyMissing, setShowOnlyMissing] = useState(false);
    const [globalMethod, setGlobalMethod] = useState<string>("mean");
    const [globalTarget, setGlobalTarget] = useState<'all' | 'numeric' | 'categorical'>("all");
    const [isLoading, setIsLoading] = useState<boolean>(true);
    const [isFilling, setIsFilling] = useState<boolean>(false);
    const [error, setError] = useState<string | null>(null);
    const [successMessage, setSuccessMessage] = useState<string | null>(null);
    const router = useRouter();

    // --- Fetch Initial Data ---
    const fetchData = useCallback(async () => {
        setIsLoading(true);
        setError(null);
        setSuccessMessage(null);
        try {
            const response = await fetch("http://localhost:8000/api/get_stored_data");
            if (!response.ok) throw new Error(`Nepodařilo se načíst data: ${response.statusText}`);
            const json: StoredData = await response.json();

            const fetchedHeaders = json.headers || [];
            const fetchedData = json.data || [];
            const fetchedTypes = json.column_types
                ? fetchedHeaders.map(h => ({
                    name: h,
                    type: json.column_types?.[h]?.type || 'Neznámý'
                })) as ColumnTypeInfo[]
                : fetchedHeaders.map(h => ({ name: h, type: 'Neznámý' })) as ColumnTypeInfo[];

            setHeaders(fetchedHeaders);
            setData(fetchedData);
            setColumnTypeInfos(fetchedTypes);

            const rows = fetchedData.length;
            const cols = fetchedHeaders.length;
            setFilledMask(
                json.filled_mask && json.filled_mask.length === rows && json.filled_mask[0]?.length === cols
                    ? json.filled_mask
                    : Array(rows).fill(0).map(() => Array(cols).fill(false))
            );

            const initialMethods: Record<string, string> = {};
            fetchedTypes.forEach(ct => {
                initialMethods[ct.name] = ct.type === "Číselný" ? "mean" : "mode";
            });
            setMethodMap(initialMethods);

        } catch (err: any) {
            console.error("Chyba při načítání dat:", err);
            setError(`Chyba při načítání dat: ${err.message}`);
            setHeaders([]);
            setData([]);
            setColumnTypeInfos([]);
            setFilledMask([]);
            setMethodMap({});
        } finally {
            setIsLoading(false);
        }
    }, []);

    useEffect(() => {
        fetchData();
    }, [fetchData]);

    // --- Derived State and Calculations (useMemo for performance) ---
    const missingCounts = useMemo(() => {
        const counts: Record<string, { count: number; percentage: number }> = {};
        const totalRows = data.length || 1;
        headers.forEach((header, colIndex) => {
            const count = data.reduce((acc, row) => acc + (isValueMissing(row?.[colIndex]) ? 1 : 0), 0);
            counts[header] = {
                count: count,
                percentage: (count / totalRows) * 100,
            };
        });
        return counts;
    }, [data, headers]);

    const hasAnyMissing = useMemo(() => {
        return Object.values(missingCounts).some(info => info.count > 0);
    }, [missingCounts]);

    const filteredDataIndices: number[] = useMemo(() => {
        if (!showOnlyMissing) {
            return data.map((_, index) => index);
        }
        return data.reduce((indices, row, rowIndex) => {
            const rowHasMissing = (row || []).some((cell, colIndex) =>
                isValueMissing(cell) || filledMask?.[rowIndex]?.[colIndex] === true
            );
            if (rowHasMissing) {
                indices.push(rowIndex);
            }
            return indices;
        }, [] as number[]); // ✅ TADY přidat `as number[]`

    }, [data, showOnlyMissing, filledMask]);

    // --- Event Handlers ---
    const handleMethodChange = (header: string, newMethod: string) => {
        setMethodMap(prev => ({ ...prev, [header]: newMethod }));
    };

    const handleApplyGlobal = useCallback(() => {
        setSuccessMessage(null);
        const updatedMethods = { ...methodMap };
        columnTypeInfos.forEach(ct => {
            const isNumeric = ct.type === "Číselný";
            const isCategorical = ct.type === "Kategorie";

            if (
                globalTarget === "all" ||
                (globalTarget === "numeric" && isNumeric) ||
                (globalTarget === "categorical" && isCategorical)
            ) {
                if (globalMethod === 'drop' || globalMethod === 'ffill' || globalMethod === 'bfill' ||
                    (isNumeric && ['mean', 'median', 'zero', 'interpolate'].includes(globalMethod)) ||
                    (isCategorical && globalMethod === 'mode'))
                {
                    updatedMethods[ct.name] = globalMethod;
                } else if (!isNumeric && !isCategorical && globalMethod !== 'drop') {
                    if (['mode', 'ffill', 'bfill'].includes(globalMethod)) {
                        updatedMethods[ct.name] = globalMethod;
                    }
                }
            }
        });
        setMethodMap(updatedMethods);
        setSuccessMessage(`Globální metoda '${globalMethod}' aplikována na ${globalTarget === 'all' ? 'všechny relevantní' : globalTarget} sloupce.`);
        setTimeout(() => setSuccessMessage(null), 3000);
    }, [methodMap, columnTypeInfos, globalTarget, globalMethod]);

    const handleFill = async () => {
        setIsFilling(true);
        setError(null);
        setSuccessMessage(null);

        const strategies: Record<string, string> = {};
        headers.forEach(h => {
            strategies[h] = methodMap[h] || (columnTypeInfos.find(ct => ct.name === h)?.type === "Číselný" ? 'mean' : 'mode');
        });

        console.log("📤 Strategie posílané na backend:", strategies);

        try {
            const response = await fetch('http://localhost:8000/api/fill_missing', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ strategies }),
            });
            const json = await response.json();

            if (!response.ok) {
                throw new Error(json.detail || `Chyba ${response.status}: ${response.statusText}`);
            }

            setHeaders(json.headers || []);
            setData(json.data || []);
            setFilledMask(json.filled_mask || []);

            // Update types and methods based on potentially modified headers/data
            const newHeaders = json.headers || [];
            const newColumnTypeInfos = newHeaders.map((h: string) =>
                columnTypeInfos.find(ct => ct.name === h) || { name: h, type: 'Neznámý' }
            );
            setColumnTypeInfos(newColumnTypeInfos);

            const newMethodMap: Record<string, string> = {};
            newHeaders.forEach((h: string) => {
                newMethodMap[h] = methodMap[h] || (
                    (newColumnTypeInfos.find((ct: ColumnTypeInfo) => ct.name === h)?.type === "Číselný")
                        ? 'mean'
                        : 'mode'
                );
            });
            setMethodMap(newMethodMap);


            setSuccessMessage(`Chybějící hodnoty byly úspěšně zpracovány (${json.filled_cells || 0} buněk doplněno/změněno).`);
            setTimeout(() => setSuccessMessage(null), 5000);


        } catch (err: any) {
            console.error("Chyba při doplňování hodnot:", err);
            setError(`Chyba při doplňování hodnot: ${err.message}`);
        } finally {
            setIsFilling(false);
        }
    };

    // --- Render Logic ---
    return (
        // Removed the outer <div className="flex ..."> assuming SidebarNav is handled by layout
        <main className="flex-1 p-6 space-y-6 bg-gray-100 dark:bg-gray-900 min-h-screen"> {/* Added background and min-height */}
            <h1 className="text-2xl font-bold text-gray-800 dark:text-gray-100">🧩 Chybějící hodnoty</h1>

            {/* --- Status Messages --- */}
            {isLoading && (
                <div className="flex items-center justify-center p-10 text-gray-500 dark:text-gray-400">
                    <svg className="animate-spin -ml-1 mr-3 h-5 w-5" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                        <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                        <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                    </svg>
                    <span>Načítání dat...</span>
                </div>
            )}
            {error && (
                <div className="p-3 bg-red-100 border border-red-300 text-red-800 rounded-md text-sm shadow-sm dark:bg-red-900/30 dark:border-red-700 dark:text-red-200" role="alert">
                    ❌ {error}
                </div>
            )}
            {successMessage && (
                <div className="p-3 bg-green-100 border border-green-300 text-green-800 rounded-md text-sm shadow-sm dark:bg-green-900/30 dark:border-green-700 dark:text-green-200">
                    ✅ {successMessage}
                </div>
            )}
            {!isLoading && !error && !hasAnyMissing && data.length > 0 && (
                <div className="p-3 bg-blue-100 border border-blue-300 text-blue-800 rounded-md text-sm shadow-sm dark:bg-blue-900/30 dark:border-blue-700 dark:text-blue-200">
                    ℹ️ V aktuálním datasetu nebyly nalezeny žádné chybějící hodnoty.
                </div>
            )}
            {!isLoading && !error && data.length === 0 && (
                <div className="p-3 bg-yellow-100 border border-yellow-300 text-yellow-800 rounded-md text-sm shadow-sm dark:bg-yellow-900/30 dark:border-yellow-700 dark:text-yellow-200">
                    ⚠️ Data nebyla nahrána nebo jsou prázdná. Nahrajte prosím data na úvodní stránce.
                </div>
            )}


            {/* --- Controls (only if data is loaded) --- */}
            {!isLoading && !error && data.length > 0 && (
                <>
                    <div className="flex flex-wrap items-center gap-4 mb-4 p-4 bg-white dark:bg-gray-800 rounded-lg shadow">
                        <label className="flex items-center gap-2 text-sm text-gray-700 dark:text-gray-300 cursor-pointer select-none">
                            <input
                                type="checkbox"
                                className="rounded border-gray-300 text-blue-600 shadow-sm focus:border-blue-300 focus:ring focus:ring-blue-200 focus:ring-opacity-50 dark:bg-gray-700 dark:border-gray-600"
                                checked={showOnlyMissing}
                                onChange={() => setShowOnlyMissing(!showOnlyMissing)}
                            />
                            Zobrazit pouze řádky s chybějícími/doplněnými hodnotami
                        </label>

                        <div className="flex flex-wrap items-center gap-2 border-l pl-4 dark:border-gray-600">
                            <span className="text-sm font-medium text-gray-600 dark:text-gray-400">Globální metoda:</span>
                            <select
                                value={globalMethod}
                                onChange={(e) => setGlobalMethod(e.target.value)}
                                className="border border-gray-300 px-2 py-1 text-sm rounded dark:bg-gray-700 dark:border-gray-600 dark:text-gray-200 focus:ring-blue-500 focus:border-blue-500"
                                aria-label="Vyberte globální metodu doplnění"
                            >
                                <option value="mean">Průměr (jen čísla)</option>
                                <option value="median">Medián (jen čísla)</option>
                                <option value="mode">Nejčastější (modus)</option>
                                <option value="zero">Nula (jen čísla)</option>
                                <option value="drop">Odstranit řádek</option>
                                <option value="ffill">Forward Fill</option>
                                <option value="bfill">Backward Fill</option>
                                <option value="interpolate">Interpolace (jen čísla)</option>
                            </select>
                            <select
                                value={globalTarget}
                                onChange={(e) => setGlobalTarget(e.target.value as any)}
                                className="border border-gray-300 px-2 py-1 text-sm rounded dark:bg-gray-700 dark:border-gray-600 dark:text-gray-200 focus:ring-blue-500 focus:border-blue-500"
                                aria-label="Vyberte cílové sloupce pro globální metodu"
                            >
                                <option value="all">Všechny sloupce</option>
                                <option value="numeric">Jen číselné</option>
                                <option value="categorical">Jen kategoriální</option>
                            </select>
                            <button
                                onClick={handleApplyGlobal}
                                className="bg-indigo-500 hover:bg-indigo-600 text-white px-3 py-1 rounded text-sm font-medium transition duration-150 ease-in-out disabled:opacity-60"
                                disabled={isFilling}
                                title="Aplikovat vybranou metodu na vybraný typ sloupců"
                            >
                                Aplikovat
                            </button>
                        </div>

                        <button
                            onClick={handleFill}
                            className="bg-green-600 hover:bg-green-700 text-white px-5 py-2 rounded shadow-sm font-semibold text-sm transition duration-150 ease-in-out disabled:opacity-60 flex items-center gap-2 ml-auto" // Added ml-auto
                            disabled={isFilling || isLoading} // Also disable if initial loading
                        >
                            {isFilling ? (
                                <>
                                    <svg className="animate-spin h-4 w-4 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                                        <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                                        <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                                    </svg>
                                    Zpracovávám...
                                </>
                            ) : (
                                'Doplnit / Zpracovat Chybějící Hodnoty'
                            )}
                        </button>
                    </div>

                    {/* --- Data Table --- */}
                    <div className="max-h-[65vh] overflow-auto border border-gray-200 dark:border-gray-700 rounded-lg shadow-md">
                        <table className="min-w-full text-sm divide-y divide-gray-200 dark:divide-gray-700 table-fixed"> {/* Added table-fixed */}
                            <colgroup> {/* Define column widths if needed */}
                                {headers.map(h => <col key={h} style={{ width: '150px' }} />)} {/* Example fixed width */}
                            </colgroup>
                            <thead className="bg-gray-100 dark:bg-gray-800 sticky top-0 z-10">
                            <tr>
                                {headers.map((header) => {
                                    const typeInfo = columnTypeInfos.find(ct => ct.name === header);
                                    const missingInfo = missingCounts[header];
                                    const currentMethod = methodMap[header] || '';
                                    const headerClass = typeInfo?.type === 'Číselný' ? 'text-blue-700 dark:text-blue-400' :
                                        typeInfo?.type === 'Kategorie' ? 'text-purple-700 dark:text-purple-400' :
                                            'text-gray-600 dark:text-gray-400';
                                    const borderColorClass = typeInfo?.type === 'Číselný' ? 'border-blue-200 dark:border-blue-700' :
                                        typeInfo?.type === 'Kategorie' ? 'border-purple-200 dark:border-purple-700' :
                                            'border-gray-200 dark:border-gray-700';

                                    return (
                                        <th key={header} scope="col" className={`px-3 py-2 text-left text-xs font-medium text-gray-600 dark:text-gray-300 uppercase tracking-wider border-b border-r ${borderColorClass}`}>
                                            <div className="flex flex-col space-y-1">
                                                <span className="font-semibold break-words">{header}</span> {/* Added break-words */}
                                                <span className={`text-xs font-normal ${headerClass}`}>({typeInfo?.type || 'N/A'})</span>
                                                <span className="text-xs text-gray-500 dark:text-gray-400">
                                                        Chybí: {missingInfo?.count ?? 0} ({missingInfo?.percentage?.toFixed(1) ?? 0}%)
                                                    </span>
                                                <select
                                                    value={currentMethod}
                                                    onChange={(e) => handleMethodChange(header, e.target.value)}
                                                    className="w-full text-xs mt-1 px-2 py-1 border border-gray-300 rounded bg-white dark:bg-gray-700 dark:border-gray-600 dark:text-gray-200 focus:ring-blue-500 focus:border-blue-500 disabled:opacity-70"
                                                    disabled={isFilling || isLoading} // Disable during loading too
                                                    aria-label={`Metoda doplnění pro ${header}`}
                                                >
                                                    <option value="mean" disabled={typeInfo?.type !== 'Číselný'}>Průměr</option>
                                                    <option value="median" disabled={typeInfo?.type !== 'Číselný'}>Medián</option>
                                                    <option value="mode">Nejčastější (modus)</option>
                                                    <option value="zero" disabled={typeInfo?.type !== 'Číselný'}>Nula</option>
                                                    <option value="drop">Odstranit řádek</option>
                                                    <option value="ffill">Forward Fill</option>
                                                    <option value="bfill">Backward Fill</option>
                                                    <option value="interpolate" disabled={typeInfo?.type !== 'Číselný'}>Interpolace</option>
                                                </select>
                                            </div>
                                        </th>
                                    );
                                })}
                            </tr>
                            </thead>
                            <tbody className="bg-white dark:bg-gray-900 divide-y divide-gray-200 dark:divide-gray-700">
                            {filteredDataIndices.length === 0 && (
                                <tr>
                                    <td colSpan={headers.length} className="px-6 py-4 text-center text-gray-500 dark:text-gray-400">
                                        {(showOnlyMissing && data.length > 0) ? 'Žádné řádky neodpovídají filtru zobrazení.' : 'Tabulka neobsahuje žádná data.'}
                                    </td>
                                </tr>
                            )}
                            {filteredDataIndices.map((rowIndex: number, idx: number) => {
                                const row = data[rowIndex];
                                if (!row || !Array.isArray(row)) {
                                    console.warn(`Skipping invalid row data at original index: ${rowIndex}`);
                                    return null;
                                }

                                return (
                                    <tr key={rowIndex} className="hover:bg-gray-50 dark:hover:bg-gray-800/50">
                                        {headers.map((_, colIndex) => { // Iterate based on headers count
                                            const cell = row[colIndex];
                                            const isOriginallyMissing = isValueMissing(cell);

                                            const isFilledByApp = !!(
                                                filledMask &&
                                                rowIndex >= 0 &&
                                                rowIndex < filledMask.length &&
                                                filledMask[rowIndex] &&
                                                colIndex >= 0 &&
                                                colIndex < filledMask[rowIndex].length &&
                                                filledMask[rowIndex][colIndex] === true
                                            );

                                            const cellKey = `${rowIndex}-${colIndex}`;
                                            const displayValue = isOriginallyMissing ? '—' : (typeof cell === 'number' || typeof cell === 'string' ? String(cell) : JSON.stringify(cell));

                                            let cellClasses = "px-3 py-1 whitespace-nowrap text-xs border-r dark:border-gray-700";
                                            if (isOriginallyMissing) {
                                                cellClasses += " text-red-500 dark:text-red-400 italic";
                                            } else {
                                                cellClasses += " text-gray-800 dark:text-gray-200";
                                            }
                                            if (isFilledByApp) {
                                                cellClasses += " bg-green-100 dark:bg-green-900/30 font-medium";
                                            }

                                            return (
                                                <td key={cellKey} className={cellClasses}>
                                                    {displayValue}
                                                </td>
                                            );
                                        })}
                                    </tr>
                                );
                            })}
                            </tbody>
                        </table>
                    </div>

                    {/* --- Navigation Button --- */}
                    <div className="mt-6 flex justify-end">
                        <button
                            className="bg-blue-600 hover:bg-blue-700 text-white px-4 py-2 rounded shadow-sm font-medium text-sm transition duration-150 ease-in-out disabled:opacity-60"
                            onClick={() => router.push('/preanalysis/outliers')}
                            disabled={isLoading || isFilling}
                        >
                            Pokračovat na analýzu odlehlých hodnot →
                        </button>
                    </div>
                </>
            )}
        </main>
        // Removed closing </div> that matched the removed flex div
    );
}