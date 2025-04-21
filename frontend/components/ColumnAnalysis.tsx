// components/ColumnAnalysis.tsx
import React, { useEffect, useState, useCallback } from 'react';
import { useRouter } from 'next/router';

// --- Rozhraní ---
interface UploadedData {
    headers: string[];
    rows: (string | number | null)[][];
    selectedColumns: boolean[];
}

interface ColumnInfo {
    type: string; // 'Kategorie' nebo 'Číselný'
    missing: string; // Procentuální vyjádření, např. '10.50%'
}

interface ColumnAnalysisProps {
    data: UploadedData;
}

// --- Komponenta ---
export default function ColumnAnalysis({ data }: ColumnAnalysisProps) {
    // --- Stavy ---
    const [columnAnalysis, setColumnAnalysis] = useState<{ [key: string]: ColumnInfo }>({});
    const [llmTypeSuggestions, setLlmTypeSuggestions] = useState<{ [key: string]: string }>({});
    const [isLlmLoading, setIsLlmLoading] = useState(false);
    const [updateErrors, setUpdateErrors] = useState<{ [key: string]: string | null }>({}); // Chyby validace pro každý sloupec
    const [isAnalyzingInitial, setIsAnalyzingInitial] = useState(true); // Indikátor pro počáteční analýzu
    const router = useRouter();

    // --- Funkce pro počáteční analýzu dat (spouští se při načtení/změně dat) ---
    const analyzeData = useCallback(async (uploaded: UploadedData) => {
        setIsAnalyzingInitial(true);
        setUpdateErrors({}); // Reset chyb při nové analýze

        const filteredHeaders = uploaded.headers.filter((_, i) => uploaded.selectedColumns[i]);
        const filteredRows = uploaded.rows.map((row) => row?.filter((_, i) => uploaded.selectedColumns[i]));

        if (filteredHeaders.length === 0) {
            setColumnAnalysis({});
            setIsAnalyzingInitial(false);
            return;
        }

        try {
            // Zavoláme backend endpoint, který provede analýzu a vrátí typy a missing %
            const response = await fetch('http://localhost:8000/api/analyze', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    // Posíláme jen vybrané sloupce a řádky
                    headers: filteredHeaders,
                    data: filteredRows
                })
            });

            if (!response.ok) {
                throw new Error(`Chyba při analýze dat na serveru: ${response.statusText}`);
            }

            const analysisResult = await response.json();

            if (analysisResult && analysisResult.column_types) {
                // Uložíme výsledky analýzy z backendu do stavu
                setColumnAnalysis(analysisResult.column_types);
            } else {
                // Fallback nebo logování chyby, pokud backend nevrátí očekávaný formát
                console.error("Backend /api/analyze nevrátil očekávaný formát 'column_types'");
                setColumnAnalysis({}); // Resetovat stav, pokud data nejsou validní
            }

        } catch (error) {
            console.error("Chyba při volání /api/analyze:", error);
            // Zde můžete nastavit nějaký chybový stav pro UI
            setColumnAnalysis({}); // Resetovat stav při chybě
        } finally {
            setIsAnalyzingInitial(false);
        }
    }, []); // Prázdné pole závislostí, pokud 'data' prop je stabilní nebo pokud chcete re-analýzu jen při mountnutí

    // --- Efekt pro spuštění analýzy při změně dat ---
    useEffect(() => {
        // Zkontrolujte, zda jsou data platná před spuštěním analýzy
        if (data && data.headers && Array.isArray(data.headers) && data.rows && Array.isArray(data.rows)) {
            analyzeData(data);
        } else {
            // Pokud data nejsou platná, resetujte stav
            setColumnAnalysis({});
            setIsAnalyzingInitial(false);
        }
    }, [data, analyzeData]); // Spustit znovu, pokud se změní `data` nebo `analyzeData`

    // --- Funkce pro analýzu typů pomocí LLM (nemění se) ---
    const analyzeWithLLM = async () => {
        setIsLlmLoading(true);
        setLlmTypeSuggestions({});
        setUpdateErrors({}); // Resetovat chyby i zde

        const filteredHeaders = data.headers.filter((_, i) => data.selectedColumns[i]);
        const filteredRows = data.rows.map((row) => row?.filter((_, i) => data.selectedColumns[i])); // Přidána kontrola na null/undefined řádek

        if (filteredHeaders.length === 0) {
            setIsLlmLoading(false);
            return; // Nic k analýze
        }

        try {
            const response = await fetch('http://127.0.0.1:8000/api/analyze_with_llm', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ headers: filteredHeaders, data: filteredRows }),
            });

            if (!response.ok || !response.body) {
                throw new Error(`Chyba při komunikaci s LLM API: ${response.statusText}`);
            }

            // ... (zpracování streamu jako dříve) ...
            const reader = response.body.getReader();
            const decoder = new TextDecoder();
            let buffer = '';
            let newSuggestions: { [key: string]: string } = {};
            let currentIndex = 0;

            while (true) {
                const { done, value } = await reader.read();
                const chunk = decoder.decode(value || new Uint8Array(), { stream: true });
                buffer += chunk;

                // Zpracování kompletních zpráv oddělených středníkem
                const parts = buffer.split(';');
                buffer = parts.pop() || ''; // Poslední část může být nekompletní

                for (const part of parts) {
                    const trimmedPart = part.trim().toLowerCase();
                    if (trimmedPart && currentIndex < filteredHeaders.length) {
                        const symbol = trimmedPart.replace(/\s+/g, '');
                        const typ = symbol === 'c' ? 'Číselný' : 'Kategorie'; // Předpoklad 'c' pro číselný, 'k' pro kategorii
                        const colName = filteredHeaders[currentIndex];
                        newSuggestions[colName] = typ;
                        currentIndex++;
                    }
                }
                setLlmTypeSuggestions({ ...newSuggestions }); // Aktualizace UI průběžně

                if (done) break;
            }

            // Zpracování posledního zbytku v bufferu
            if (buffer.trim() && currentIndex < filteredHeaders.length) {
                const symbol = buffer.trim().toLowerCase().replace(/\s+/g, '');
                const typ = symbol === 'c' ? 'Číselný' : 'Kategorie';
                const colName = filteredHeaders[currentIndex];
                newSuggestions[colName] = typ;
                setLlmTypeSuggestions({ ...newSuggestions });
            }


        } catch (error) {
            console.error("Chyba při LLM analýze typů:", error);
            alert(`Chyba při komunikaci s LLM: ${error instanceof Error ? error.message : String(error)}`);
        } finally {
            setIsLlmLoading(false);
        }
    };

    // --- Handler pro změnu typu v comboboxu (s validací na backendu) ---
    const handleTypeChange = useCallback(async (header: string, newType: string) => {
        const originalType = columnAnalysis[header]?.type;

        // Nedělat nic, pokud typ neexistuje nebo se nemění
        if (!originalType || originalType === newType) {
            return;
        }

        // 1. Optimistická aktualizace UI (pro lepší UX)
        setColumnAnalysis((prev) => {
            // Zajistíme, že pracujeme s kopií, abychom nemutovali přímo stav
            const newState = { ...prev };
            if (newState[header]) {
                newState[header] = { ...newState[header], type: newType };
            }
            return newState;
        });
        setUpdateErrors((prev) => ({ ...prev, [header]: null })); // Smazat předchozí chybu pro tento sloupec

        // 2. Volání backendu pro validaci a uložení
        try {
            const response = await fetch('http://localhost:8000/api/validate_and_update_column_type', { // Nový validační endpoint
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    column: header,
                    newType: newType,
                }),
            });

            if (!response.ok) {
                // Backend validace selhala
                const errorData = await response.json();
                const errorMessage = errorData.detail || `Server vrátil chybu ${response.status}`;
                console.error(`Chyba validace typu pro '${header}' na '${newType}':`, errorMessage);

                // a) Zobrazit chybu u sloupce
                setUpdateErrors((prev) => ({ ...prev, [header]: errorMessage }));
                // b) Vrátit UI na původní typ
                setColumnAnalysis((prev) => {
                    const newState = { ...prev };
                    if (newState[header]) {
                        newState[header] = { ...newState[header], type: originalType };
                    }
                    return newState;
                });
            } else {
                // Vše OK, backend potvrdil a uložil změnu
                console.log(`Typ pro sloupec '${header}' úspěšně ověřen a změněn na '${newType}' na backendu.`);
                // Optimistické UI už je správně, není třeba další akce
            }
        } catch (error) {
            // Síťová chyba nebo jiný problém s fetch
            const errorMessage = error instanceof Error ? error.message : String(error);
            console.error(`Síťová chyba při aktualizaci typu pro '${header}':`, errorMessage);
            setUpdateErrors((prev) => ({ ...prev, [header]: `Chyba komunikace: ${errorMessage}` }));
            // Vrátit UI na původní typ
            setColumnAnalysis((prev) => {
                const newState = { ...prev };
                if (newState[header]) {
                    newState[header] = { ...newState[header], type: originalType };
                }
                return newState;
            });
        }
    }, [columnAnalysis]); // Závislost na columnAnalysis pro přístup k originalType

    // --- Příprava dat pro zobrazení ---
    // Filtrujeme hlavičky a řádky POUZE JEDNOU zde pro konzistenci
    const visibleHeaders = data.headers.filter((_, i) => data.selectedColumns[i]);
    const visibleRows = data.rows
        .map((row) => row?.filter((_, i) => data.selectedColumns[i])) // Přidána kontrola na null/undefined řádek
        .filter(row => row !== undefined); // Odstranit případné undefined řádky


    // Zjištění, zda existují chybějící hodnoty pro tlačítko "Pokračovat"
    const hasMissingValues = Object.values(columnAnalysis).some((col) => parseFloat(col.missing) > 0);

    // --- Renderování ---
    if (isAnalyzingInitial) {
        return <div className="text-center p-6 text-gray-600 dark:text-gray-400">Analyzuji strukturu dat...</div>;
    }
    if (visibleHeaders.length === 0) {
        return <div className="text-center p-6 text-gray-600 dark:text-gray-400">Žádné sloupce nebyly vybrány nebo data nebyla nahrána.</div>;
    }

    return (
        <div className="p-4 md:p-6 bg-white dark:bg-gray-900 rounded-lg shadow">
            {/* --- LLM Sekce --- */}
            <div className="mb-6 flex items-center gap-4">
                <button
                    onClick={analyzeWithLLM}
                    disabled={isLlmLoading}
                    className="px-4 py-2 rounded bg-indigo-600 text-white hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:ring-offset-2 disabled:opacity-50 disabled:cursor-not-allowed transition duration-150 ease-in-out dark:focus:ring-offset-gray-900"
                >
                    {isLlmLoading ? 'Analyzuji...' : '🔍 Analyzovat typy pomocí AI'}
                </button>
                {isLlmLoading && (
                    <div className="text-sm text-indigo-600 dark:text-indigo-400 animate-pulse">
                        Probíhá AI analýza...
                    </div>
                )}
            </div>


            {/* --- Tabulka s analýzou a daty --- */}
            <div className="max-h-[60vh] overflow-auto border border-gray-200 dark:border-gray-700 rounded shadow-sm mb-6 relative"> {/* Omezení výšky, relativní pozice pro sticky thead */}
                <table className="min-w-full text-sm divide-y divide-gray-200 dark:divide-gray-700">
                    <thead className="bg-gray-100 dark:bg-gray-800 sticky top-0 z-10"> {/* Sticky thead */}
                    <tr>
                        {visibleHeaders.map((header) => (
                            <th key={header} className="px-3 py-2.5 border-b border-gray-200 dark:border-gray-700 text-left text-xs font-semibold text-gray-600 dark:text-gray-300 uppercase tracking-wider">
                                {header}
                            </th>
                        ))}
                    </tr>
                    <tr className="bg-gray-50 dark:bg-gray-700 border-t border-gray-200 dark:border-gray-600">
                        {/* Řádek pro ovládací prvky typů */}
                        {visibleHeaders.map((header) => (
                            <td key={`${header}-controls`} className={`px-3 py-2 border-b border-gray-200 dark:border-gray-600 align-top ${updateErrors[header] ? 'outline outline-2 outline-red-500 outline-offset-[-1px]' : ''}`}> {/* Zvýraznění chyby */}
                                {!columnAnalysis[header] ? (
                                    <div className="text-xs text-gray-400 dark:text-gray-500">Načítání...</div>
                                ) : (
                                    <div className="text-xs text-gray-700 dark:text-gray-300 space-y-1">
                                        <div className="font-medium">Chybí: <span className="text-gray-900 dark:text-gray-100">{columnAnalysis[header].missing}</span></div>
                                        <div>
                                            <label htmlFor={`type-${header}`} className="block text-xs font-medium text-gray-500 dark:text-gray-400 mb-0.5">Typ:</label>
                                            <select
                                                id={`type-${header}`}
                                                value={columnAnalysis[header].type}
                                                onChange={(e) => handleTypeChange(header, e.target.value)}
                                                className={`w-full p-1 border rounded text-xs bg-white dark:bg-gray-800 text-gray-900 dark:text-gray-100 focus:outline-none focus:ring-1 focus:ring-indigo-500 ${updateErrors[header] ? 'border-red-500 dark:border-red-600 focus:border-red-500 focus:ring-red-500' : 'border-gray-300 dark:border-gray-600 focus:border-indigo-500'}`}
                                            >
                                                <option value="Kategorie">Kategorie</option>
                                                <option value="Číselný">Číselný</option>
                                            </select>
                                        </div>

                                        {/* Zobrazení validační chyby */}
                                        {updateErrors[header] && (
                                            <div className="text-xs text-red-600 dark:text-red-400 pt-0.5">
                                                {updateErrors[header]}
                                            </div>
                                        )}

                                        {/* Zobrazení návrhu od LLM */}
                                        {llmTypeSuggestions[header] && (
                                            <div className="text-xs text-purple-600 dark:text-purple-400 pt-0.5">
                                                AI návrh: {llmTypeSuggestions[header]}
                                            </div>
                                        )}
                                    </div>
                                )}
                            </td>
                        ))}
                    </tr>
                    </thead>
                    <tbody className="bg-white dark:bg-gray-900 divide-y divide-gray-200 dark:divide-gray-700">
                    {/* Zobrazení prvních N řádků dat */}
                    {visibleRows.slice(0, 50).map((row, ri) => ( // Omezit počet zobrazených řádků pro výkon
                        <tr key={ri} className={ri % 2 === 0 ? 'bg-white dark:bg-gray-800/50' : 'bg-gray-50 dark:bg-gray-800'}>
                            {row?.map((cell, ci) => ( // Přidána kontrola na null/undefined řádek
                                <td key={ci} className="px-3 py-1.5 border-r border-gray-200 dark:border-gray-700 whitespace-nowrap overflow-hidden text-ellipsis max-w-[150px] text-xs text-gray-700 dark:text-gray-300" title={String(cell)}>
                                    {cell === null || cell === undefined ? <span className="italic text-gray-400 dark:text-gray-500">null</span> : String(cell)}
                                </td>
                            ))}
                            {/* Zajistit, aby řádek měl správný počet buněk, i když jsou data nekonzistentní */}
                            {(!row || row.length < visibleHeaders.length) &&
                                Array.from({ length: visibleHeaders.length - (row?.length || 0) }).map((_, idx) => (
                                    <td key={`empty-${idx}`} className="px-3 py-1.5 border-r border-gray-200 dark:border-gray-700"></td>
                                ))
                            }
                        </tr>
                    ))}
                    {visibleRows.length > 50 && (
                        <tr>
                            <td colSpan={visibleHeaders.length} className="px-3 py-2 text-center text-xs text-gray-500 dark:text-gray-400 italic">
                                ... a dalších {visibleRows.length - 50} řádků (zobrazeno prvních 50)
                            </td>
                        </tr>
                    )}
                    {visibleRows.length === 0 && (
                        <tr>
                            <td colSpan={visibleHeaders.length} className="px-3 py-4 text-center text-sm text-gray-500 dark:text-gray-400">
                                Dataset neobsahuje žádná data (po filtrování sloupců).
                            </td>
                        </tr>
                    )}
                    </tbody>
                </table>
            </div>

            {/* --- Tlačítko Pokračovat --- */}
            <div className="mt-6 text-right">
                <button
                    onClick={() =>
                        router.push(hasMissingValues ? '/preanalysis/missing' : '/preanalysis/outliers')
                    }
                    className="px-5 py-2.5 rounded bg-emerald-600 text-white hover:bg-emerald-700 focus:outline-none focus:ring-2 focus:ring-emerald-500 focus:ring-offset-2 dark:focus:ring-offset-gray-900 transition duration-150 ease-in-out"
                >
                    Pokračovat na analýzu {hasMissingValues ? 'chybějících hodnot' : 'odlehlých hodnot'} {'->'}
                </button>
            </div>
        </div>
    );
}